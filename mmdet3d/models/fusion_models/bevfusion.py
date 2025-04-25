import copy
import io
from PIL import Image
from typing import Any, Dict

import cv2
from matplotlib import pyplot as plt
import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils.visualize import MAP_PALETTE, OBJECT_PALETTE
import mmcv
import numpy as np
from typing import List, Optional, Tuple
from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        #self.gru_concat_target_point = waypoint_config["gru_concat_target_point"]
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.summary_writer = SummaryWriter(log_dir="/home/bevfusion/tf_logs")
        self.object_classes = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone"
        ]
        self.point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        iter,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_trajectory,
        gt_trajectory_speed,
        goal_point,
        goal_point_speed,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                iter,
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_trajectory,
                gt_trajectory_speed,
                goal_point,
                goal_point_speed,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs
        
    def visualize_camera(
        self,
        image: np.ndarray,
        *,
        bboxes: Optional[LiDARInstance3DBoxes] = None,
        labels: Optional[np.ndarray] = None,
        transform: Optional[np.ndarray] = None,
        classes: Optional[List[str]] = None,
        color: Optional[Tuple[int, int, int]] = None,
        thickness: float = 4,
        index_cam: int 
    ) -> None:
        canvas = image.copy()
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

       
        PALETTE = OBJECT_PALETTE

        if bboxes is not None and len(bboxes) > 0:
            corners = bboxes.corners
            num_bboxes = corners.shape[0]

            coords = np.concatenate(
                [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
            )
            transform = copy.deepcopy(transform).reshape(4, 4)
            coords = coords @ transform.T
            coords = coords.reshape(-1, 8, 4)

            indices = np.all(coords[..., 2] > 0, axis=1)
            coords = coords[indices]
            labels = labels[indices]

            indices = np.argsort(-np.min(coords[..., 2], axis=1))
            coords = coords[indices]
            labels = labels[indices]

            coords = coords.reshape(-1, 4)
            coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
            coords[:, 0] /= coords[:, 2]
            coords[:, 1] /= coords[:, 2]

            coords = coords[..., :2].reshape(-1, 8, 2)
            for index in range(coords.shape[0]):
                name = classes[labels[index]]
                for start, end in [
                    (0, 1),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 5),
                    (3, 2),
                    (3, 7),
                    (4, 5),
                    (4, 7),
                    (2, 6),
                    (5, 6),
                    (6, 7),
                ]:
                    cv2.line(
                        canvas,
                        coords[index, start].astype(np.int),
                        coords[index, end].astype(np.int),
                        color or PALETTE[name],
                        thickness,
                        cv2.LINE_AA,
                    )
            canvas = canvas.astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        self.summary_writer.add_image(
            f"train/sample_rgb_hwc_{index_cam}",
            canvas,                    # still a NumPy H×W×C array (uint8 or float)
            global_step=0,
            dataformats="HWC"
        )

        #self.summary_writer.flush()


    def visualize_lidar(
        self,
        lidar: Optional[np.ndarray] = None,
        *,
        bboxes: Optional[LiDARInstance3DBoxes] = None,
        labels: Optional[np.ndarray] = None,
        classes: Optional[List[str]] = None,
        xlim: Tuple[float, float] = (-50, 50),
        ylim: Tuple[float, float] = (-50, 50),
        color: Optional[Tuple[int, int, int]] = None,
        radius: float = 15,
        thickness: float = 2
    ) -> None:
        """
        Logs a top‐down LiDAR + 3D-box plot into TensorBoard.
        
        Args:
            writer:       an initialized torch.utils.tensorboard.SummaryWriter
            tag:          the name under which to log this figure
            global_step:  the step index for TensorBoard
            lidar:        (N,3) point cloud
            bboxes:       LiDARInstance3DBoxes
            labels:       (M,) integer class indices
            classes:      list mapping label → class name
            xlim, ylim:   plot extents
            color:        override color for all boxes (RGB tuple 0–255)
            radius:       point‐size for scatter
            thickness:    line width for boxes
            mode:         if contains "simbev" picks alternate palette
        """
        # 1) build the figure
        fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

        ax = fig.gca()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(1)
        ax.axis("off")
        

        PALETTE = OBJECT_PALETTE

        # 2) plot points
        if lidar is not None:
            ax.scatter(
                lidar[:, 0], lidar[:, 1],
                s=radius, c="white"
            )

        # 3) plot boxes
        if bboxes is not None and len(bboxes) > 0:
            # corner ordering: front-left, front-right, back-right, back-left, repeat front-left
            coords = bboxes.corners[:, [0,3,7,4,0], :2]
            for index in range(coords.shape[0]):
                name = classes[labels[index]]
                plt.plot(
                    coords[index, :, 0],
                    coords[index, :, 1],
                    linewidth=thickness,
                    color=np.array(color or PALETTE[name]) / 255,
                )

        buf = io.BytesIO()
        plt.savefig(
            buf,                 # Save to the buffer object
            dpi=10,              # Low DPI as in your example (will look pixelated)
            facecolor="black",   # Background color of the figure *outside* the axes
            format="png",        # Explicitly set format
            bbox_inches="tight", # Adjust bounding box
            pad_inches=0         # Padding around the bounding box
        )
        buf.seek(0)

        img = Image.open(buf)
        # Convert the PIL image to a NumPy array
        img_array = np.array(img)

        if img_array.shape[2] == 4: # Check if it has 4 channels (RGBA)
            img_array = img_array[:, :, :3]
        # 4) log to TensorBoard
        self.summary_writer.add_image("train/sample_lidar", img_array,global_step=0, dataformats='HWC')
        buf.close()
        #self.summary_writer.flush()
        plt.close(fig)


    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        iter,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_trajectory,
        gt_trajectory_speed,
        goal_point,
        goal_point_speed,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                elif type == "planner":
                    losses = head(x, goal_point, gt_trajectory,gt_trajectory_speed,goal_point_speed)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                                "gt_bboxes_3d": gt_bboxes_3d[0].to("cpu"),
                                "gt_labels_3d": gt_labels_3d[0].cpu()
                            }
                        )
                    if iter % 50 == 0:
                        bboxes_viz = outputs[0]["boxes_3d"].tensor.numpy()
                        scores_viz = outputs[0]["scores_3d"].numpy()
                        labels_viz = outputs[0]["labels_3d"].numpy()

                        indices = scores_viz >= 0.1
                        bboxes_viz = bboxes_viz[indices]
                        scores_viz = scores_viz[indices]
                        labels_viz = labels_viz[indices]

                        bboxes_viz[..., 2] -= bboxes_viz[..., 5] / 2
                        bboxes_viz = LiDARInstance3DBoxes(bboxes_viz, box_dim=9)
                        #img_viz = img[0]
                        for k, image_path in enumerate(metas[0]["filename"]):
                            image = mmcv.imread(image_path)
                            self.visualize_camera(
                                image,
                                bboxes=bboxes_viz,
                                labels=labels_viz,
                                transform=metas[0]["lidar2image"][k],
                                classes=self.object_classes,
                                thickness = 2,
                                index_cam = k
                            )

                        lidar_viz = points[0].cpu().numpy()
                        self.visualize_lidar(
                            lidar_viz,
                            bboxes=bboxes_viz,
                            labels=labels_viz,
                            xlim=[self.point_cloud_range[d] for d in [0, 3]],
                            ylim=[self.point_cloud_range[d] for d in [1, 4]],
                            classes=self.object_classes,
                            thickness = 12
                        )
                    
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                elif type == "planner":
                    pred_wp = head(x, goal_point, gt_trajectory,gt_trajectory_speed,goal_point_speed)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "pred_waypoints" : pred_wp[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
