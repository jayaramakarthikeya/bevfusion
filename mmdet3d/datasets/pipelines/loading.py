import os
from typing import Any, Dict, Tuple

import mmcv
import torch
import numpy as np
import torch.nn.functional as F
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from PIL import Image


from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams, load_points, trim_points


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles:
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["image_paths"]
        # img is of shape (h, w, c, num_views)
        # modified for waymo
        images = []
        h, w = 0, 0
        for name in filename:
            images.append(Image.open(name))
        
        #TODO: consider image padding in waymo

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = images
        # [1600, 900]
        results["img_shape"] = images[0].size
        results["ori_shape"] = images[0].size
        # Set initial values for default meta_keys
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@PIPELINES.register_module()
class LoadBEVSegmentation:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        location = data["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(np.bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        data["gt_masks_bev"] = labels
        return data


@PIPELINES.register_module()
class LoadSimBEVBEVSegmentation:
    '''
    Load SimBEV BEV segmentation masks.

    Args:
        is_train: whether the data is for training.
        bev_res_x: BEV grid cell length along the x axis.
        bev_dim_x: BEV grid dimension along the x axis.
        bev_res_y: BEV grid cell length along the y axis.
        bev_dim_y: BEV grid dimension along the y axis.
        dType: data type to use for calculations.
    '''

    def __init__(self, is_train, bev_res_x, bev_dim_x, bev_res_y=None, bev_dim_y=None, dType=torch.float):
        self.is_train = is_train
        
        if bev_res_y is None:
            bev_res_y = bev_res_x
        
        if bev_dim_y is None:
            bev_dim_y = bev_dim_x
        
        self.xRes = bev_res_x
        self.yRes = bev_res_y
        
        self.DxDim = bev_dim_x
        self.DyDim = bev_dim_y
        
        self.dType = dType

    def __call__(self, data):
        # Load BEV ground truth.
        gt_seg_path = data['gt_seg_path']

        mmcv.check_file_exist(gt_seg_path)

        if gt_seg_path.endswith('.npz'):
            gt_masks = np.load(gt_seg_path)['data']
        else:
            gt_masks = np.load(gt_seg_path)
        
        gt_masks = np.rot90(gt_masks, 2, axes=(2, 1)).copy()

        # car_mask = gt_masks[1]
        # truck_mask = np.logical_or(gt_masks[2], gt_masks[3])
        # cyclist_mask = np.logical_or.reduce((gt_masks[4], gt_masks[5], gt_masks[6]))
        # pedestrian_mask = gt_masks[7]

        # road_mask = np.logical_and(
        #     gt_masks[0],
        #     np.logical_not(np.logical_or.reduce((car_mask, truck_mask, cyclist_mask, pedestrian_mask)))
        # )

        # gt_masks = np.array([road_mask, car_mask, truck_mask, cyclist_mask, pedestrian_mask])

        if self.is_train:
            gt_masks = torch.from_numpy(gt_masks).to(self.dType)
            
            # Calculate transformation matrix.
            lidar2point = data['lidar_aug_matrix']

            lidar2point[:3, :3] = lidar2point[:3, :3].T

            point2lidar = np.linalg.inv(lidar2point)

            lidar2ego = data['lidar2ego']

            point2ego = lidar2ego @ point2lidar

            R = torch.tensor(point2ego[[0, 1, 3], :][:, [0, 1, 3]], dtype=self.dType)

            xDim = gt_masks.shape[-2]
            yDim = gt_masks.shape[-1]

            xLim = xDim * self.xRes / 2
            yLim = yDim * self.yRes / 2

            R[:2, 2] /= torch.Tensor([xLim, yLim]).to(self.dType)

            R[:2, 2] = R[:2, 2].flip(dims=(0,))

            angle = torch.atan2(R[1, 0], R[0, 0])

            R[:2, 2] = torch.Tensor(
                [[torch.cos(angle), -torch.sin(angle) * (xLim / yLim)],
                [torch.sin(angle) * (yLim / xLim), torch.cos(angle)]]
            ).to(self.dType) @ R[:2, 2]

            R[:2, :2] *= torch.Tensor([[1, (xLim / yLim)], [(yLim / xLim), 1]]).to(self.dType)

            theta = torch.unsqueeze(R[:2, :], 0)

            unsqueezed_gt_masks = torch.unsqueeze(gt_masks, 0)

            grid = F.affine_grid(theta, unsqueezed_gt_masks.size(), align_corners=False)

            new_gt_mask_nearest = F.grid_sample(
                unsqueezed_gt_masks,
                grid,
                mode='nearest',
                align_corners=False
            ).squeeze(0).to(torch.bool)
            
            new_gt_mask_bilinear = F.grid_sample(
                unsqueezed_gt_masks,
                grid,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            new_gt_mask = new_gt_mask_nearest.detach().clone().cpu().numpy()
            new_gt_mask[4:] = np.logical_or(
                new_gt_mask[4:],
                new_gt_mask_bilinear[4:].detach().clone().cpu().numpy() > 0.25
            )

            data['gt_masks_bev'] = new_gt_mask[
                :,
                ((xDim - self.DxDim) // 2):((xDim + self.DxDim) // 2),
                ((yDim - self.DyDim) // 2):((yDim + self.DyDim) // 2)
            ]
        else:
            xDim = gt_masks.shape[-2]
            yDim = gt_masks.shape[-1]

            data['gt_masks_bev'] = gt_masks[
                :,
                ((xDim - self.DxDim) // 2):((xDim + self.DxDim) // 2),
                ((yDim - self.DyDim) // 2):((yDim + self.DyDim) // 2)
            ]

        return data


@PIPELINES.register_module()
class LoadPointsFromFile:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # TODO: make it more general
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results


@PIPELINES.register_module()
class LoadSimBEVPointsFromFile:
    '''
    Load lidar point cloud from NumPy file.

    Args:
        - coord_type: coordinate type of the data.
        - trim_step: channel step size for trimming the point cloud.
    '''

    def __init__(self, coord_type, trim_step):
        self.coord_type = coord_type
        self.trim_step = trim_step
    
    def __call__(self, results):
        '''
        Load point cloud data from file.

        Args:
            results: dictionary containing path to point cloud file.
        
        Returns:
            results: dictionary containing point cloud data.
        '''
        lidar_path = results['lidar_path']
        
        points = load_points(lidar_path)

        if self.trim_step > 1:
            points = trim_points(points, self.trim_step)
        
        points_class = get_points_type(self.coord_type)
        
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        
        results['points'] = points

        return results


@PIPELINES.register_module()
class LoadSimBEVPointsFromMultiSweeps:
    '''
    Load and aggregate multiple point clouds.

    Args:
        - num_sweeps: desired number of point clouds.
        - trim_step: channel step size for trimming the point cloud.
        - time_step: time step between successive point clouds, i.e.
            simulation time step.
        - is_train: whether the data is for training.
    '''

    def __init__(self, num_sweeps, trim_step, time_step, is_train=True):
        self.num_sweeps = num_sweeps
        self.trim_step = trim_step
        self.time_step = time_step
        self.is_train = is_train

    def __call__(self, results):
        '''
        Load and aggregate multiple point clouds.

        Args:
            results: dictionary containing path to point cloud files.
        
        Returns:
            results: dictionary containing point cloud data.
        '''
        points = results['points']

        # Add the time dimension to the principal point cloud data and set it
        # to zero.
        points.tensor = torch.cat((points.tensor, torch.zeros((points.tensor.shape[0], 1))), dim=1)

        ego2global = results['ego2global']
        lidar2ego = results['lidar2ego']

        total_num_sweeps = len(results['sweeps_lidar_paths'])
        
        # If, during training, the desired number of point clouds is less than
        # the total number available, randomly sample the desired number of
        # point clouds from those available. Otherwise, and during testing,
        # choose the previous point clouds in order.
        if self.num_sweeps >= total_num_sweeps:
            choices = np.arange(total_num_sweeps)
        elif not self.is_train:
            choices = np.arange(self.num_sweeps)
        else:
            choices = np.random.choice(total_num_sweeps, self.num_sweeps, replace=False)

        sweep_points_list = [points]

        for i in choices:
            lidar_path = results['sweeps_lidar_paths'][i]
            
            sweep_points = load_points(lidar_path)

            if self.trim_step > 1:
                sweep_points = trim_points(sweep_points, self.trim_step)

            # Transform point cloud to the coordinate system of the principal
            # point cloud.
            sweep_ego2global = results['sweeps_ego2global'][i]

            lidar2lidar = np.linalg.inv(ego2global @ lidar2ego) @ sweep_ego2global @ lidar2ego

            sweep_points = points.new_point(
                (lidar2lidar @ np.append(sweep_points, np.ones((sweep_points.shape[0], 1)), 1).T)[:3].T
            )

            sweep_points.tensor = torch.cat(
                (sweep_points.tensor, torch.full((sweep_points.tensor.shape[0], 1), self.time_step * (i + 1))),
                dim=1
            )
            
            sweep_points_list.append(sweep_points)
        
        points = points.cat(sweep_points_list)

        results['points'] = points

        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
    """

    def __init__(
        self,
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_bbox=False,
        with_label=False,
        with_mask=False,
        with_seg=False,
        with_bbox_depth=False,
        poly2mask=True,
    ):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
        )
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results["gt_bboxes_3d"] = results["ann_info"]["gt_bboxes_3d"]
        results["bbox3d_fields"].append("gt_bboxes_3d")
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results["centers2d"] = results["ann_info"]["centers2d"]
        results["depths"] = results["ann_info"]["depths"]
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["gt_labels_3d"] = results["ann_info"]["gt_labels_3d"]
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["attr_labels"] = results["ann_info"]["attr_labels"]
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)

        return results
