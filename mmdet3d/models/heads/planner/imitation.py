from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import HEADS

__all__ = ["ImitationHead"]


@HEADS.register_module()
class ImitationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, waypoint_config: Dict[str, Any]) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gru_concat_target_point = waypoint_config["gru_concat_target_point"]
        self.pred_len = waypoint_config["pred_len"]

        #waypoint prediction
        self.join = nn.Sequential(
                            nn.Linear(in_channels, in_channels),
                            nn.ReLU(inplace=True),
                            nn.Linear(in_channels, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        )

        self.decoder = nn.GRUCell(input_size=6 if self.gru_concat_target_point else 3, # 2 represents x,y coordinate
                                  hidden_size=waypoint_config["gru_hidden_size"])
        
        self.gru_hidden_size = waypoint_config["gru_hidden_size"]   
        self.output = nn.Sequential(
            nn.Linear(self.gru_hidden_size,4),
            nn.ReLU(inplace=True),
            nn.Linear(4,4),
            nn.Linear(4,3)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # self.init_fc = nn.Sequential(
        #     nn.Linear(in_channels, self.gru_hidden_size),
        #     nn.ReLU(inplace=True),
        # )

        
    def transform_pose_to_xyv(self,pose,v):
        x = pose[:,0,3]
        y = pose[:,1,3]
        
        return torch.stack([x,y,v],dim=1)



    def forward(self, z, goal_point, gt_trajectory,gt_trajectory_speed,goal_point_speed):
        if isinstance(z, (list, tuple)):
            z = z[0]

        z = self.pool(z).view(z.shape[0], -1)
        z = self.join(z)
        #z = self.init_fc(z) 

    
        output_wp = list()
        
        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 3), dtype=z.dtype).to(z.device)

        goal_x = goal_point[:,0,3]
        goal_y = goal_point[:,1,3]
        goal_v = goal_point_speed
        goal_point = torch.stack([goal_x,goal_y,goal_v],dim=1)
        #target_point[:, 1] *= -1
        
        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            if self.gru_concat_target_point:
                x_in = torch.cat([x, goal_point], dim=1)
            else:
                x_in = x
            
            z = self.decoder(x_in, z)
            dx = self.output(z)
            
            x = dx[:,:3] + x
            
            output_wp.append(x[:,:3])
            
        pred_wp = torch.stack(output_wp, dim=1)

        if self.training:
            losses = {}
            gt_waypoints = torch.zeros_like(pred_wp)
            for i in range(self.pred_len-1):
                waypoint = self.transform_pose_to_xyv(gt_trajectory[:, i, :,:],gt_trajectory_speed[:,i])
                gt_waypoints[:,i,:] = waypoint
            gt_waypoints[:,-1,:] = goal_point  
            gt_waypoints = gt_waypoints.to(pred_wp.device)

            losses["waypoint_loss"] = 1.0 * F.l1_loss(pred_wp, gt_waypoints, reduction="mean")
            return losses
        

        # pred the wapoints in the vehicle coordinate and we convert it to lidar coordinate here because the GT waypoints is in lidar coordinate
        #pred_wp[:, :, 0] = pred_wp[:, :, 0] - self.config.lidar_pos[0]
            
        return pred_wp
    
