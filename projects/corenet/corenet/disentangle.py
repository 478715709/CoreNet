import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from .ref_dfa import BEV_DFA
from einops import rearrange
import numbers
from mmcv.cnn import ConvModule, build_conv_layer

###################################
# use norm conv as the backbone 
class ConvFeatureExtraction(nn.Module):
    def __init__(self, dim=64 , out_dim=128 ,num_layers=3):
        super(ConvFeatureExtraction, self).__init__()
        
        self.conv_layers  = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)))

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

    def forward(self, x):
        for i in range(len(self.conv_layers)):
            x = F.relu(x + self.conv_layers[i](x))
        return self.conv1x1(x)

###################################
@MODELS.register_module()    
class DisentangleMoudule(nn.Module):
    def __init__(self, pts_dim, img_dim):
        super(DisentangleMoudule, self).__init__()

        self.pts_specific = ConvFeatureExtraction(pts_dim, num_layers=3)
        self.img_specific = ConvFeatureExtraction(img_dim, num_layers=3)
        
        self.pts_share = ConvFeatureExtraction(pts_dim, num_layers=3)
        self.img_share = ConvFeatureExtraction(img_dim, num_layers=3)

    def forward(self, img_bev, pts_bev):
        img_share = self.img_share(img_bev)
        pts_share = self.pts_share(pts_bev)
        img_specific = self.img_specific(img_bev)
        pts_specific = self.pts_specific(pts_bev)
        
        return img_specific, pts_specific, [img_share, pts_share]