import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
class SPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPModule, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.dilated_conv3x3_rate6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.dilated_conv3x3_rate12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.dilated_conv3x3_rate6(x)
        x4 = self.dilated_conv3x3_rate12(x)
        ret = self.fuse(torch.cat([x1, x2, x3, x4], dim=1))
        return ret
    
class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)
    
@MODELS.register_module()    
class LRF(nn.Module): 
    def __init__(self, in_channels: int, out_channels: int):
        super(LRF, self).__init__()
        
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.seblock = SE_Block(out_channels)

        self.sspblock = SPPModule(sum(in_channels), out_channels)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels*3, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        self.conv_d0 = nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=2, padding=1 )
        self.conv_d1 = nn.Conv2d(in_channels[1], in_channels[1], kernel_size=3, stride=2, padding=1 )

        self.ms_conv0 = nn.Sequential(
            nn.Conv2d(sum(in_channels), in_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels[0], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        self.ms_conv1 = nn.Sequential(
            nn.Conv2d(sum(in_channels), in_channels[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels[1], eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

    def forward(self, inputs):
        # MSIT
        feat0 = inputs[0]
        f0_small = self.conv_d0(feat0)
        f0_small = F.interpolate(f0_small, scale_factor=2, mode='bilinear', align_corners=False)

        feat1 = inputs[1]
        f1_small = self.conv_d1(feat1)
        f1_small = F.interpolate(f1_small, scale_factor=2, mode='bilinear', align_corners=False)

        feat_A = self.ms_conv0(torch.cat([feat0, f1_small],dim=1))
        feat_B = self.ms_conv1(torch.cat([feat1, f0_small],dim=1))

        inputs_ms = [feat_A, feat_B]

        # Trident Fusion
        x1 = self.seblock(self.conv3x3(torch.cat(inputs_ms, dim=1)))
        x2 = self.sspblock(torch.cat(inputs, dim=1))
        x3 = self.conv1x1(torch.cat(inputs, dim=1))
        
        x= self.fuse(torch.cat([x1, x2, x3], dim=1))

        # CA is in dbfusion_head 
        return x

# instance-level fusion
@MODELS.register_module()
class dimconv(nn.Module):

    def __init__(self, indim , outdim) -> None:
        super(dimconv, self).__init__()
        self.linear = nn.Linear(sum(indim), outdim)
        self.norm = nn.LayerNorm(outdim)
        
    def forward(self, inputs) -> torch.Tensor:
        outputs = self.linear(torch.cat(inputs, dim=1).permute(0,2,1).contiguous())
        outputs = self.norm(outputs).permute(0,2,1).contiguous()
        return outputs

# instance-level fusion
@MODELS.register_module()
class dimconv_ts(nn.Module):

    def __init__(self, indim , outdim) -> None:
        super(dimconv_ts, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(sum(indim), indim[0]),
            nn.LayerNorm(indim[0]),
            nn.ReLU())

        self.fn_0 = nn.Sequential(
            nn.Linear(indim[0], indim[0]),
            nn.LayerNorm(indim[0]),
            nn.ReLU())
        self.gamma_0 = nn.Linear(indim[0], indim[0])
        self.beta_0 = nn.Linear(indim[0], indim[0])

        self.fn_1 = nn.Sequential(
            nn.Linear(indim[1], indim[1]),
            nn.LayerNorm(indim[1]),
            nn.ReLU())
        self.gamma_1 = nn.Linear(indim[1], indim[1])
        self.beta_1 = nn.Linear(indim[1], indim[1])


        self.linear = nn.Linear(sum(indim), outdim)
        self.norm = nn.LayerNorm(outdim)

        self.init_weight()
    
    def init_weight(self):
        nn.init.zeros_(self.gamma_0.weight)
        nn.init.zeros_(self.beta_0.weight)
        nn.init.ones_(self.gamma_0.bias)
        nn.init.zeros_(self.beta_0.bias)

        nn.init.zeros_(self.gamma_1.weight)
        nn.init.zeros_(self.beta_1.weight)
        nn.init.ones_(self.gamma_1.bias)
        nn.init.zeros_(self.beta_1.bias)

    def forward(self, inputs) -> torch.Tensor:

        input_0 = inputs[0].permute(0,2,1).contiguous()
        input_1 = inputs[1].permute(0,2,1).contiguous()

        merge_inputs = self.ffn(torch.cat(inputs, dim=1).permute(0,2,1).contiguous())

        x_0 = self.fn_0(merge_inputs)
        gamma_0 = self.gamma_0(x_0)
        beta_0 = self.beta_0(x_0)
        out_0 = gamma_0 * input_0 + beta_0

        x_1 = self.fn_1(merge_inputs)
        gamma_1 = self.gamma_1(x_1)
        beta_1 = self.beta_1(x_1)
        out_1 = gamma_1 * input_1 + beta_1
        
        outputs = self.linear(torch.cat([out_0, out_1], dim=-1))
        outputs = self.norm(outputs).permute(0,2,1).contiguous()
        return outputs