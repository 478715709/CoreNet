import torch
from mmdet3d.registry import MODELS
import torch.nn as nn
import pdb
import copy
import torch.nn.functional as F
import math
from mmdet.models.layers.positional_encoding import SinePositionalEncoding
from .deformable_atten import MSDeformAttn

def get_ref_2d(H, W, bs, device='cuda', dtype=torch.float):
    ref_y, ref_x = torch.meshgrid(
                    torch.linspace(
                        0.5, H - 0.5, H, dtype=dtype, device=device),
                    torch.linspace(
                        0.5, W - 0.5, W, dtype=dtype, device=device)
                )
    ref_y = ref_y.reshape(-1)[None] / H
    ref_x = ref_x.reshape(-1)[None] / W
    ref_2d = torch.stack((ref_x, ref_y), -1)
    ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
    return ref_2d

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class BEV_DFA_layer(nn.Module):
    def __init__(self, query_channel = 256, dropout=0.1):
        super(BEV_DFA_layer, self).__init__()
        

        # attention
        self.atten = MSDeformAttn(query_channel)
        self.norm_1 = nn.LayerNorm(query_channel)
        self.dropout1 = nn.Dropout(dropout)

        # ffn
        self.linear1 = nn.Linear(query_channel, query_channel)
        self.activation = F.relu
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(query_channel, query_channel)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(query_channel)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, query, ref_points, input_spatial_shapes, input_level_start_index, pos=None):  # N C H W
        outs  = self.atten(self.with_pos_embed(query,pos), ref_points, query, input_spatial_shapes, input_level_start_index)

        query = query + self.dropout1(outs)
        query = self.norm_1(query)

        query = self.forward_ffn(query)
        return query
    
@MODELS.register_module()
class BEV_DFA(nn.Module):
    def __init__(self, query_channel=256 ,num_layer=3):
        super(BEV_DFA, self).__init__()
        self.proj_query = nn.Linear(query_channel, query_channel)
        self.positional_encoding = SinePositionalEncoding(int(query_channel/2))
        self.layers = _get_clones(BEV_DFA_layer(query_channel), num_layer)
    def forward(self, BEV_feat, pos=None):
        bs, C, H, W = BEV_feat.shape
        output = self.proj_query(BEV_feat.flatten(2).permute(0,2,1).contiguous())
        
        ref_points = get_ref_2d(H,W,bs)
        input_spatial_shapes = torch.Tensor([[H,W]]).cuda().long()
        input_level_start_index = torch.Tensor([0]).cuda().long()

        bev_mask = torch.zeros((bs, H, W), # [B H W]
                               device=BEV_feat.device).to(torch.float32)
        bev_pos = self.positional_encoding(bev_mask).to(torch.float32).flatten(2).permute(0, 2, 1).contiguous()
        for _, layer in enumerate(self.layers):
            output = layer(output, ref_points, input_spatial_shapes, input_level_start_index, bev_pos)

        return output.permute(0,2,1).contiguous().reshape(bs,C,H,W)
if __name__ == '__main__':
    
    DFA = BEV_DFA(64,3).cuda()
    BEV_feat = torch.ones(12, 64, 256, 704).cuda()
    outs = DFA(BEV_feat)
    pdb.set_trace()