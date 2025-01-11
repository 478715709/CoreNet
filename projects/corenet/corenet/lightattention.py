import torch
import torch.nn as nn
from einops import rearrange
import torch.functional as F
from mmdet3d.registry import MODELS
class AttentionBase(nn.Module):
    def __init__(self,
                 dim=128,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv1d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv1d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, hidden_dim, proposal_num]
        b, c, n = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
      
        q = rearrange(q, 'b (head c) n -> b head c n',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) n -> b head c n',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) n -> b head c n',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c n -> b (head c) n',
                        head=self.num_heads)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv1d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv1d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv1d(
            hidden_features, in_features, kernel_size=1, bias=bias)
        
        self.act = nn.GELU()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.project_out(x)
        return x

class lightMMatten(nn.Module): # Self attention in SRF
    def __init__(self,
                 dim=128,
                 num_heads=8,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(lightMMatten, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
        self.conv_cls = nn.Conv1d(dim*2, dim, 1)
        self.conv_pts = nn.Conv1d(dim*2, dim, 1)

    def forward(self, x, num_proposals):
        x = x + self.attn(self.norm1(x.transpose(2,1)).transpose(2,1))
        x = x + self.mlp(self.norm2(x.transpose(2,1)).transpose(2,1))
        img_query_feat = x[:,:,:num_proposals]
        pts_query_feat = x[:,:,num_proposals:]

        cls_query_feat = self.conv_cls(torch.cat([img_query_feat, pts_query_feat], dim=1))
        bbox_query_feat = self.conv_pts(torch.cat([pts_query_feat, img_query_feat], dim=1))

        return cls_query_feat, bbox_query_feat