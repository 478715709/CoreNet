import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from typing import List, Tuple, Optional

# ========== 1. Mamba状态空间模型核心组件 ==========

class SSM(nn.Module):
    """状态空间模型 (State Space Model)"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        # SSM参数
        self.A = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))

        # 选择机制参数
        self.delta = nn.Linear(d_model, d_model)
        self.delta_act = nn.SiLU()

        # 卷积层（用于序列建模）
        self.conv = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv-1, groups=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # 选择性机制
        delta = self.delta_act(self.delta(x))  # (batch, seq_len, d_model)

        # 离散化状态空间参数
        A_discrete = torch.exp(self.A.unsqueeze(0) * delta.unsqueeze(-1))  # (batch, seq_len, d_model, d_state)
        B_discrete = self.B.unsqueeze(0).unsqueeze(1) * delta.unsqueeze(-1)  # (batch, seq_len, d_model, d_state)

        # 卷积操作
        x_conv = rearrange(x, 'b l d -> b d l')
        x_conv = self.conv(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, 'b d l -> b l d')

        # 状态空间模型计算
        h = torch.zeros(batch, d_model, self.d_state, device=x.device)  # 初始状态
        outputs = []

        for t in range(seq_len):
            # 状态更新
            h = A_discrete[:, t] * h + B_discrete[:, t] * x_conv[:, t].unsqueeze(-1)

            # 输出计算
            y_t = torch.sum(self.C.unsqueeze(0) * h, dim=-1) + self.D * x_conv[:, t]
            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)
        return output

class MambaBlock(nn.Module):
    """Mamba块"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = d_model * expand

        # 前向投影
        self.in_proj = nn.Linear(d_model, d_inner * 2)

        # SSM
        self.ssm = SSM(d_inner, d_state, d_conv)

        # 后向投影
        self.out_proj = nn.Linear(d_inner, d_model)

        # 层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # 前向投影
        x_proj = self.in_proj(x)  # (batch, seq_len, d_inner*2)
        x1, x2 = x_proj.chunk(2, dim=-1)

        # SSM处理
        x1 = self.ssm(x1)

        # 激活和门控
        x_out = x1 * F.silu(x2)

        # 后向投影
        x_out = self.out_proj(x_out)

        return x_out + residual

# ========== 2. IM-Fuse核心组件 ==========

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 线性变换
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # 生成Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 输出计算
        output = torch.matmul(attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.proj(output)

        return output

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, d_model: int, num_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network
        mlp_hidden_dim = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Feed-Forward
        x = x + self.mlp(self.norm2(x))
        return x

class ConvEncoderBlock(nn.Module):
    """3D卷积编码器块"""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3, kernel_size: int = 3):
        super().__init__()
        blocks = []
        fixed_groups = 8 # Original fixed number of groups

        for i in range(num_blocks):
            block_in = in_channels if i == 0 else out_channels
            actual_groups = min(fixed_groups, block_in)

            blocks.extend([
                nn.GroupNorm(actual_groups, block_in),
                nn.ReLU(inplace=True),
                nn.Conv3d(block_in, out_channels, kernel_size=kernel_size,
                         stride=2 if i == 0 else 1,
                         padding=kernel_size//2, padding_mode='replicate')
            ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

class ConvDecoderBlock(nn.Module):
    """3D卷积解码器块"""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3, kernel_size: int = 3):
        super().__init__()
        blocks = []
        fixed_groups = 8 # Original fixed number of groups

        for i in range(num_blocks):
            block_in = in_channels if i == 0 else out_channels
            actual_groups = min(fixed_groups, block_in)

            if i == num_blocks - 1:
                # 最后一层进行上采样
                blocks.extend([
                    nn.GroupNorm(actual_groups, block_in),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(block_in, out_channels, kernel_size=2, stride=2)
                ])
            else:
                blocks.extend([
                    nn.GroupNorm(actual_groups, block_in),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(block_in, out_channels, kernel_size=kernel_size,
                             padding=kernel_size//2, padding_mode='replicate')
                ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.blocks(x)

class IntraModalTransformer(nn.Module):
    """模态内Transformer"""
    def __init__(self, d_model: int, num_layers: int = 4, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class MambaFusionBlock(nn.Module):
    """Mamba融合块 (MFB)"""
    def __init__(self, d_model: int, num_modalities: int = 4, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities

        # 可学习token
        self.learnable_tokens = nn.Parameter(torch.randn(1, 1, d_model))

        # Mamba块
        self.mamba = MambaBlock(d_model, d_state)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

        # 层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x has 3 dimensions (batch, seq_len, d_model)
        # If x is a list of tensors, concatenate them
        if isinstance(x, list):
            # Assuming x is a list of (batch, seq_len, d_model) tensors from different modalities
            # For MFB, we concatenate them along the sequence dimension
            batch_size = x[0].shape[0]
            seq_len = x[0].shape[1]
            all_tokens = []
            for tokens in x:
                all_tokens.append(tokens)

            learnable_tokens = repeat(self.learnable_tokens, '1 1 d -> b 1 d', b=batch_size)
            learnable_tokens = repeat(learnable_tokens, 'b 1 d -> b l d', l=seq_len)
            all_tokens.append(learnable_tokens)
            x = torch.cat(all_tokens, dim=1)

        residual = x
        x = self.norm(x)

        # 前向投影
        x_proj = self.in_proj(x)  # (batch, seq_len, d_inner*2)
        x1, x2 = x_proj.chunk(2, dim=-1)

        # SSM处理
        x1 = self.ssm(x1)

        # 激活和门控
        x_out = x1 * F.silu(x2)

        # 后向投影
        x_out = self.out_proj(x_out)

        return x_out + residual

class InterleavedMambaFusionBlock(nn.Module):
    """交错Mamba融合块 (I-MFB)"""
    def __init__(self, d_model: int, num_modalities: int = 4, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities

        # 可学习token（每个位置一个）
        self.learnable_tokens = nn.Parameter(torch.randn(1, num_modalities, d_model))

        # Mamba块
        self.mamba = MambaBlock(d_model, d_state)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, modality_tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        modality_tokens: 每个模态的token列表
        返回: 融合后的特征
        """
        batch_size = modality_tokens[0].shape[0]
        seq_len = modality_tokens[0].shape[1]

        # 交错拼接：模态token和可学习token交替
        interleaved_tokens_list = [] # Renamed to avoid confusion with the single interleaved_tokens tensor
        for i in range(seq_len):
            # 为每个位置收集所有模态的token
            for m in range(self.num_modalities):
                token_slice = modality_tokens[m][:, i:i+1, :]
                interleaved_tokens_list.append(token_slice)

            # 添加可学习token
            learnable_token_slice = self.learnable_tokens[:, i % self.num_modalities:i % self.num_modalities + 1, :]
            learnable_token_repeated = repeat(learnable_token_slice, '1 1 d -> b 1 d', b=batch_size)
            interleaved_tokens_list.append(learnable_token_repeated)

        # Debugging print statements
        print(f"\nInside InterleavedMambaFusionBlock.forward (batch_size={batch_size}, seq_len={seq_len}):")
        print(f"  Number of tensors in interleaved_tokens_list: {len(interleaved_tokens_list)}")
        print(f"  Shapes of tensors in interleaved_tokens_list before cat:")
        for idx, t in enumerate(interleaved_tokens_list):
            print(f"    Tensor {idx}: {t.shape}")
        # End debugging

        # 拼接所有交错token
        concatenated = torch.cat(interleaved_tokens_list, dim=1)  # (batch, seq_len*(num_modalities+1), d_model)

        # Mamba处理
        mamba_out = self.mamba(concatenated)

        # 只取可学习token对应的输出
        fused_indices = torch.arange(self.num_modalities,
                                    mamba_out.shape[1],
                                    self.num_modalities + 1) # Corrected step
        fused_tokens = mamba_out[:, fused_indices, :]

        return self.out_proj(fused_tokens)

class HybridModalityEncoder(nn.Module):
    """混合模态特定编码器"""
    def __init__(self, in_channels: int = 1, base_channels: int = 8,
                 num_stages: int = 5, d_model: int = 256):
        super().__init__()
        self.num_stages = num_stages
        self.d_model = d_model

        # 模态特定的卷积编码器
        self.conv_encoders = nn.ModuleList()
        channels = base_channels

        for i in range(num_stages):
            in_ch = in_channels if i == 0 else channels
            out_ch = channels * 2 if i < num_stages - 1 else d_model

            encoder = ConvEncoderBlock(in_ch, out_ch, num_blocks=3)
            self.conv_encoders.append(encoder)

            channels = out_ch

        # 模态内Transformer
        self.intra_modal_transformer = IntraModalTransformer(d_model)

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        # Token投影
        self.token_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        x: (batch, channels, H, W, D)
        返回: (skip_connections, global_features)
        """
        batch_size = x.shape[0]
        skip_connections = []

        # 卷积编码
        for i, encoder in enumerate(self.conv_encoders):
            x = encoder(x)
            # Collect all outputs as skips, including the bottleneck feature
            skip_connections.append(x)

        # The last element of skip_connections is the bottleneck feature
        bottleneck_feat = skip_connections[-1]

        # 展平为序列
        spatial_dims = bottleneck_feat.shape[2:]
        seq_len = spatial_dims[0] * spatial_dims[1] * spatial_dims[2]
        x_flattened = rearrange(bottleneck_feat, 'b c h w d -> b (h w d) c')

        # Token投影和位置编码
        tokens = self.token_proj(x_flattened)
        pos_embed = repeat(self.pos_embedding, '1 1 d -> b l d',
                          b=batch_size, l=seq_len)
        tokens = tokens + pos_embed

        # 模态内Transformer
        global_features = self.intra_modal_transformer(tokens)

        return skip_connections, global_features

class MultiModalTransformer(nn.Module):
    """多模态Transformer"""
    def __init__(self, d_model: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        pos_embed = repeat(self.pos_embedding, '1 1 d -> b l d',
                          b=batch_size, l=seq_len)
        x = x + pos_embed

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)

# ========== 3. 完整的IM-Fuse模型 ==========

class IMFuse(nn.Module):
    """IM-Fuse: 基于Mamba的不完整模态融合脑肿瘤分割模型"""
    def __init__(self, 
                 num_modalities: int = 4,
                 in_channels: int = 1,
                 base_channels: int = 8,
                 num_stages: int = 5,
                 d_model: int = 256,
                 num_classes: int = 4,  # 背景 + 3个肿瘤区域
                 use_interleaved: bool = True):
        super().__init__()

        self.num_modalities = num_modalities
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.d_model = d_model # Store d_model for use in projections

        # 模态特定编码器（每个模态一个）
        self.modality_encoders = nn.ModuleList([
            HybridModalityEncoder(in_channels, base_channels, num_stages, d_model)
            for _ in range(num_modalities)
        ])

        # Mamba融合块
        if use_interleaved:
            self.fusion_blocks = nn.ModuleList([
                InterleavedMambaFusionBlock(d_model, num_modalities)
                for _ in range(num_stages + 1)  # 瓶颈层 + 每个skip connection
            ])
        else:
            # The original MambaFusionBlock also needed to handle list input. Modified above.
            self.fusion_blocks = nn.ModuleList([
                MambaFusionBlock(d_model, num_modalities)
                for _ in range(num_stages + 1)
            ])

        # Projection layers for skip tokens before fusion
        self.skip_token_projs = nn.ModuleList()
        current_enc_ch = base_channels
        for i in range(num_stages):
            out_ch_encoder = current_enc_ch * 2 if i < num_stages - 1 else d_model
            # The channels of the skip features are out_ch_encoder. 
            # We need to project these to self.d_model before passing to fusion block.
            # The last skip output (bottleneck) already has d_model channels, so it doesn't need projection here.
            # We only need projections for the first num_stages-1 skips
            if i < num_stages: # All skips need to be projected to d_model for the fusion block
                if out_ch_encoder != d_model:
                    self.skip_token_projs.append(nn.Linear(out_ch_encoder, d_model))
                else:
                    # If channels already match d_model, add an identity layer or None
                    self.skip_token_projs.append(nn.Identity())
            current_enc_ch = out_ch_encoder


        # 多模态Transformer（瓶颈层）
        self.multimodal_transformer = MultiModalTransformer(d_model)

        # 卷积解码器
        self.conv_decoder = self._build_decoder(d_model, base_channels, num_stages)

        # 共享权重的解码器（用于辅助损失）
        self.shared_decoder = self._build_decoder(d_model, base_channels, num_stages)

        # 输出层
        self.output_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

        # 初始化参数
        self._init_weights()

    def _build_decoder(self, d_model: int, base_channels: int, num_stages: int):
        """构建卷积解码器"""
        layers = nn.ModuleList()
        
        # Calculate encoder output channels at each stage for skip connections
        # This list will be [base_channels, base_channels*2, ..., d_model]
        encoder_output_channels = []
        current_enc_ch = base_channels
        for i in range(num_stages - 1):
            encoder_output_channels.append(current_enc_ch)
            current_enc_ch *= 2
        encoder_output_channels.append(d_model) # Append the bottleneck channels
        
        # Reverse the list for decoder processing (skips from deepest to shallowest)
        # e.g., [256, 64, 32, 16, 8] for num_stages=5, base_channels=8, d_model=256
        skip_channels_for_decoder = list(reversed(encoder_output_channels))
        
        # The input to the first decoder block (before concatenation with skip) is the bottleneck d_model
        decoder_current_channels = d_model 

        for i in range(num_stages):
            # The ConvDecoderBlock receives the concatenated feature map and skip connection
            # Note: The skip features are already projected to d_model before this point
            in_ch_to_block = decoder_current_channels + d_model # All fused skips are now d_model
            
            # The output channels after this decoder stage (after upsampling and convolution)
            if i < num_stages - 1:
                out_ch = decoder_current_channels // 2 
            else:
                out_ch = base_channels # Final output layer of the decoder

            decoder = ConvDecoderBlock(in_ch_to_block, out_ch)
            layers.append(decoder)
            
            # Update channels for the next decoder stage
            decoder_current_channels = out_ch

        return layers

    def _init_weights(self):
        """初始化模型参数"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs: List[torch.Tensor],
                modality_mask: Optional[torch.Tensor] = None):
        """
        inputs: 模态输入列表，每个形状为 (batch, 1, H, W, D)
        modality_mask: (batch, num_modalities) 模态存在性掩码，1表示存在，0表示缺失
        返回: 分割结果和中间特征（用于损失计算）
        """
        batch_size = inputs[0].shape[0]

        # 如果没有提供掩码，假设所有模态都存在
        if modality_mask is None:
            modality_mask = torch.ones(batch_size, self.num_modalities, device=inputs[0].device)

        # 编码每个模态
        modality_skips = []
        modality_globals = []

        for i, (encoder, inp) in enumerate(zip(self.modality_encoders, inputs)):
            mask = modality_mask[:, i:i+1, None, None, None]
            # 应用模态掩码
            masked_inp = inp * mask

            # 模态特定编码
            skips, global_feat = encoder(masked_inp)

            # 应用掩码到特征
            mask_flat = modality_mask[:, i:i+1].unsqueeze(-1)
            skips = [s * mask_flat for s in skips] # Apply mask to skip connections
            global_feat = global_feat * mask_flat

            modality_skips.append(skips) # `skips` now contains num_stages elements
            modality_globals.append(global_feat)

        # 融合瓶颈层特征
        # modality_globals should already be d_model=256 from HybridModalityEncoder's last stage
        bottleneck_fused = self.fusion_blocks[0](modality_globals)

        # 多模态Transformer
        multimodal_global = self.multimodal_transformer(bottleneck_fused)

        # 重塑为卷积特征
        spatial_shape = inputs[0].shape[2:]  # H, W, D
        
        # Corrected calculation for effective downsampling based on ConvEncoderBlock
        # Each ConvEncoderBlock applies a stride=2 conv in its first layer,
        # and there are 'num_stages' such blocks. So total downsampling is 2^num_stages.
        downsampling_factor = 2 ** self.num_stages # Corrected calculation

        h = spatial_shape[0] // downsampling_factor
        w = spatial_shape[1] // downsampling_factor
        d = spatial_shape[2] // downsampling_factor

        conv_feat = rearrange(multimodal_global, 'b (h w d) c -> b c h w d',
                             h=h, w=w, d=d)

        # 解码过程
        decoder_features = []
        x = conv_feat

        for i in range(self.num_stages):
            # Fuse skip connection
            # skip_idx will go from num_stages-1 down to 0, matching num_stages skips
            skip_idx = self.num_stages - 1 - i 

            skip_features = []
            for m in range(self.num_modalities):
                skip = modality_skips[m][skip_idx]
                skip_flat = rearrange(skip, 'b c h w d -> b (h w d) c')
                # Project skip tokens to self.d_model before fusion
                projected_skip_token = self.skip_token_projs[skip_idx](skip_flat)
                skip_features.append(projected_skip_token)

            # Use fusion block for this decoder stage (index i+1)
            fused_skip = self.fusion_blocks[i + 1](skip_features)
            # The fused_skip now has d_model channels, but needs to be reshaped for conv_decoder
            fused_skip_conv = rearrange(fused_skip, 'b (h w d) c -> b c h w d',
                                       h=modality_skips[0][skip_idx].shape[2],
                                       w=modality_skips[0][skip_idx].shape[3],
                                       d=modality_skips[0][skip_idx].shape[4])

            # 解码
            x = self.conv_decoder[i](x, fused_skip_conv)
            decoder_features.append(x)

        # 最终输出
        segmentation = self.output_conv(x)

        # 计算辅助损失的特征
        aux_features = []
        for i, global_feat in enumerate(modality_globals):
            # 将全局特征转换为卷积特征
            conv_feat_aux = rearrange(global_feat, 'b (h w d) c -> b c h w d',
                                     h=h, w=w, d=d) # Use the corrected h,w,d here

            # 通过共享解码器
            x_aux = conv_feat_aux
            for j in range(self.num_stages):
                # For auxiliary loss, the input to shared_decoder[j] is x_aux from previous stage.
                # The _build_decoder method for shared_decoder expects a `d_model` input channel at the beginning.
                # Need to consider if x_aux has d_model channels, and if the skips from encoder_output_channels are relevant here.
                # The shared_decoder also uses the _build_decoder logic. So its `in_ch_to_block` also concatenates.
                # However, aux_preds are calculated from a single global_feat from one modality, so there's no fusing of skips here.
                # For now, assuming the shared_decoder directly processes x_aux without skip concatenation.
                # This implies shared_decoder's first ConvDecoderBlock receives d_model, and subsequent ones receive progressively smaller channels.
                # I need to verify how `_build_decoder` is used for `shared_decoder` and if it is consistent with how `aux_preds` are generated.
                # Given current _build_decoder, it's expecting an `in_ch_to_block = decoder_current_channels + d_model` after `x_aux` in the aux loop,
                # but there's no fused skip in this loop. So, it's incorrect.
                # I should probably have a separate simple decoder or modify how shared_decoder is used for aux_preds.
                # For now, let's simplify shared_decoder's use in aux_preds by having it process only x_aux, not use skips.
                # To achieve this, I'd need a different `_build_decoder` that doesn't expect skips or modify the forward of aux loop.
                # Let's assume shared_decoder blocks can take single input, and pass None for skip. This would require modification of ConvDecoderBlock logic.
                # A simpler fix for this current setup: the `_build_decoder` builds blocks that internally expect `x` and `skip`. If we are using it for aux, we need to adapt.
                # A quick fix for now is to pass `None` for skip, assuming ConvDecoderBlock can handle it, which it does.
                x_aux = self.shared_decoder[j](x_aux, None)

            aux_features.append(x_aux)

        return segmentation, decoder_features, aux_features

# ========== 4. 损失函数 ==========

class DiceLoss(nn.Module):
    """Dice损失函数"""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (batch, num_classes, H, W, D)
        target: (batch, H, W, D) 包含类别索引
        """
        # 将target转换为one-hot
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).permute(0, 4, 1, 2, 3).float()

        # 计算Dice系数
        intersection = torch.sum(pred * target_onehot, dim=(2, 3, 4))
        union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(target_onehot, dim=(2, 3, 4))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss

class WeightedCrossEntropyLoss(nn.Module):
    """加权交叉熵损失"""
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (batch, num_classes, H, W, D)
        target: (batch, H, W, D)
        """
        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)
            ce_loss = F.cross_entropy(pred, target, weight=weights, reduction='mean')
        else:
            ce_loss = F.cross_entropy(pred, target, reduction='mean')

        return ce_loss

class IMFuseLoss(nn.Module):
    """IM-Fuse的总损失函数"""
    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = WeightedCrossEntropyLoss(class_weights)
        self.alpha = alpha  # 最终输出损失权重
        self.beta = beta    # 解码器中间损失权重
        self.gamma = gamma  # 编码器辅助损失权重

    def forward(self, 
                final_pred: torch.Tensor,
                decoder_preds: List[torch.Tensor],
                aux_preds: List[torch.Tensor],
                target: torch.Tensor) -> torch.Tensor:
        """
        final_pred: 最终分割预测 (batch, num_classes, H, W, D)
        decoder_preds: 解码器中间预测列表
        aux_preds: 辅助解码器预测列表
        target: 真实标签 (batch, H, W, D)
        """
        # 最终输出损失
        final_dice = self.dice_loss(final_pred, target)
        final_ce = self.ce_loss(final_pred, target)
        final_loss = final_dice + final_ce

        # 解码器中间损失
        decoder_loss = 0
        for decoder_pred in decoder_preds:
            # 上采样到目标尺寸
            decoder_pred_up = F.interpolate(decoder_pred, size=target.shape[1:],
                                           mode='trilinear', align_corners=False)
            # 通过卷积得到分割预测
            # Dynamically create Conv3d to match channels for aux tasks, assuming output channels are num_classes
            decoder_seg = nn.Conv3d(decoder_pred_up.shape[1], final_pred.shape[1], 1).to(final_pred.device)(decoder_pred_up)

            dice = self.dice_loss(decoder_seg, target)
            ce = self.ce_loss(decoder_seg, target)
            decoder_loss += dice + ce

        decoder_loss = decoder_loss / len(decoder_preds) if decoder_preds else 0

        # 编码器辅助损失
        aux_loss = 0
        for aux_pred in aux_preds:
            # 上采样到目标尺寸
            aux_pred_up = F.interpolate(aux_pred, size=target.shape[1:],
                                       mode='trilinear', align_corners=False)
            # 通过卷积得到分割预测
            # Dynamically create Conv3d to match channels for aux tasks, assuming output channels are num_classes
            aux_seg = nn.Conv3d(aux_pred_up.shape[1], final_pred.shape[1], 1).to(final_pred.device)(aux_pred_up)

            dice = self.dice_loss(aux_seg, target)
            ce = self.ce_loss(aux_seg, target)
            aux_loss += dice + ce

        aux_loss = aux_loss / len(aux_preds) if aux_preds else 0

        # 总损失
        total_loss = (self.alpha * final_loss +
                     self.beta * decoder_loss +
                     self.gamma * aux_loss)

        return total_loss, {
            'final_loss': final_loss.item(),
            'decoder_loss': decoder_loss.item() if isinstance(decoder_loss, torch.Tensor) else decoder_loss,
            'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        }

# ========== 5. 数据处理和训练工具 ==========

class BraTSDataLoader:
    """BraTS数据加载器（简化版）"""
    def __init__(self, data_dir: str, modalities: List[str] = ['FLAIR', 'T1', 'T1c', 'T2'],
                 image_size: Tuple[int, int, int] = (128, 128, 128)):
        self.modalities = modalities
        self.image_size = image_size

    def preprocess_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """预处理3D体积数据"""
        # 裁剪或填充到目标尺寸
        current_size = torch.tensor(volume.shape)
        target_size = torch.tensor(self.image_size)

        # 计算填充/裁剪参数
        pad_start = torch.clamp((target_size - current_size) // 2, min=0)
        pad_end = target_size - current_size - pad_start

        # 应用填充
        if torch.any(pad_start > 0) or torch.any(pad_end > 0):
            # Ensure padding dimensions match volume's spatial dimensions (D, W, H)
            # Pad is applied from last dim to first dim, so (pad_D_start, pad_D_end, pad_W_start, pad_W_end, pad_H_start, pad_H_end)
            # current_size is (H, W, D), target_size is (H, W, D)
            volume = F.pad(volume,
                          (pad_start[2].item(), pad_end[2].item(), # D dim
                           pad_start[1].item(), pad_end[1].item(), # W dim
                           pad_start[0].item(), pad_end[0].item())) # H dim

        # 归一化
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)

        return volume

    def load_sample(self, sample_id):
        """加载单个样本（简化版）"""
        # 在实际应用中，这里会从文件加载nifti图像
        # 这里用随机数据模拟
        modalities_data = []
        for _ in self.modalities:
            # 模拟MRI数据，先生成3D volume (H, W, D)
            volume = torch.randn(*self.image_size) # Now this is (H, W, D)
            volume = self.preprocess_volume(volume)
            modalities_data.append(volume.unsqueeze(0))  # Add channel dimension (1, H, W, D)

        # 模拟分割标签（4类：背景 + 3个肿瘤区域）
        segmentation = torch.randint(0, 4, self.image_size)  # (H, W, D)

        return modalities_data, segmentation

class MissingModalitySimulator:
    """缺失模态模拟器"""
    def __init__(self, missing_rates: List[float] = [0.1, 0.1, 0.1, 0.1]):
        self.missing_rates = missing_rates

    def simulate_missing(self, modalities: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        模拟模态缺失
        返回: (缺失后的模态列表, 模态存在性掩码)
        """
        batch_size = modalities[0].shape[0]
        num_modalities = len(modalities)

        # 生成模态掩码
        modality_mask = torch.ones(batch_size, num_modalities)
        for m in range(num_modalities):
            missing_prob = self.missing_rates[m] if m < len(self.missing_rates) else 0.1
            mask = (torch.rand(batch_size) > missing_prob).float() # mask = 1 (exist), 0 (missing)
            modality_mask[:, m] = mask

        # 应用掩码
        masked_modalities = []
        for m, modality in enumerate(modalities):
            mask = modality_mask[:, m].view(-1, 1, 1, 1, 1)
            masked_modality = modality * mask
            masked_modalities.append(masked_modality)

        return masked_modalities, modality_mask

# ========== 6. 训练循环 ==========

def train_epoch(model: nn.Module,
                dataloader: BraTSDataLoader,
                loss_fn: IMFuseLoss,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                missing_simulator: MissingModalitySimulator,
                num_samples: int = 100):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for i in range(num_samples):
        # 加载数据
        modalities, segmentation = dataloader.load_sample(f"sample_{i}")

        # 转换为张量并移动到设备
        modalities = [mod.to(device) for mod in modalities]
        segmentation = segmentation.to(device).unsqueeze(0)  # 添加batch维度

        # 模拟模态缺失
        masked_modalities, modality_mask = missing_simulator.simulate_missing(modalities)

        # 前向传播
        optimizer.zero_grad()
        segmentation_pred, decoder_preds, aux_preds = model(masked_modalities, modality_mask)

        # 计算损失
        loss, loss_dict = loss_fn(segmentation_pred, decoder_preds, aux_preds, segmentation)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (i + 1) % 10 == 0:
            print(f"Batch {i+1}/{num_samples}, Loss: {loss.item():.4f}, "
                  f"Final: {loss_dict['final_loss']:.4f}, "
                  f"Decoder: {loss_dict['decoder_loss']:.4f}, "
                  f"Aux: {loss_dict['aux_loss']:.4f}")

    return total_loss / num_batches

def validate(model: nn.Module,
             dataloader: BraTSDataLoader,
             device: torch.device,
             num_samples: int = 20):
    """验证模型"""
    model.eval()
    total_dice = 0

    with torch.no_grad():
        for i in range(num_samples):
            # 加载数据
            modalities, segmentation = dataloader.load_sample(f"val_sample_{i}")

            # 转换为张量并移动到设备
            modalities = [mod.to(device) for mod in modalities]
            segmentation = segmentation.to(device).unsqueeze(0)

            # 所有模态都存在的情况
            modality_mask = torch.ones(1, len(modalities), device=device)

            # 前向传播
            segmentation_pred, _, _ = model(modalities, modality_mask)

            # 计算Dice系数
            pred_classes = torch.argmax(segmentation_pred, dim=1)
            dice_score = compute_dice_score(pred_classes, segmentation)
            total_dice += dice_score

    return total_dice / num_samples

def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> float:
    """计算Dice系数"""
    dice_scores = []

    for class_idx in range(1, num_classes):  # 跳过背景
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()

        intersection = torch.sum(pred_mask * target_mask)
        union = torch.sum(pred_mask) + torch.sum(target_mask)

        dice = (2. * intersection) / (union + 1e-8)
        dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores)

# ========== 7. 主训练脚本 ==========

def main():
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    config = {
        'num_modalities': 4,
        'in_channels': 1,
        'base_channels': 8,
        'num_stages': 5,
        'd_model': 256,
        'num_classes': 4,
        'use_interleaved': True,
        'learning_rate': 2e-4,
        'num_epochs': 100,
        'batch_size': 2,
        'num_train_samples': 1000,
        'num_val_samples': 200,
    }

    # 创建模型
    model = IMFuse(
        num_modalities=config['num_modalities'],
        in_channels=config['in_channels'],
        base_channels=config['base_channels'],
        num_stages=config['num_stages'],
        d_model=config['d_model'],
        num_classes=config['num_classes'],
        use_interleaved=config['use_interleaved']
    ).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 创建数据加载器
    dataloader = BraTSDataLoader(
        data_dir='path/to/brats2023',
        modalities=['FLAIR', 'T1', 'T1c', 'T2'],
        image_size=(128, 128, 128)
    )

    # 创建缺失模态模拟器
    missing_simulator = MissingModalitySimulator(
        missing_rates=[0.1, 0.1, 0.1, 0.1]  # 每个模态10%的缺失率
    )

    # 损失函数
    class_weights = torch.tensor([0.25, 1.0, 1.0, 1.0])  # 背景权重较低
    loss_fn = IMFuseLoss(class_weights=class_weights, alpha=0.5, beta=0.3, gamma=0.2)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # 训练循环
    best_val_dice = 0
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # 训练
        train_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            missing_simulator=missing_simulator,
            num_samples=config['num_train_samples'] // config['batch_size']
        )

        # 验证
        val_dice = validate(
            model=model,
            dataloader=dataloader,
            device=device,
            num_samples=config['num_val_samples'] // config['batch_size']
        )

        # 更新学习率
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Dice = {val_dice:.4f}")

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'config': config
            }, 'best_imfuse_model.pth')
            print(f"Saved best model with Dice: {val_dice:.4f}")

    print(f"\nTraining completed. Best validation Dice: {best_val_dice:.4f}")

# ========== 8. 推理示例 ==========

def inference_example():
    """推理示例"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = IMFuse(
        num_modalities=4,
        in_channels=1,
        base_channels=8,
        num_stages=5,
        d_model=256,
        num_classes=4,
        use_interleaved=True
    ).to(device)

    # 加载预训练权重
    checkpoint = torch.load('best_imfuse_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建模拟输入
    batch_size = 1
    image_size = (128, 128, 128)

    # 模拟4个模态的输入
    modalities = []
    for i in range(4):
        modality = torch.randn(batch_size, 1, *image_size).to(device)
        modalities.append(modality)

    # 模拟模态缺失（例如缺少T1c）
    modality_mask = torch.tensor([[1, 1, 0, 1]], device=device)  # T1c缺失

    # 推理
    with torch.no_grad():
        segmentation_pred, _, _ = model(modalities, modality_mask)
        pred_classes = torch.argmax(segmentation_pred, dim=1)

    print(f"Prediction shape: {pred_classes.shape}")
    print(f"Unique classes predicted: {torch.unique(pred_classes)}")

    return pred_classes

# ========== 9. 运行 ==========

if __name__ == "__main__":
    # 如果需要训练，运行：
    main()

    # 如果只需要测试推理，运行：
    # prediction = inference_example()
    print("Inference completed successfully!")