"""
Gated Frequency Fusion Module (GFFM)
====================================
门控频域融合器 - 核心创新模块

这个模块智能地融合来自不同ViT层级的频域特征，
使用可学习的门控权重来自适应地确定每一层的重要性。

创新点:
1. 可学习的门控权重: 自动学习不同层级特征的重要性
2. Focus Token机制: 专门的token用于聚合频域伪造特征
3. Transformer融合: 使用注意力机制进行特征交互
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """FP16兼容的LayerNorm"""
    
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """快速GELU激活函数"""
    
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Transformer的残差注意力块
    
    包含:
    - Multi-head Self-Attention
    - Feed-Forward Network (MLP)
    - Layer Normalization
    - Residual Connections
    """
    
    def __init__(self, d_model: int, n_head: int, reduction_factor: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        # 多头自注意力
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        
        # 前馈网络 (MLP)
        # reduction_factor用于控制MLP的瓶颈维度，减少参数量
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // reduction_factor),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // reduction_factor, d_model),
            nn.Dropout(0.5)
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    
    def attention(self, x: torch.Tensor):
        """应用多头自注意力"""
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        Args:
            x: shape = [seq_len, batch_size, d_model]
        
        Returns:
            shape = [seq_len, batch_size, d_model]
        """
        # 自注意力 + 残差连接
        x = x + self.attention(self.ln_1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


class FrequencyFusionTransformer(nn.Module):
    """
    频域融合Transformer
    
    使用浅层Transformer来融合加权后的频域特征序列
    """
    
    def __init__(self, width: int, layers: int, heads: int, reduction_factor: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        
        # 堆叠多个残差注意力块
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, reduction_factor, attn_mask)
            for _ in range(layers)
        ])
    
    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        Args:
            x: shape = [seq_len, batch_size, width]
        
        Returns:
            out: 中间层输出字典 (可用于深度监督)
            x: 最终输出, shape = [seq_len, batch_size, width]
        """
        out = {}
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            # 保存每层第一个token (Focus Token) 的输出
            out[f'layer{idx}'] = x[0]
        
        return out, x


class GatedFrequencyFusionModule(nn.Module):
    """
    门控频域融合模块 (核心创新)
    
    功能:
    1. 使用可学习的门控权重对不同层级的频域特征进行加权
    2. 使用Focus Token聚合所有频域特征
    3. 通过Transformer进行特征交互和融合
    
    Args:
        feature_dim: 特征维度 (默认: 1024)
        num_layers: 融合的层数，即频域特征的数量 (默认: 12)
        transformer_layers: Transformer的层数 (默认: 2)
        transformer_heads: 注意力头数 (默认: 2)
        reduction_factor: MLP的缩减因子 (默认: 1)
        dropout_prob: Dropout概率 (默认: 0.5)
    """
    
    def __init__(
        self,
        feature_dim=1024,
        num_layers=12,
        transformer_layers=2,
        transformer_heads=2,
        reduction_factor=1,
        dropout_prob=0.5
    ):
        super(GatedFrequencyFusionModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # ============ 创新点 1: 可学习的层重要性门控权重 ============
        # 这是一个可训练的参数，用于学习每一层频域特征的重要性
        # shape: [num_layers]
        # 初始化为均匀分布，训练过程中会自动调整
        self.layer_importance_weights = nn.Parameter(torch.ones(num_layers))
        
        # ============ 创新点 2: Focus Frequency CLS Token ============
        # 这是一个专门用于聚合频域伪造特征的可学习token
        # 灵感来自论文2的Focus CLS，但这里专注于频域特征
        self.focus_freq_token = nn.Parameter(torch.zeros(feature_dim))
        
        # ============ Transformer融合器 ============
        # 使用浅层Transformer来融合特征序列
        self.transformer = FrequencyFusionTransformer(
            width=feature_dim,
            layers=transformer_layers,
            heads=transformer_heads,
            reduction_factor=reduction_factor
        )
        
        # Layer Normalization (在分类前应用)
        self.ln_post = nn.LayerNorm(feature_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        # 初始化Focus Token (使用正态分布)
        nn.init.normal_(self.focus_freq_token, std=0.02)
        
        # layer_importance_weights会在第一次使用时通过Softmax归一化
        # 所以初始化为1就可以，训练时会自动学习
    
    def forward(self, frequency_features):
        """
        前向传播
        
        Args:
            frequency_features: 频域特征列表
                               长度 = num_layers
                               每个元素 shape = [batch_size, feature_dim]
        
        Returns:
            fused_feature: 融合后的特征 (来自Focus Token)
                          shape = [batch_size, feature_dim]
        
        流程:
            1. 计算门控权重 (Softmax归一化)
            2. 对每个频域特征应用门控加权
            3. 拼接 Focus Token 和加权特征
            4. 送入Transformer进行融合
            5. 提取融合后的Focus Token作为输出
        """
        batch_size = frequency_features[0].shape[0]
        
        # ============ Step 1: 计算门控权重 ============
        # 使用Softmax确保权重和为1，表示相对重要性
        # shape: [num_layers]
        gate_weights = F.softmax(self.layer_importance_weights, dim=0)
        
        # ============ Step 2: 应用门控加权 ============
        # 将每个频域特征乘以对应的门控权重
        weighted_features = []
        for i, freq_feat in enumerate(frequency_features):
            # freq_feat shape: [B, D]
            # gate_weights[i] shape: scalar
            # weighted shape: [B, D]
            weighted = gate_weights[i] * freq_feat
            weighted_features.append(weighted)
        
        # ============ Step 3: 拼接Focus Token和加权特征 ============
        # Focus Token: [1, D] -> [B, 1, D]
        focus_token = self.focus_freq_token.view(1, 1, -1).repeat(batch_size, 1, 1)
        
        # 将所有加权特征堆叠: [B, num_layers, D]
        weighted_stack = torch.stack(weighted_features, dim=1)
        
        # 拼接: [B, 1 + num_layers, D]
        # 第一个token是Focus Token，后面是加权的频域特征
        feature_sequence = torch.cat([focus_token, weighted_stack], dim=1)
        
        # ============ Step 4: Transformer融合 ============
        # Transformer期望输入: [seq_len, batch_size, feature_dim]
        # 所以需要转置: [B, S, D] -> [S, B, D]
        feature_sequence = feature_sequence.permute(1, 0, 2)
        
        # 通过Transformer进行特征交互和融合
        _, fused_sequence = self.transformer(feature_sequence)
        
        # 转回: [S, B, D] -> [B, S, D]
        fused_sequence = fused_sequence.permute(1, 0, 2)
        
        # ============ Step 5: 提取融合后的Focus Token ============
        # 第一个token包含了所有频域特征的融合信息
        # shape: [B, D]
        fused_feature = self.ln_post(fused_sequence[:, 0, :])
        
        return fused_feature
    
    def get_gate_weights(self):
        """
        获取当前的门控权重 (用于可视化和分析)
        
        Returns:
            归一化后的门控权重, shape = [num_layers]
        """
        with torch.no_grad():
            return F.softmax(self.layer_importance_weights, dim=0)


# 测试代码
if __name__ == "__main__":
    print("测试 GatedFrequencyFusionModule...")
    
    # 创建模块
    gffm = GatedFrequencyFusionModule(
        feature_dim=1024,
        num_layers=12,
        transformer_layers=2,
        transformer_heads=2,
        reduction_factor=1
    )
    
    # 创建虚拟输入 (12个频域特征)
    batch_size = 4
    dummy_freq_features = [torch.randn(batch_size, 1024) for _ in range(12)]
    
    # 前向传播
    fused_feature = gffm(dummy_freq_features)
    
    print(f"输入: {len(dummy_freq_features)} 个频域特征")
    print(f"每个特征 shape: {dummy_freq_features[0].shape}")
    print(f"输出 shape: {fused_feature.shape}")
    
    # 查看门控权重
    gate_weights = gffm.get_gate_weights()
    print(f"\n门控权重: {gate_weights}")
    print(f"权重和: {gate_weights.sum().item()}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in gffm.parameters())
    trainable_params = sum(p.numel() for p in gffm.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
