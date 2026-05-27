"""
GFFM Ablation Study Variants
==============================
用于消融实验的 GFFM 变体版本

包含以下变体:
1. GatedFrequencyFusionModule_NoGate: 移除门控，使用均匀权重
2. GatedFrequencyFusionModule_NoFocus: 移除 Focus Token
3. GatedFrequencyFusionModule_NoTransformer: 移除 Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gated_frequency_fusion_module import (
    LayerNorm, 
    FrequencyFusionTransformer
)


class GatedFrequencyFusionModule_NoGate(nn.Module):
    """
    消融实验: 移除门控机制
    
    使用固定的均匀权重 (1/N) 代替可学习的门控权重
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
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # ============ 固定的均匀权重 (不可学习) ============
        # 使用 register_buffer 注册为非参数的缓冲区
        uniform_weights = torch.ones(num_layers) / num_layers
        self.register_buffer('gate_weights', uniform_weights)
        
        # Focus Token (保留)
        self.focus_freq_token = nn.Parameter(torch.zeros(feature_dim))
        
        # Transformer (保留)
        self.transformer = FrequencyFusionTransformer(
            width=feature_dim,
            layers=transformer_layers,
            heads=transformer_heads,
            reduction_factor=reduction_factor
        )
        
        self.ln_post = nn.LayerNorm(feature_dim)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.focus_freq_token, std=0.02)
    
    def forward(self, frequency_features):
        batch_size = frequency_features[0].shape[0]
        
        # Step 1: 使用固定的均匀权重 (无 Softmax)
        gate_weights = self.gate_weights  # [num_layers], 每个=1/12
        
        # Step 2: 应用固定权重
        weighted_features = []
        for i, freq_feat in enumerate(frequency_features):
            weighted = gate_weights[i] * freq_feat
            weighted_features.append(weighted)
        
        # Step 3-5: 与原版相同
        focus_token = self.focus_freq_token.view(1, 1, -1).repeat(batch_size, 1, 1)
        weighted_stack = torch.stack(weighted_features, dim=1)
        feature_sequence = torch.cat([focus_token, weighted_stack], dim=1)
        
        feature_sequence = feature_sequence.permute(1, 0, 2)
        _, fused_sequence = self.transformer(feature_sequence)
        fused_sequence = fused_sequence.permute(1, 0, 2)
        
        fused_feature = self.ln_post(fused_sequence[:, 0, :])
        return fused_feature
    
    def get_gate_weights(self):
        """返回固定的均匀权重"""
        return self.gate_weights


class GatedFrequencyFusionModule_NoFocus(nn.Module):
    """
    消融实验: 移除 Focus Token
    
    直接对加权特征进行 Transformer 处理，使用第一个特征作为输出
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
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # 门控权重 (保留)
        self.layer_importance_weights = nn.Parameter(torch.ones(num_layers))
        
        # 移除 Focus Token
        # self.focus_freq_token = ...
        
        # Transformer (保留)
        self.transformer = FrequencyFusionTransformer(
            width=feature_dim,
            layers=transformer_layers,
            heads=transformer_heads,
            reduction_factor=reduction_factor
        )
        
        self.ln_post = nn.LayerNorm(feature_dim)
    
    def forward(self, frequency_features):
        batch_size = frequency_features[0].shape[0]
        
        # Step 1: 计算门控权重
        gate_weights = F.softmax(self.layer_importance_weights, dim=0)
        
        # Step 2: 应用门控加权
        weighted_features = []
        for i, freq_feat in enumerate(frequency_features):
            weighted = gate_weights[i] * freq_feat
            weighted_features.append(weighted)
        
        # Step 3: 直接堆叠加权特征 (无 Focus Token)
        weighted_stack = torch.stack(weighted_features, dim=1)  # [B, 12, D]
        
        # Step 4: Transformer 处理
        feature_sequence = weighted_stack.permute(1, 0, 2)  # [12, B, D]
        _, fused_sequence = self.transformer(feature_sequence)
        fused_sequence = fused_sequence.permute(1, 0, 2)  # [B, 12, D]
        
        # Step 5: 使用第一个位置的特征作为输出 (代替 Focus Token)
        fused_feature = self.ln_post(fused_sequence[:, 0, :])
        return fused_feature
    
    def get_gate_weights(self):
        with torch.no_grad():
            return F.softmax(self.layer_importance_weights, dim=0)


class GatedFrequencyFusionModule_NoTransformer(nn.Module):
    """
    消融实验: 移除 Transformer
    
    使用简单的加权平均代替 Transformer 融合
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
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # 门控权重 (保留)
        self.layer_importance_weights = nn.Parameter(torch.ones(num_layers))
        
        # Focus Token (保留，但不使用)
        self.focus_freq_token = nn.Parameter(torch.zeros(feature_dim))
        
        # 移除 Transformer
        # self.transformer = ...
        
        self.ln_post = nn.LayerNorm(feature_dim)
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.focus_freq_token, std=0.02)
    
    def forward(self, frequency_features):
        # Step 1: 计算门控权重
        gate_weights = F.softmax(self.layer_importance_weights, dim=0)
        
        # Step 2: 应用门控加权
        weighted_features = []
        for i, freq_feat in enumerate(frequency_features):
            weighted = gate_weights[i] * freq_feat
            weighted_features.append(weighted)
        
        # Step 3: 简单平均融合 (代替 Transformer)
        weighted_stack = torch.stack(weighted_features, dim=1)  # [B, 12, D]
        fused_feature = weighted_stack.mean(dim=1)  # [B, D]
        
        # Step 4: LayerNorm
        fused_feature = self.ln_post(fused_feature)
        return fused_feature
    
    def get_gate_weights(self):
        with torch.no_grad():
            return F.softmax(self.layer_importance_weights, dim=0)


class GatedFrequencyFusionModule_WeightedSum(nn.Module):
    """
    消融实验: 仅使用加权求和
    
    移除 Focus Token 和 Transformer，直接加权求和
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
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # 门控权重 (保留)
        self.layer_importance_weights = nn.Parameter(torch.ones(num_layers))
        
        self.ln_post = nn.LayerNorm(feature_dim)
    
    def forward(self, frequency_features):
        # Step 1: 计算门控权重
        gate_weights = F.softmax(self.layer_importance_weights, dim=0)
        
        # Step 2: 加权求和
        fused_feature = sum(gate_weights[i] * freq_feat 
                           for i, freq_feat in enumerate(frequency_features))
        
        # Step 3: LayerNorm
        fused_feature = self.ln_post(fused_feature)
        return fused_feature
    
    def get_gate_weights(self):
        with torch.no_grad():
            return F.softmax(self.layer_importance_weights, dim=0)


# 用于快速切换的工厂函数
def get_gffm_variant(variant='full', **kwargs):
    """
    根据变体名称返回对应的 GFFM 模块
    
    Args:
        variant: 'full', 'no_gate', 'no_focus', 'no_transformer', 'weighted_sum'
        **kwargs: GFFM 的配置参数
    
    Returns:
        GFFM 模块实例
    """
    from .gated_frequency_fusion_module import GatedFrequencyFusionModule
    
    variants = {
        'full': GatedFrequencyFusionModule,
        'no_gate': GatedFrequencyFusionModule_NoGate,
        'no_focus': GatedFrequencyFusionModule_NoFocus,
        'no_transformer': GatedFrequencyFusionModule_NoTransformer,
        'weighted_sum': GatedFrequencyFusionModule_WeightedSum,
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. "
                        f"Available: {list(variants.keys())}")
    
    return variants[variant](**kwargs)


# 测试代码
if __name__ == "__main__":
    print("测试 GFFM 消融实验变体...")
    
    # 创建虚拟输入
    batch_size = 4
    dummy_freq_features = [torch.randn(batch_size, 1024) for _ in range(12)]
    
    variants = ['full', 'no_gate', 'no_focus', 'no_transformer', 'weighted_sum']
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"测试变体: {variant}")
        print('='*60)
        
        # 创建模块
        gffm = get_gffm_variant(
            variant=variant,
            feature_dim=1024,
            num_layers=12,
            transformer_layers=2,
            transformer_heads=2,
            reduction_factor=1
        )
        
        # 前向传播
        output = gffm(dummy_freq_features)
        
        print(f"输出 shape: {output.shape}")
        
        # 统计参数
        total_params = sum(p.numel() for p in gffm.parameters())
        trainable_params = sum(p.numel() for p in gffm.parameters() if p.requires_grad)
        
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 门控权重
        try:
            gate_weights = gffm.get_gate_weights()
            print(f"门控权重和: {gate_weights.sum().item():.4f}")
            print(f"门控权重: {gate_weights[:3].tolist()} ...")
        except:
            print("无门控权重")
    
    print("\n" + "="*60)
    print("所有变体测试完成!")
    print("="*60)
