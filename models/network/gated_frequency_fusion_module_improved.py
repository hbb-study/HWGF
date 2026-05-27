"""
Gated Frequency Fusion Module - IMPROVED VERSION
=================================================
改进的门控频域融合器 - 用于促进门控权重分化

改进点：
1. ✅ 随机初始化门控权重（打破对称性）
2. ✅ 添加可选的多样性正则化损失
3. ✅ 提供权重分析工具

使用方法：
    # 替换原始 GFFM
    from models.network.gated_frequency_fusion_module_improved import GatedFrequencyFusionModule_Improved
    
    gffm = GatedFrequencyFusionModule_Improved(
        feature_dim=1024,
        num_layers=12,
        init_method='random',  # 'uniform', 'random', 'truncated_normal'
        init_std=0.5,          # 初始化标准差
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入原始模块的辅助类
from .gated_frequency_fusion_module import (
    LayerNorm, 
    QuickGELU, 
    FrequencyFusionTransformer
)


class GatedFrequencyFusionModule_Improved(nn.Module):
    """
    改进的门控频域融合模块
    
    ============================================
    🔧 改动 1: 初始化策略
    ============================================
    原始版本：torch.ones(num_layers) → 全部为1 → Softmax后=1/12
    改进版本：支持多种初始化方式，打破对称性
    
    Args:
        feature_dim: 特征维度 (默认: 1024)
        num_layers: 融合的层数 (默认: 12)
        transformer_layers: Transformer层数 (默认: 2)
        transformer_heads: 注意力头数 (默认: 2)
        reduction_factor: MLP缩减因子 (默认: 1)
        dropout_prob: Dropout概率 (默认: 0.5)
        
        ⭐ 新增参数:
        init_method: 初始化方法 ('uniform', 'random', 'truncated_normal', 'xavier')
        init_std: 初始化标准差 (用于random和truncated_normal)
    """
    
    def __init__(
        self,
        feature_dim=1024,
        num_layers=12,
        transformer_layers=2,
        transformer_heads=2,
        reduction_factor=1,
        dropout_prob=0.5,
        # ⭐ 新增参数
        init_method='random',
        init_std=0.5
    ):
        super(GatedFrequencyFusionModule_Improved, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.init_method = init_method
        self.init_std = init_std
        
        # ============================================
        # 🔧 改动 1: 改进的门控权重初始化
        # ============================================
        # 原始代码:
        # self.layer_importance_weights = nn.Parameter(torch.ones(num_layers))
        
        # 改进代码: 根据init_method选择不同的初始化策略
        if init_method == 'uniform':
            # 方法1: 均匀分布 (与原始相同，用于对比)
            init_weights = torch.ones(num_layers)
            
        elif init_method == 'random':
            # 方法2: 随机正态分布 (推荐)
            # 打破对称性，让模型从不同的起点开始学习
            init_weights = torch.randn(num_layers) * init_std
            
        elif init_method == 'truncated_normal':
            # 方法3: 截断正态分布
            # 避免初始值过大或过小
            init_weights = torch.randn(num_layers) * init_std
            init_weights = torch.clamp(init_weights, -2*init_std, 2*init_std)
            
        elif init_method == 'xavier':
            # 方法4: Xavier初始化
            init_weights = torch.empty(num_layers)
            nn.init.xavier_uniform_(init_weights.unsqueeze(0))
            init_weights = init_weights.squeeze()
            
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        self.layer_importance_weights = nn.Parameter(init_weights)
        
        print(f"\n{'='*60}")
        print(f"🔧 改进的GFFM初始化")
        print(f"{'='*60}")
        print(f"初始化方法: {init_method}")
        print(f"初始化标准差: {init_std}")
        print(f"初始权重 (Softmax前):")
        print(f"  均值: {init_weights.mean():.4f}")
        print(f"  标准差: {init_weights.std():.4f}")
        print(f"  最大: {init_weights.max():.4f}")
        print(f"  最小: {init_weights.min():.4f}")
        
        # Softmax后的权重
        with torch.no_grad():
            init_gate_weights = F.softmax(init_weights, dim=0)
        print(f"\n初始门控权重 (Softmax后):")
        print(f"  均值: {init_gate_weights.mean():.4f} (理想值: {1/num_layers:.4f})")
        print(f"  标准差: {init_gate_weights.std():.4f}")
        print(f"  最大: {init_gate_weights.max():.4f}")
        print(f"  最小: {init_gate_weights.min():.4f}")
        print(f"{'='*60}\n")
        
        # ============ 创新点 2: Focus Frequency Token ============
        self.focus_freq_token = nn.Parameter(torch.zeros(feature_dim))
        
        # ============ Transformer融合器 ============
        self.transformer = FrequencyFusionTransformer(
            width=feature_dim,
            layers=transformer_layers,
            heads=transformer_heads,
            reduction_factor=reduction_factor
        )
        
        # Layer Normalization
        self.ln_post = nn.LayerNorm(feature_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化其他权重"""
        # 初始化Focus Token
        nn.init.normal_(self.focus_freq_token, std=0.02)
    
    def forward(self, frequency_features):
        """
        前向传播
        
        Args:
            frequency_features: 列表，长度=num_layers
                               每个元素 shape = [batch_size, feature_dim]
        
        Returns:
            fused_feature: [batch_size, feature_dim]
        """
        batch_size = frequency_features[0].shape[0]
        
        # ============ Step 1: 计算门控权重 (Softmax归一化) ============
        gate_weights = F.softmax(self.layer_importance_weights, dim=0)
        
        # ============ Step 2: 应用门控加权 ============
        weighted_features = []
        for i, freq_feat in enumerate(frequency_features):
            weighted = gate_weights[i] * freq_feat
            weighted_features.append(weighted)
        
        # ============ Step 3: 准备Focus Token ============
        focus_token = self.focus_freq_token.view(1, 1, -1).repeat(batch_size, 1, 1)
        
        # ============ Step 4: 构建特征序列 ============
        weighted_stack = torch.stack(weighted_features, dim=1)
        feature_sequence = torch.cat([focus_token, weighted_stack], dim=1)
        
        # ============ Step 5: Transformer融合 ============
        feature_sequence = feature_sequence.permute(1, 0, 2)
        _, fused_sequence = self.transformer(feature_sequence)
        fused_sequence = fused_sequence.permute(1, 0, 2)
        
        # ============ Step 6: 提取Focus Token ============
        fused_feature = self.ln_post(fused_sequence[:, 0, :])
        
        return fused_feature
    
    def get_gate_weights(self):
        """获取当前的门控权重"""
        with torch.no_grad():
            return F.softmax(self.layer_importance_weights, dim=0)
    
    # ============================================
    # 🔧 改动 2: 新增多样性正则化损失
    # ============================================
    def compute_diversity_loss(self, loss_type='variance'):
        """
        计算门控权重的多样性损失
        
        目的: 鼓励权重分化，但防止过度集中
        
        Args:
            loss_type: 损失类型
                - 'variance': 鼓励方差增大，但限制最大权重
                - 'entropy': 熵正则化 (最稳定，推荐)
                - 'l2': L2距离到均匀分布
        
        Returns:
            diversity_loss: 标量张量（非负）
        """
        gate_weights = F.softmax(self.layer_importance_weights, dim=0)
        
        if loss_type == 'variance':
            # 方法1: 鼓励方差，但惩罚过度集中
            # 目标：方差适中（0.01-0.05），防止单个权重过大（>0.5）
            variance = torch.var(gate_weights)
            max_weight = torch.max(gate_weights)
            
            # 方差损失：目标方差为 0.02 (对应 std ≈ 0.14)
            target_var = 0.02
            var_loss = (variance - target_var) ** 2
            
            # 过度集中惩罚：单个权重超过 0.5 时惩罚
            concentration_penalty = torch.relu(max_weight - 0.5) ** 2
            
            diversity_loss = var_loss + 10.0 * concentration_penalty
            
        elif loss_type == 'entropy':
            # 方法2: 熵正则化（最稳定）
            # 目标：熵适中，既不均匀也不过度集中
            epsilon = 1e-8
            entropy = -torch.sum(gate_weights * torch.log(gate_weights + epsilon))
            
            # 最大熵 = log(12) ≈ 2.48（均匀分布）
            # 目标熵 ≈ 2.0（适度分化）
            max_entropy = torch.log(torch.tensor(self.num_layers, dtype=torch.float32))
            target_entropy = max_entropy * 0.8  # 80% 的最大熵
            
            # 熵损失：目标是适中的熵值
            diversity_loss = (entropy - target_entropy) ** 2
            
        elif loss_type == 'l2':
            # 方法3: L2 距离 + 集中度惩罚
            # 鼓励远离均匀分布，但防止过度集中
            uniform = torch.ones_like(gate_weights) / self.num_layers
            l2_dist = torch.sum((gate_weights - uniform) ** 2)
            
            # 目标：适度偏离均匀分布
            target_l2 = 0.02
            l2_loss = (l2_dist - target_l2) ** 2
            
            # 过度集中惩罚
            max_weight = torch.max(gate_weights)
            concentration_penalty = torch.relu(max_weight - 0.5) ** 2
            
            diversity_loss = l2_loss + 10.0 * concentration_penalty
            
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        return diversity_loss
    
    # ============================================
    # 🔧 改动 3: 新增权重分析工具
    # ============================================
    def analyze_weights(self, verbose=True):
        """
        分析当前门控权重的分布
        
        Returns:
            stats: 字典，包含统计信息
        """
        with torch.no_grad():
            gate_weights = F.softmax(self.layer_importance_weights, dim=0).cpu().numpy()
        
        stats = {
            'mean': float(gate_weights.mean()),
            'std': float(gate_weights.std()),
            'max': float(gate_weights.max()),
            'min': float(gate_weights.min()),
            'max_idx': int(gate_weights.argmax()),
            'min_idx': int(gate_weights.argmin()),
            'range': float(gate_weights.max() - gate_weights.min()),
            'cv': float(gate_weights.std() / gate_weights.mean()),  # 变异系数
        }
        
        # 计算与均匀分布的距离
        uniform = 1.0 / self.num_layers
        stats['distance_to_uniform'] = float(((gate_weights - uniform) ** 2).sum())
        
        # 计算熵
        epsilon = 1e-8
        entropy = -sum(w * np.log(w + epsilon) for w in gate_weights)
        stats['entropy'] = float(entropy)
        stats['max_entropy'] = float(np.log(self.num_layers))  # 均匀分布的熵
        stats['entropy_ratio'] = stats['entropy'] / stats['max_entropy']
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"门控权重分析")
            print(f"{'='*60}")
            print(f"均值: {stats['mean']:.6f} (理想均匀: {uniform:.6f})")
            print(f"标准差: {stats['std']:.6f}")
            print(f"最大值: {stats['max']:.6f} (Layer {stats['max_idx']*2})")
            print(f"最小值: {stats['min']:.6f} (Layer {stats['min_idx']*2})")
            print(f"范围: {stats['range']:.6f}")
            print(f"变异系数: {stats['cv']:.6f}")
            print(f"到均匀分布距离: {stats['distance_to_uniform']:.6f}")
            print(f"熵: {stats['entropy']:.4f} / {stats['max_entropy']:.4f} ({stats['entropy_ratio']*100:.1f}%)")
            
            if stats['std'] < 0.001:
                print("\n⚠️  权重几乎均匀分布 (std < 0.001)")
            elif stats['std'] < 0.01:
                print("\n⚠️  权重分布较均匀 (std < 0.01)")
            else:
                print(f"\n✓ 权重已分化 (std = {stats['std']:.4f})")
            
            print(f"{'='*60}\n")
        
        return stats


# 导入numpy用于分析
import numpy as np


# ============================================
# 测试代码
# ============================================
if __name__ == "__main__":
    print("测试改进的GFFM模块...\n")
    
    # 测试不同的初始化方法
    init_methods = ['uniform', 'random', 'truncated_normal', 'xavier']
    
    for method in init_methods:
        print(f"\n{'#'*70}")
        print(f"测试初始化方法: {method}")
        print(f"{'#'*70}")
        
        gffm = GatedFrequencyFusionModule_Improved(
            feature_dim=1024,
            num_layers=12,
            init_method=method,
            init_std=0.5
        )
        
        # 前向传播测试
        batch_size = 4
        dummy_features = [torch.randn(batch_size, 1024) for _ in range(12)]
        output = gffm(dummy_features)
        print(f"输出 shape: {output.shape}")
        
        # 分析权重
        stats = gffm.analyze_weights(verbose=True)
        
        # 测试多样性损失
        div_loss = gffm.compute_diversity_loss(loss_type='variance')
        print(f"多样性损失 (variance): {div_loss.item():.6f}")
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)
