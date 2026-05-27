"""
Hierarchical Wavelet-Gated-Former Network
=========================================
完整的网络架构，整合了三个核心模块：

1. 冻结的CLIP ViT-L/14 Backbone (提取分层特征)
2. Hierarchical Wavelet-Head (频域特征提取)
3. Gated Frequency Fusion Module (GFFM) (门控融合)
4. 分类器 (二分类: 真/假)

这个架构解决了两篇论文的局限性：
- Wavelet-CLIP: 只在最后的特征上应用小波变换
- ForgeLens: 融合的是通用的空间/语义特征

我们的创新: 在多个层级提取频域特征，并通过门控机制智能融合
"""

import torch
import torch.nn as nn
from models.network.clip import clip
from models.network.hierarchical_wavelet_head import HierarchicalWaveletHead
from models.network.gated_frequency_fusion_module import GatedFrequencyFusionModule


class LayerNorm(nn.LayerNorm):
    """FP16兼容的LayerNorm"""
    
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class WaveletGatedFormer(nn.Module):
    """
    Hierarchical Wavelet-Gated-Former 主模型
    
    架构流程:
    图像 -> CLIP ViT (frozen) -> 分层CLS tokens 
         -> Hierarchical Wavelet-Head -> 频域特征 
         -> GFFM -> 融合特征 -> 分类器 -> 预测
    
    Args:
        feature_dim: 特征维度 (默认: 1024 for ViT-L/14)
        num_wavelet_heads: 小波头数量，对应提取的ViT层数 (默认: 12)
        transformer_layers: GFFM中Transformer的层数 (默认: 2)
        transformer_heads: GFFM中注意力头数 (默认: 2)
        reduction_factor: MLP缩减因子 (默认: 1)
        dropout_prob: Dropout概率 (默认: 0.5)
        output_dim: 输出维度，二分类为2 (默认: 1，使用BCEWithLogitsLoss)
    """
    
    def __init__(
        self,
        feature_dim=1024,
        num_wavelet_heads=12,
        transformer_layers=2,
        transformer_heads=2,
        reduction_factor=1,
        dropout_prob=0.5,
        output_dim=1
    ):
        super(WaveletGatedFormer, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_wavelet_heads = num_wavelet_heads
        
        # ============ Module 0: 冻结的CLIP ViT-L/14 Backbone ============
        # 加载预训练的CLIP模型
        # 注意: 这里我们需要修改CLIP模型以提取分层的[CLS] tokens
        # 不使用WSGM模块
        print("加载 CLIP ViT-L/14 backbone...")
        self.backbone, _ = clip.load('ViT-L/14', device='cpu')
        
        # 冻结所有CLIP参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        print("CLIP backbone 已冻结")
        
        # ============ Module 1: 分层小波头 ============
        # 在多个ViT层级上提取并转换为频域特征
        print(f"初始化 Hierarchical Wavelet-Head (num_heads={num_wavelet_heads})...")
        self.hierarchical_wavelet_head = HierarchicalWaveletHead(
            feature_dim=feature_dim,
            num_heads=num_wavelet_heads,
            wave='db6',  # Daubechies 6 小波 (与Wavelet-CLIP相同)
            J=3,         # 3层小波分解
            dropout_prob=dropout_prob
        )
        
        # ============ Module 2: 门控频域融合器 (GFFM) ============
        # 核心创新: 使用门控权重智能融合频域特征
        print(f"初始化 GFFM (transformer_layers={transformer_layers}, heads={transformer_heads})...")
        self.gffm = GatedFrequencyFusionModule(
            feature_dim=feature_dim,
            num_layers=num_wavelet_heads,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            reduction_factor=reduction_factor,
            dropout_prob=dropout_prob
        )
        
        # ============ Module 3: 分类器 ============
        # 简单的线性分类器，输入是GFFM融合后的Focus Token
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(feature_dim, output_dim)
        )
        
        # 初始化分类器权重
        self._initialize_classifier()
        
        print("模型初始化完成!")
        self._print_model_info()
    
    def _initialize_classifier(self):
        """初始化分类器权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        trainable_ratio = (trainable_params / total_params) * 100
        
        print("\n" + "="*60)
        print("模型参数统计:")
        print(f"  总参数量: {total_params:.2f}M")
        print(f"  可训练参数: {trainable_params:.2f}M ({trainable_ratio:.2f}%)")
        print("="*60 + "\n")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像, shape = [batch_size, 3, 224, 224]
        
        Returns:
            result: 分类logits, shape = [batch_size, output_dim]
        
        完整流程:
            1. 通过冻结的CLIP ViT提取分层的[CLS] tokens
            2. 对每个[CLS] token应用小波变换，得到频域特征
            3. 通过GFFM融合所有频域特征
            4. 使用分类器进行二分类预测
        """
        
        # ============ Step 1: 提取分层[CLS] tokens ============
        # 使用CLIP的encode_image方法
        # 返回: (final_feature, cls_tokens_list)
        # cls_tokens_list包含从指定层提取的所有[CLS] tokens
        with torch.no_grad():
            _, cls_tokens = self.backbone.encode_image(x)
        
        # 根据num_wavelet_heads均匀采样[CLS] tokens
        # 例如: 如果ViT有24层，num_wavelet_heads=12，则每隔2层采样一次
        total_layers = len(cls_tokens)
        if total_layers < self.num_wavelet_heads:
            raise ValueError(
                f"ViT层数 ({total_layers}) 少于所需的小波头数量 ({self.num_wavelet_heads})"
            )
        
        # 均匀采样策略
        sampling_indices = [
            int(i * total_layers / self.num_wavelet_heads) 
            for i in range(self.num_wavelet_heads)
        ]
        sampled_cls_tokens = [cls_tokens[i] for i in sampling_indices]
        
        # ============ Step 2: 应用分层小波头 ============
        # 将每个[CLS] token转换为频域特征
        frequency_features = self.hierarchical_wavelet_head(sampled_cls_tokens)
        
        # ============ Step 3: 门控频域融合 ============
        # 使用GFFM融合所有频域特征
        fused_feature = self.gffm(frequency_features)
        
        # ============ Step 4: 分类 ============
        # 通过线性分类器得到最终预测
        result = self.classifier(fused_feature)
        
        return result
    
    def get_gate_weights(self):
        """
        获取GFFM的门控权重 (用于分析和可视化)
        
        Returns:
            门控权重, shape = [num_wavelet_heads]
        """
        return self.gffm.get_gate_weights()
    
    def freeze_backbone(self):
        """确保backbone保持冻结状态"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_wavelet_heads(self):
        """解冻小波头 (通常它们默认是可训练的)"""
        for param in self.hierarchical_wavelet_head.parameters():
            param.requires_grad = True
    
    def unfreeze_gffm(self):
        """解冻GFFM (通常它默认是可训练的)"""
        for param in self.gffm.parameters():
            param.requires_grad = True
    
    def unfreeze_classifier(self):
        """解冻分类器 (通常它默认是可训练的)"""
        for param in self.classifier.parameters():
            param.requires_grad = True


# 测试代码
if __name__ == "__main__":
    print("测试 WaveletGatedFormer 模型...")
    
    # 创建模型
    model = WaveletGatedFormer(
        feature_dim=1024,
        num_wavelet_heads=12,
        transformer_layers=2,
        transformer_heads=2,
        reduction_factor=1,
        dropout_prob=0.5,
        output_dim=1
    )
    
    # 创建虚拟输入
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\n输入图像 shape: {dummy_images.shape}")
    
    # 前向传播
    try:
        output = model(dummy_images)
        print(f"输出 shape: {output.shape}")
        print(f"输出值: {output}")
        
        # 查看门控权重
        gate_weights = model.get_gate_weights()
        print(f"\n门控权重分布:")
        for i, weight in enumerate(gate_weights):
            print(f"  Layer {i}: {weight.item():.4f}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
