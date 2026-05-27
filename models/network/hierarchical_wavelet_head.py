"""
Hierarchical Wavelet-Head Module
================================
这个模块在ViT的多个层级上提取[CLS] token并应用小波变换，
将空域特征转换为频域特征。

灵感来自 Wavelet-CLIP 论文，但应用在分层特征提取上。
"""

import torch
import torch.nn as nn

try:
    from pytorch_wavelets import DWT1DForward, DWT1DInverse
except ImportError:
    raise ImportError("请安装 pytorch_wavelets: pip install pytorch_wavelets")


class WaveletHead(nn.Module):
    """
    单个小波头模块
    
    对单个[CLS] token应用小波变换，提取频域特征。
    
    Args:
        feature_dim: 输入特征维度 (例如 1024 for CLIP ViT-L/14)
        wave: 小波基函数类型 (默认: 'db6' - Daubechies 6)
        J: 小波分解层数 (默认: 3)
        dropout_prob: Dropout概率 (默认: 0.5)
    """
    
    def __init__(self, feature_dim=1024, wave='db6', J=3, dropout_prob=0.5):
        super(WaveletHead, self).__init__()
        
        self.feature_dim = feature_dim
        self.dropout_prob = dropout_prob
        
        # 初始化小波变换
        # DWT1DForward: 离散小波变换 (Discrete Wavelet Transform)
        # 输出: (yl, yh) 其中 yl是低频系数，yh是高频系数列表
        self.dwt = DWT1DForward(wave=wave, J=J)
        
        # IDWT: 逆离散小波变换 (Inverse Discrete Wavelet Transform)
        # 用于从小波系数重构信号
        self.idwt = DWT1DInverse(wave=wave)
        
        # MLP 和投影层将在第一次前向传播时根据实际的维度创建
        self.slp = None
        self.projection = None  # 用于将IDWT重构后的维度投影回feature_dim
        self.low_freq_dim = None
        self.reconstructed_dim = None
        
        print(f"  WaveletHead initialized: feature_dim={feature_dim}, wave={wave}, J={J} (Layers will be created on first forward)")
    
    def _initialize_weights(self):
        """初始化MLP权重"""
        for m in self.slp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入的[CLS] token, shape = [batch_size, feature_dim]
        
        Returns:
            频域转换后的特征, shape = [batch_size, feature_dim]
        
        流程:
            1. [fv_low, fv_high] = DWT(x)  # 小波分解
            2. fv_low' = MLP(fv_low)        # 处理低频系数
            3. x_new = IDWT([fv_low', fv_high])  # 重构信号
        """
        # 保存原始数据类型（可能是float16用于混合精度训练）
        original_dtype = x.dtype
        
        # 使用autocast(enabled=False)禁用小波变换部分的混合精度
        # 因为pytorch_wavelets不支持float16（包括反向传播）
        with torch.cuda.amp.autocast(enabled=False):
            # 转换为float32
            x = x.float()
            
            # 添加channel维度以适配DWT: [B, D] -> [B, 1, D]
            x_unsqueezed = x.unsqueeze(1)
            
            # Step 1: 应用小波变换，分解为低频和高频系数
            # yl: 低频系数 (approximation coefficients), shape = [B, 1, low_freq_dim]
            # yh: 高频系数列表 (detail coefficients)
            yl, yh = self.dwt(x_unsqueezed)
        
            # 延迟初始化：在第一次前向传播时根据实际维度创建 MLP 和投影层
            if self.slp is None:
                batch_size, channels, actual_low_freq_dim = yl.shape
                self.low_freq_dim = actual_low_freq_dim
                
                # 创建 MLP 处理低频系数
                self.slp = nn.Sequential(
                    nn.Linear(self.low_freq_dim, self.low_freq_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob)
                ).to(x.device)
                
                # 初始化 MLP 权重
                self._initialize_weights()
                
                print(f"  [First Forward] MLP created with low_freq_dim={self.low_freq_dim}")
            
            # Step 2: 通过可训练的MLP处理低频系数
            # yl shape: [B, 1, low_freq_dim]
            batch_size = yl.shape[0]
            yl_reshaped = yl.view(batch_size, -1)  # [B, low_freq_dim]
            
            # 通过MLP处理
            yl_new = self.slp(yl_reshaped)  # [B, low_freq_dim]
            
            # 恢复形状用于IDWT
            yl_new = yl_new.unsqueeze(1)  # [B, 1, low_freq_dim]
            
            # Step 3: 使用处理后的低频系数和原始高频系数重构信号
            x_reconstructed = self.idwt((yl_new, yh))
            
            # 移除channel维度: [B, 1, D'] -> [B, D']
            x_reconstructed = x_reconstructed.squeeze(1)
            
            # 延迟创建投影层：如果重构维度与输入不同，创建投影层
            if self.projection is None:
                self.reconstructed_dim = x_reconstructed.shape[-1]
                
                if self.reconstructed_dim != self.feature_dim:
                    # 需要投影回原始维度
                    self.projection = nn.Linear(self.reconstructed_dim, self.feature_dim).to(x.device)
                    nn.init.xavier_uniform_(self.projection.weight)
                    nn.init.constant_(self.projection.bias, 0)
                    print(f"  [First Forward] Projection created: {self.reconstructed_dim} -> {self.feature_dim}")
                else:
                    # 维度匹配，不需要投影
                    self.projection = nn.Identity()
                    print(f"  [First Forward] No projection needed: {self.reconstructed_dim} == {self.feature_dim}")
            
            # Step 4: 投影回原始维度（如果需要）
            output = self.projection(x_reconstructed)
        
        # 转换回原始数据类型（如果使用混合精度训练，转回float16）
        output = output.to(original_dtype)
        
        return output


class HierarchicalWaveletHead(nn.Module):
    """
    分层小波头模块
    
    在ViT的多个层级上应用小波变换，生成分层的频域特征。
    
    Args:
        feature_dim: 特征维度 (默认: 1024 for CLIP ViT-L/14)
        num_heads: 小波头的数量，对应提取的ViT层数 (默认: 12)
        wave: 小波基函数 (默认: 'db6')
        J: 小波分解层数 (默认: 3)
        dropout_prob: Dropout概率 (默认: 0.5)
    """
    
    def __init__(self, feature_dim=1024, num_heads=12, wave='db6', J=3, dropout_prob=0.5):
        super(HierarchicalWaveletHead, self).__init__()
        
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        # 为每个提取的层创建一个独立的小波头
        # 这允许模型为不同层级学习不同的频域特征提取策略
        self.wavelet_heads = nn.ModuleList([
            WaveletHead(feature_dim, wave, J, dropout_prob)
            for _ in range(num_heads)
        ])
    
    def forward(self, cls_tokens):
        """
        前向传播
        
        Args:
            cls_tokens: 来自ViT不同层的[CLS] token列表
                       每个元素shape = [batch_size, feature_dim]
                       列表长度 = num_heads
        
        Returns:
            频域特征列表，每个元素shape = [batch_size, feature_dim]
        
        例如，如果num_heads=12 (对应ViT-L/14的24层中均匀采样):
            - cls_tokens[0]: 第2层的[CLS] token
            - cls_tokens[1]: 第4层的[CLS] token
            - ...
            - cls_tokens[11]: 第24层的[CLS] token
        """
        if len(cls_tokens) != self.num_heads:
            raise ValueError(
                f"期望 {self.num_heads} 个CLS tokens，但得到了 {len(cls_tokens)} 个"
            )
        
        # 对每个[CLS] token应用对应的小波头
        frequency_features = []
        for i, (cls_token, wavelet_head) in enumerate(zip(cls_tokens, self.wavelet_heads)):
            freq_feature = wavelet_head(cls_token)
            frequency_features.append(freq_feature)
        
        return frequency_features


# 测试代码
if __name__ == "__main__":
    # 测试单个小波头
    print("测试 WaveletHead...")
    wavelet_head = WaveletHead(feature_dim=1024)
    dummy_cls = torch.randn(4, 1024)  # batch_size=4
    output = wavelet_head(dummy_cls)
    print(f"输入shape: {dummy_cls.shape}")
    print(f"输出shape: {output.shape}")
    print(f"低频维度: {wavelet_head.low_freq_dim}")
    
    print("\n测试 HierarchicalWaveletHead...")
    hierarchical_head = HierarchicalWaveletHead(feature_dim=1024, num_heads=12)
    dummy_cls_tokens = [torch.randn(4, 1024) for _ in range(12)]
    freq_features = hierarchical_head(dummy_cls_tokens)
    print(f"输入: {len(dummy_cls_tokens)} 个CLS tokens")
    print(f"输出: {len(freq_features)} 个频域特征")
    print(f"每个频域特征shape: {freq_features[0].shape}")
