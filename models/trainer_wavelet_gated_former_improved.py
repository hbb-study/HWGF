"""
Trainer for Hierarchical Wavelet-Gated-Former - IMPROVED VERSION
================================================================
改进的训练器 - 用于促进门控权重分化

改进点：
1. ✅ 为门控权重设置独立的更大学习率
2. ✅ 在训练损失中添加多样性正则化
3. ✅ 增强的权重监控和可视化

使用方法:
    from models.trainer_wavelet_gated_former_improved import Trainer_WaveletGatedFormer_Improved
    
    trainer = Trainer_WaveletGatedFormer_Improved(
        opt,
        gate_lr_multiplier=50,      # 门控学习率倍数
        diversity_weight=0.1,       # 多样性损失权重
        diversity_loss_type='variance'  # 多样性损失类型
    )
"""

import os
import numpy as np
from torch.optim import lr_scheduler
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from models.network.net_wavelet_gated_former import WaveletGatedFormer


class Trainer_WaveletGatedFormer_Improved:
    """
    改进的 Hierarchical Wavelet-Gated-Former 训练器
    
    ============================================
    🔧 改动说明
    ============================================
    1. 独立学习率: 为门控权重设置更大的学习率
    2. 多样性正则化: 鼓励权重分化
    3. 增强监控: 详细记录权重变化
    """
    
    def __init__(
        self, 
        opt,
        # ⭐ 新增参数
        gate_lr_multiplier=50,           # 门控学习率倍数
        diversity_weight=0.1,            # 多样性损失权重
        diversity_loss_type='variance',  # 'variance', 'entropy', 'l2'
        use_improved_gffm=True           # 是否使用改进的GFFM初始化
    ):
        """
        初始化改进的训练器
        
        Args:
            opt: 配置选项对象
            gate_lr_multiplier: 门控权重学习率倍数 (默认: 50倍)
            diversity_weight: 多样性损失权重 (默认: 0.1)
            diversity_loss_type: 多样性损失类型 (默认: 'variance')
            use_improved_gffm: 是否使用改进的GFFM (默认: True)
        """
        print("="*70)
        print("🔧 初始化改进的 Wavelet-Gated-Former 训练器")
        print("="*70)
        
        self.gate_lr_multiplier = gate_lr_multiplier
        self.diversity_weight = diversity_weight
        self.diversity_loss_type = diversity_loss_type
        self.use_improved_gffm = use_improved_gffm
        
        print(f"\n改进配置:")
        print(f"  门控学习率倍数: {gate_lr_multiplier}x")
        print(f"  多样性损失权重: {diversity_weight}")
        print(f"  多样性损失类型: {diversity_loss_type}")
        print(f"  使用改进GFFM: {use_improved_gffm}\n")
        
        # ============================================
        # 创建模型 (根据配置选择GFFM版本)
        # ============================================
        if use_improved_gffm:
            # 使用改进的GFFM模块
            print("使用改进的GFFM模块 (随机初始化)")
            from models.network.gated_frequency_fusion_module_improved import GatedFrequencyFusionModule_Improved
            
            # 先创建模型
            self.model = WaveletGatedFormer(
                feature_dim=1024,
                num_wavelet_heads=opt.num_wavelet_heads,
                transformer_layers=opt.gffm_transformer_layers,
                transformer_heads=opt.gffm_transformer_heads,
                reduction_factor=opt.gffm_reduction_factor,
                dropout_prob=0.5,
                output_dim=1
            )
            
            # 替换GFFM模块为改进版本
            self.model.gffm = GatedFrequencyFusionModule_Improved(
                feature_dim=1024,
                num_layers=opt.num_wavelet_heads,
                transformer_layers=opt.gffm_transformer_layers,
                transformer_heads=opt.gffm_transformer_heads,
                reduction_factor=opt.gffm_reduction_factor,
                dropout_prob=0.5,
                init_method='random',      # 随机初始化
                init_std=0.5              # 标准差
            )
        else:
            # 使用原始GFFM模块
            print("使用原始GFFM模块 (均匀初始化)")
            self.model = WaveletGatedFormer(
                feature_dim=1024,
                num_wavelet_heads=opt.num_wavelet_heads,
                transformer_layers=opt.gffm_transformer_layers,
                transformer_heads=opt.gffm_transformer_heads,
                reduction_factor=opt.gffm_reduction_factor,
                dropout_prob=0.5,
                output_dim=1
            )
        
        # 打印模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        trainable_ratio = (trainable_params / total_params) * 100
        
        print(f"\n模型参数统计:")
        print(f"  总参数: {total_params:.2f}M")
        print(f"  可训练参数: {trainable_params:.2f}M ({trainable_ratio:.2f}%)")
        
        # ============================================
        # 🔧 改动 1: 为门控权重设置独立学习率
        # ============================================
        # 原始代码:
        # trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        # self.optimizer = torch.optim.Adam(trainable_parameters, lr=opt.learning_rate)
        
        # 改进代码: 分离门控权重和其他参数
        gate_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'layer_importance_weights' in name:
                    gate_params.append(param)
                    print(f"  🔑 门控参数: {name}")
                else:
                    other_params.append(param)
        
        print(f"\n参数分组:")
        print(f"  门控参数数量: {len(gate_params)}")
        print(f"  其他参数数量: {len(other_params)}")
        
        # 创建参数组，为门控权重设置更大的学习率
        base_lr = opt.learning_rate
        gate_lr = base_lr * gate_lr_multiplier
        
        param_groups = [
            {
                'params': other_params, 
                'lr': base_lr,
                'name': 'other'
            },
            {
                'params': gate_params, 
                'lr': gate_lr,
                'name': 'gate'
            }
        ]
        
        self.optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))
        
        print(f"\n学习率配置:")
        print(f"  基础学习率: {base_lr:.2e}")
        print(f"  门控学习率: {gate_lr:.2e} ({gate_lr_multiplier}x)")
        
        # 设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\n使用设备: {self.device}")
        
        # 学习率调度器
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, 
            step_size=opt.lr_decay_step, 
            gamma=opt.lr_decay_factor
        )
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 记录最佳验证损失
        self.best_val_loss = float('inf')
        
        print("\n" + "="*70)
        print("✓ 改进的训练器初始化完成!")
        print("="*70 + "\n")
    
    def load_checkpoint(self, checkpoint_path):
        """
        从检查点恢复训练
        
        Args:
            checkpoint_path: 检查点文件路径
        
        Returns:
            start_epoch: 开始的epoch（用于继续训练）
        """
        print(f"\n{'='*70}")
        print(f"🔄 正在从检查点恢复训练...")
        print(f"{'='*70}")
        print(f"检查点路径: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 恢复模型状态
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"✓ 模型状态已恢复")
        
        # 恢复优化器状态
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"✓ 优化器状态已恢复")
        elif 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✓ 优化器状态已恢复")
        
        # 获取起始 epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        # 恢复学习率调度器（重新设置到正确的epoch）
        for _ in range(checkpoint.get('epoch', 0)):
            self.scheduler.step()
        print(f"✓ 学习率调度器已恢复")
        
        # 恢复最佳验证损失
        if 'loss' in checkpoint:
            self.best_val_loss = checkpoint['loss']
            print(f"✓ 最佳验证损失已恢复: {self.best_val_loss:.4f}")
        
        # 打印恢复的信息
        print(f"\n恢复训练信息:")
        print(f"  起始 Epoch: {start_epoch}")
        print(f"  上次验证损失: {checkpoint.get('loss', 'N/A')}")
        print(f"  上次准确率: {checkpoint.get('acc', 'N/A'):.4f}" if 'acc' in checkpoint else "  上次准确率: N/A")
        print(f"  上次平均精度: {checkpoint.get('ap', 'N/A'):.4f}" if 'ap' in checkpoint else "  上次平均精度: N/A")
        print(f"  当前基础学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  当前门控学习率: {self.optimizer.param_groups[1]['lr']:.2e}")
        
        # 显示门控权重
        if 'gate_weights' in checkpoint:
            gate_weights = checkpoint['gate_weights']
            if isinstance(gate_weights, torch.Tensor):
                gate_weights = gate_weights.cpu().numpy()
            print(f"\n  上次门控权重分布:")
            for i, w in enumerate(gate_weights):
                print(f"    Layer {i*2:2d}: {w:.4f}", end="  ")
                if (i+1) % 4 == 0:
                    print()
            if len(gate_weights) % 4 != 0:
                print()
            print(f"    标准差: {gate_weights.std():.6f}")
        
        print("="*70 + "\n")
        
        return start_epoch
    
    def train_epoch(self, data_loader, criterion, epoch):
        """
        训练一个epoch
        
        ============================================
        🔧 改动 2: 添加多样性正则化损失
        ============================================
        """
        self.model.train()
        self.model.to(self.device)
        
        # 确保骨干网络保持冻结
        self.model.freeze_backbone()
        
        running_loss = 0.0
        running_clf_loss = 0.0
        running_div_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
        
        for i, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            with autocast():
                output = self.model(data).squeeze()
                
                # ============================================
                # 分类损失
                # ============================================
                classification_loss = criterion(output, target)
                
                # ============================================
                # 🔧 改动 2: 添加多样性正则化损失
                # ============================================
                # 原始代码:
                # loss = classification_loss
                
                # 改进代码: 添加多样性损失
                if self.diversity_weight > 0:
                    if self.use_improved_gffm:
                        diversity_loss = self.model.gffm.compute_diversity_loss(
                            loss_type=self.diversity_loss_type
                        )
                    else:
                        # 为原始GFFM计算多样性损失
                        gate_weights = torch.softmax(self.model.gffm.layer_importance_weights, dim=0)
                        variance = torch.var(gate_weights)
                        diversity_loss = -variance  # 最大化方差
                    
                    # 总损失 = 分类损失 + λ * 多样性损失
                    total_loss = classification_loss + self.diversity_weight * diversity_loss
                else:
                    total_loss = classification_loss
                    diversity_loss = torch.tensor(0.0)
            
            # 反向传播
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 记录损失
            running_loss += total_loss.item()
            running_clf_loss += classification_loss.item()
            running_div_loss += diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else 0
            
            # 记录预测结果
            with torch.no_grad():
                predictions = torch.sigmoid(output).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(target.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'clf': f'{classification_loss.item():.4f}',
                'div': f'{diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else 0:.4f}'
            })
        
        # 计算epoch统计
        epoch_loss = running_loss / len(data_loader)
        epoch_clf_loss = running_clf_loss / len(data_loader)
        epoch_div_loss = running_div_loss / len(data_loader)
        
        # 计算准确率
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        predictions_binary = (all_predictions > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, predictions_binary)
        
        # 计算AP
        try:
            ap = average_precision_score(all_labels, all_predictions)
        except:
            ap = 0.0
        
        # ============================================
        # 🔧 改动 3: 打印门控权重统计
        # ============================================
        with torch.no_grad():
            gate_weights = torch.softmax(self.model.gffm.layer_importance_weights, dim=0).cpu().numpy()
            gate_std = gate_weights.std()
            gate_max = gate_weights.max()
            gate_min = gate_weights.min()
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} 训练完成")
        print(f"{'='*70}")
        print(f"总损失: {epoch_loss:.4f} = 分类损失 {epoch_clf_loss:.4f} + "
              f"多样性损失 {epoch_div_loss:.4f}")
        print(f"准确率: {accuracy:.4f} | AP: {ap:.4f}")
        print(f"门控权重: std={gate_std:.6f}, max={gate_max:.4f}, min={gate_min:.4f}")
        print(f"{'='*70}\n")
        
        return epoch_loss, accuracy, ap, gate_std
    
    def validate_epoch(self, data_loader, criterion, epoch, writer=None):
        """验证一个epoch"""
        self.model.eval()
        self.model.to(self.device)
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc=f"Validation Epoch {epoch}"):
                data, target = data.to(self.device), target.to(self.device).float()
                
                with autocast():
                    output = self.model(data).squeeze()
                    loss = criterion(output, target)
                
                running_loss += loss.item()
                
                predictions = torch.sigmoid(output).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(target.cpu().numpy())
        
        # 计算统计
        epoch_loss = running_loss / len(data_loader)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        predictions_binary = (all_predictions > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, predictions_binary)
        
        try:
            ap = average_precision_score(all_labels, all_predictions)
        except:
            ap = 0.0
        
        # 获取门控权重
        gate_weights = self.model.gffm.get_gate_weights().cpu().numpy()
        
        # 打印验证结果
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} 验证完成")
        print(f"{'='*70}")
        print(f"验证损失: {epoch_loss:.4f}")
        print(f"准确率: {accuracy:.4f} | AP: {ap:.4f}")
        print(f"\n当前门控权重:")
        for i, w in enumerate(gate_weights):
            print(f"  Layer {i*2:2d}: {w:.4f}", end="  ")
            if (i+1) % 4 == 0:
                print()
        print(f"\n{'='*70}\n")
        
        # 记录到TensorBoard
        if writer:
            writer.add_scalar('val/loss', epoch_loss, epoch)
            writer.add_scalar('val/accuracy', accuracy, epoch)
            writer.add_scalar('val/ap', ap, epoch)
            
            # 记录门控权重
            for i, w in enumerate(gate_weights):
                writer.add_scalar(f'gate_weights/layer_{i}', w, epoch)
            
            writer.add_scalar('gate_weights/std', gate_weights.std(), epoch)
            writer.add_scalar('gate_weights/max', gate_weights.max(), epoch)
            writer.add_scalar('gate_weights/min', gate_weights.min(), epoch)
        
        return epoch_loss, accuracy, ap
    
    def train(self, train_loader, val_loader, criterion, epochs, save_path, writer, start_epoch=1):
        """
        完整的训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            epochs: 总训练轮数
            save_path: 模型保存路径
            writer: TensorBoard writer
            start_epoch: 起始轮数（用于断点续训，默认为1）
        """
        # 🔧 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n{'='*70}")
        if start_epoch > 1:
            print(f"🔄 继续训练 (从 Epoch {start_epoch} 到 {epochs})")
        else:
            print(f"开始训练 (共 {epochs} 个epochs)")
        print(f"{'='*70}\n")
        
        for epoch in range(start_epoch, epochs + 1):
            # 训练
            train_loss, train_acc, train_ap, gate_std = self.train_epoch(
                train_loader, criterion, epoch
            )
            
            # 记录到TensorBoard
            if writer:
                writer.add_scalar('train/loss', train_loss, epoch)
                writer.add_scalar('train/accuracy', train_acc, epoch)
                writer.add_scalar('train/ap', train_ap, epoch)
                writer.add_scalar('train/gate_std', gate_std, epoch)
                writer.add_scalar('train/lr_base', self.optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('train/lr_gate', self.optimizer.param_groups[1]['lr'], epoch)
            
            # 验证
            val_loss, val_acc, val_ap = self.validate_epoch(
                val_loader, criterion, epoch, writer
            )
            
            # 学习率衰减
            self.scheduler.step()
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'acc': val_acc,
                    'ap': val_ap,
                    'gate_weights': self.model.gffm.get_gate_weights()
                }
                save_file = os.path.join(save_path, 'model_best_val_loss.pth')
                torch.save(checkpoint, save_file)
                print(f"✓ 保存最佳模型 (epoch {epoch}, val_loss={val_loss:.4f})")
            
            # 定期保存
            if epoch % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'acc': val_acc,
                    'ap': val_ap,
                    'gate_weights': self.model.gffm.get_gate_weights()
                }
                save_file = os.path.join(save_path, f'model_epoch_{epoch}.pth')
                torch.save(checkpoint, save_file)
        
        print(f"\n{'='*70}")
        print("训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
