"""
Trainer for Hierarchical Wavelet-Gated-Former
=============================================
训练器模块，用于训练Wavelet-Gated-Former模型

与ForgeLens的区别:
- 单阶段训练 (不需要两阶段训练)
- 使用新的网络架构
- 支持门控权重的监控和可视化
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


class Trainer_WaveletGatedFormer:
    """
    Hierarchical Wavelet-Gated-Former 训练器
    
    特点:
    - 单阶段端到端训练
    - 自动混合精度训练 (AMP)
    - 学习率衰减
    - 模型检查点保存
    - TensorBoard日志记录
    - 门控权重监控
    """
    
    def __init__(self, opt):
        """
        初始化训练器
        
        Args:
            opt: 配置选项对象，包含所有超参数
        """
        print("="*60)
        print("初始化 Wavelet-Gated-Former 训练器...")
        print("="*60)
        
        # 创建模型
        self.model = WaveletGatedFormer(
            feature_dim=1024,  # CLIP ViT-L/14 的特征维度
            num_wavelet_heads=opt.num_wavelet_heads,
            transformer_layers=opt.gffm_transformer_layers,
            transformer_heads=opt.gffm_transformer_heads,
            reduction_factor=opt.gffm_reduction_factor,
            dropout_prob=0.5,
            output_dim=1  # 二分类使用1个输出 + BCEWithLogitsLoss
        )
        
        # 打印模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"总模型参数: {total_params:.2f}M")
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        trainable_ratio = (trainable_params / total_params) * 100
        print(f"可训练参数: {trainable_params:.2f}M ({trainable_ratio:.2f}%)")
        
        # 优化器 - 只优化可训练的参数
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            trainable_parameters, 
            lr=opt.learning_rate, 
            betas=(0.9, 0.999)
        )
        
        # 设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
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
        
        print("训练器初始化完成!")
        print("="*60 + "\n")
    
    def load_checkpoint(self, checkpoint_path):
        """
        从检查点恢复训练
        
        Args:
            checkpoint_path: 检查点文件路径
        
        Returns:
            start_epoch: 开始的epoch（用于继续训练）
        """
        print(f"正在从检查点恢复训练...")
        print(f"检查点路径: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path)
        
        # 恢复模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型状态已恢复")
        
        # 恢复优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ 优化器状态已恢复")
        
        # 恢复学习率调度器（重新设置到正确的epoch）
        start_epoch = checkpoint['epoch'] + 1
        for _ in range(start_epoch):
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
        print(f"  上次准确率: {checkpoint.get('acc', 'N/A')}")
        print(f"  上次平均精度: {checkpoint.get('ap', 'N/A')}")
        print(f"  当前学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        if 'gate_weights' in checkpoint:
            gate_weights = checkpoint['gate_weights']
            print(f"\n  上次门控权重分布:")
            for i in range(0, len(gate_weights), 3):
                weights_str = "    "
                for j in range(i, min(i+3, len(gate_weights))):
                    weights_str += f"Layer {j:2d}: {gate_weights[j]:.4f}  "
                print(weights_str)
        
        print("="*60 + "\n")
        
        return start_epoch
    
    def train_epoch(self, dataloader: DataLoader, criterion):
        """
        训练一个epoch
        
        Args:
            dataloader: 训练数据加载器
            criterion: 损失函数
        
        Returns:
            平均训练损失
        """
        total_loss = 0.0
        total_batches = 0
        
        # 设置为训练模式
        self.model.to(self.device)
        self.model.train()
        
        # 确保backbone保持冻结
        self.model.freeze_backbone()
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="训练")):
            # 数据移到GPU
            data, target = data.to(self.device), target.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                output = self.model(data)
                loss = criterion(output.squeeze(1), target.type(torch.float32))
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 累积损失
            total_loss += loss.item()
            total_batches += 1
        
        return total_loss / (total_batches + 1e-8)
    
    def validate_epoch(self, dataloader: DataLoader, criterion, epoch: int, writer: SummaryWriter = None):
        """
        验证一个epoch
        
        Args:
            dataloader: 验证数据加载器
            criterion: 损失函数
            epoch: 当前epoch数
            writer: TensorBoard writer
        
        Returns:
            (验证损失, 准确率, 平均精度)
        """
        # 设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        running_loss = 0.0
        dataset_preds = []
        dataset_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="验证"):
                data, target = data.to(self.device), target.to(self.device)
                
                # 混合精度前向传播
                with autocast():
                    output = self.model(data)
                    loss = criterion(output.squeeze(1), target.type(torch.float32))
                    
                    running_loss += loss.item()
                    
                    # 收集预测和标签
                    pred_prob = output.cpu().numpy()
                    target_np = target.cpu().numpy()
                    
                    dataset_preds.append(pred_prob)
                    dataset_targets.append(target_np)
        
        # 合并所有批次的预测
        dataset_preds = np.concatenate(dataset_preds)
        dataset_targets = np.concatenate(dataset_targets)
        
        # 计算指标
        acc = accuracy_score(dataset_targets, dataset_preds > 0)
        ap = average_precision_score(dataset_targets, dataset_preds)
        
        # 记录到TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/Validation', running_loss / len(dataloader), epoch)
            writer.add_scalar('Accuracy', acc, epoch)
            writer.add_scalar('Average Precision', ap, epoch)
            
            # 记录门控权重
            gate_weights = self.model.get_gate_weights()
            for i, weight in enumerate(gate_weights):
                writer.add_scalar(f'GateWeights/Layer_{i}', weight.item(), epoch)
        
        return running_loss / len(dataloader), acc, ap
    
    def train(
        self, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        criterion, 
        num_epochs: int,
        checkpoint_dir: str = None, 
        writer: SummaryWriter = None,
        start_epoch: int = 0
    ):
        """
        完整训练流程
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            criterion: 损失函数
            num_epochs: 训练轮数
            checkpoint_dir: 检查点保存目录
            writer: TensorBoard writer
            start_epoch: 起始epoch（用于断点续训，默认为0）
        """
        print("\n" + "="*60)
        if start_epoch > 0:
            print(f"继续训练 Wavelet-Gated-Former (从 Epoch {start_epoch+1} 开始)")
        else:
            print("开始训练 Wavelet-Gated-Former")
        print("="*60 + "\n")
        
        best_val_loss = self.best_val_loss  # 使用类属性（可能已从checkpoint恢复）
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # 训练阶段
            print("\n[训练阶段]")
            train_loss = self.train_epoch(train_dataloader, criterion)
            
            # 验证阶段
            print("\n[验证阶段]")
            val_loss, acc, ap = self.validate_epoch(
                val_dataloader, 
                criterion, 
                epoch, 
                writer=writer
            )
            
            # 打印统计信息
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1} 结果:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  准确率: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  平均精度: {ap:.4f}")
            
            # 打印门控权重
            gate_weights = self.model.get_gate_weights()
            print(f"\n  门控权重分布:")
            for i in range(0, len(gate_weights), 3):  # 每行打印3个
                weights_str = "    "
                for j in range(i, min(i+3, len(gate_weights))):
                    weights_str += f"Layer {j:2d}: {gate_weights[j].item():.4f}  "
                print(weights_str)
            
            print(f"{'='*60}\n")
            
            # Tensor
            if writer is not None:
                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 保存检查点
            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # 每个epoch保存一次
                if (epoch + 1) % 1 == 0:
                    checkpoint_path_1 = os.path.join(
                        checkpoint_dir, 
                        f'model_epoch_{epoch+1}.pth'
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                        'acc': acc,
                        'ap': ap,
                        'gate_weights': gate_weights.cpu().numpy()
                    }, checkpoint_path_1)
                    print(f"✓ 检查点已保存: {checkpoint_path_1}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path_2 = os.path.join(
                        checkpoint_dir, 
                        'model_best_val_loss.pth'
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                        'acc': acc,
                        'ap': ap,
                        'gate_weights': gate_weights.cpu().numpy()
                    }, checkpoint_path_2)
                    print(f"✓ 最佳模型已保存: {checkpoint_path_2} (val_loss: {val_loss:.4f})")
            
            # 学习率衰减
            self.scheduler.step()
        
        print("\n" + "="*60)
        print("训练完成!")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print("="*60 + "\n")


# 测试代码
if __name__ == "__main__":
    from options.options import Options
    
    print("测试 Trainer_WaveletGatedFormer...")
    
    # 创建配置选项
    options = Options()
    opt = options.parse()
    
    # 添加新模型的配置
    opt.num_wavelet_heads = 12
    opt.gffm_transformer_layers = 2
    opt.gffm_transformer_heads = 2
    opt.gffm_reduction_factor = 1
    opt.learning_rate = 2e-6
    opt.lr_decay_step = 2
    opt.lr_decay_factor = 0.7
    
    # 创建训练器
    trainer = Trainer_WaveletGatedFormer(opt)
    
    print("\n训练器创建成功!")
