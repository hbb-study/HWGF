"""
训练脚本 - Hierarchical Wavelet-Gated-Former
==========================================
使用新的Wavelet-Gated-Former模型进行单阶段训练
"""

import random
from models.trainer_wavelet_gated_former import Trainer_WaveletGatedFormer
from options.options import Options
from util import *
import util
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from tensorboardX import SummaryWriter
import torch.nn as nn


def seed_torch(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Hierarchical Wavelet-Gated-Former Training Script".center(70))
    print("="*70 + "\n")
    
    # ============ 配置选项 ============
    options = Options()
    opt = options.parse()
    
    # 设置随机种子
    seed_torch(opt.seed)
    
    # ============ 创建日志目录 ============
    log_path = os.path.join('./check_points', opt.experiment_name)
    os.makedirs(log_path, exist_ok=True)
    
    # TensorBoard writer
    train_writer = SummaryWriter(os.path.join(log_path, 'train_wavelet_gated_former'))
    
    # 日志文件
    Logger(os.path.join(log_path, 'train_wavelet_gated_former', 'train.log'))
    
    # 打印配置
    print("\n" + "="*70)
    print("配置参数:".center(70))
    print("="*70)
    options.print_options()
    print("="*70 + "\n")
    
    # ============ 数据加载 ============
    print("正在加载训练数据...")
    train_dataset = get_dataset(opt.train_data_root, opt.train_classes)
    sampler = get_bal_sampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.wgf_batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        num_workers=opt.num_workers
    )
    print(f"✓ 训练数据已加载: {len(train_dataset)} 样本")
    
    print("正在加载验证数据...")
    val_dataset = get_dataset_test(opt.val_data_root, opt.val_classes)
    sampler = get_bal_sampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.wgf_batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        num_workers=opt.num_workers
    )
    print(f"✓ 验证数据已加载: {len(val_dataset)} 样本\n")
    
    # ============ 创建训练器 ============
    # 设置新模型的特定参数
    opt.learning_rate = opt.wgf_learning_rate
    opt.lr_decay_step = opt.wgf_lr_decay_step
    opt.lr_decay_factor = opt.wgf_lr_decay_factor
    
    trainer = Trainer_WaveletGatedFormer(opt)
    
    # ============ 检查是否需要断点续训 ============
    start_epoch = 0
    if hasattr(opt, 'resume') and opt.resume:
        # 如果指定了resume，从最新的检查点恢复
        checkpoint_dir = os.path.join(log_path, 'train_wavelet_gated_former', 'model')
        
        # 查找最新的检查点
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_')]
            if checkpoints:
                # 按epoch排序，取最新的
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"\n找到检查点: {latest_checkpoint}")
                
                # 触发延迟初始化
                print("触发延迟初始化...")
                trainer.model.cuda()
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224).cuda()
                    _ = trainer.model(dummy_input)
                print("✓ 延迟初始化完成\n")
                
                # 恢复训练
                start_epoch = trainer.load_checkpoint(latest_checkpoint)
            else:
                print("\n⚠ 未找到检查点，从头开始训练\n")
        else:
            print("\n⚠ 检查点目录不存在，从头开始训练\n")
    
    # ============ 开始训练 ============
    print("="*70)
    if start_epoch > 0:
        print(f"继续训练 (从 Epoch {start_epoch+1} 开始)".center(70))
    else:
        print("开始训练".center(70))
    print("="*70 + "\n")
    
    trainer.train(
        train_loader, 
        val_loader, 
        nn.BCEWithLogitsLoss(),  # 二分类损失函数
        opt.wgf_epochs,
        os.path.join(log_path, 'train_wavelet_gated_former', 'model'), 
        train_writer,
        start_epoch=start_epoch  # 传入起始epoch
    )
    
    print("\n" + "="*70)
    print("训练完成!".center(70))
    print("="*70)
    print(f"\n模型保存在: {os.path.join(log_path, 'train_wavelet_gated_former', 'model')}")
    print(f"日志保存在: {os.path.join(log_path, 'train_wavelet_gated_former')}")
    print("\n" + "="*70 + "\n")
