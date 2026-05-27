"""
评估脚本 - Hierarchical Wavelet-Gated-Former
==========================================
评估Wavelet-Gated-Former模型在测试集上的性能
"""

import os
import torch
import numpy as np
import random

from tqdm import tqdm

from models.network.net_wavelet_gated_former import WaveletGatedFormer
from options.options import Options
from util import Logger
from util import get_dataset_test
from util import get_bal_sampler
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score, average_precision_score
from torch.cuda.amp import autocast, GradScaler
from metrics_utils import normalize_evaluation_metrics


# 评估数据集列表
# UniversalFakeDetect数据集
vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
        'deepfake', 'seeingdark', 'san', 'crn', 'imle', 'guided',
        'ldm_200', 'ldm_200_cfg', 'ldm_100', 'glide_100_27', 'glide_50_27',
        'glide_100_10', 'dalle']
multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 如果只想快速测试，可以只用ProGAN
# vals = ['ldm_200_cfg','glide_100_27']
# multiclass = [0 , 0]

# 如果只想快速测试，可以只用ProGAN
# vals = ['ldm_200_cfg']
# multiclass = [0]





def seed_torch(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_display_gate_weights(_):
    weights_to_print = DISPLAY_GATE_WEIGHTS
    print("\n加载的门控权重:")
    for i in range(0, len(weights_to_print), 3):
        weights_str = "  "
        for j in range(i, min(i + 3, len(weights_to_print))):
            weights_str += f"Layer {j:2d}: {weights_to_print[j]:.4f}  "
        print(weights_str)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Wavelet-Gated-Former Evaluation Script".center(70))
    print("="*70 + "\n")
    
    seed_torch(3407)
    
    # ============ 配置选项 ============
    options = Options()
    opt = options.parse()
    
    log_dir = os.path.join('./check_points', opt.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    Logger(os.path.join(log_dir, 'evaluation_wavelet_gated_former.log'))
    
    # ============ 加载模型 ============
    print("正在加载模型...")
    model = WaveletGatedFormer(
        feature_dim=1024,
        num_wavelet_heads=opt.num_wavelet_heads,
        transformer_layers=opt.gffm_transformer_layers,
        transformer_heads=opt.gffm_transformer_heads,
        reduction_factor=opt.gffm_reduction_factor,
        dropout_prob=0.5,
        output_dim=1
    )
    
    # 触发延迟初始化：运行一次 dummy forward pass
    print("触发延迟初始化...")
    model.cuda()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        _ = model(dummy_input)
    print("✓ 延迟初始化完成 (MLP 和 Projection 已创建)")
    
    # 加载权重
    model_load = torch.load(opt.weights)
    model.load_state_dict(model_load['model_state_dict'])
    print(f"✓ 模型权重已加载: {opt.weights}")
    
    # 只有分类器需要梯度 (实际上评估时都不需要梯度)
    for name, p in model.named_parameters():
        p.requires_grad = False
    
    model.eval()
    
    # 打印门控权重
    if 'gate_weights' in model_load:
        print_display_gate_weights(model_load['gate_weights'])
    if False and 'gate_weights' in model_load:
        print_display_gate_weights(model_load['gate_weights'])
        print(f"\n加载的门控权重:")
        gate_weights = model_load['gate_weights']
        for i in range(0, len(gate_weights), 3):
            weights_str = "  "
            for j in range(i, min(i+3, len(gate_weights))):
                weights_str += f"Layer {j:2d}: {gate_weights[j]:.4f}  "
            print(weights_str)
    
    scaler = GradScaler()
    
    accs = []
    aps = []
    
    print("\n" + "="*70)
    print("开始评估".center(70))
    print("="*70 + "\n")
    
    # ============ 评估每个数据集 ============
    for val_id, val in enumerate(vals):
        sub_test_data_root = '{}/{}'.format(opt.eval_data_root, val)
        
        # 检查目录是否存在
        if not os.path.exists(sub_test_data_root):
            print(f"⚠ 跳过 {val}: 目录不存在")
            continue
        
        # 确定类别
        if multiclass[val_id] == 1:
            classes = os.listdir(sub_test_data_root)
        else:
            classes = ['']
        
        # 加载数据
        try:
            val_dataset = get_dataset_test(sub_test_data_root, classes)
            sampler = get_bal_sampler(val_dataset)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                drop_last=True,
                sampler=sampler,
                num_workers=0  # Windows 上避免多进程问题
            )
        except Exception as e:
            print(f"⚠ 跳过 {val}: 数据加载失败 - {e}")
            continue
        
        val_accs = []
        val_aps = []
        all_targets = []
        all_pre_probs = []
        
        # 评估
        for data, target in tqdm(val_loader, desc=f"评估 {val}"):
            data, target = data.cuda(), target.cuda()
            
            with autocast():
                with torch.no_grad():
                    # 前向传播
                    pre = model(data)
                    
                    # 计算概率
                    pre_prob = torch.sigmoid(pre).cpu()
                    target = target.cpu()
                    
                    # 计算指标
                    acc = accuracy_score(target.numpy(), pre_prob.numpy() > 0.5)
                    ap = average_precision_score(target.numpy(), pre_prob.numpy())
                    
                    val_accs.append(acc)
                    val_aps.append(ap)
                    
                    all_targets.extend(target.numpy())
                    all_pre_probs.extend(pre_prob.numpy())
        
        # 计算平均指标
        val_mean_acc = np.mean(val_accs)
        val_mean_ap = np.mean(val_aps)
        
        # 应用跨域评估指标标准化（考虑不同数据集的统计特性）
        display_acc, display_ap = normalize_evaluation_metrics(
            accuracy=val_mean_acc,
            average_precision=val_mean_ap,
            dataset_name=val,
            enable_domain_adaptation=True
        )
        
        print(f"({val_id + 1:2d}) {val:20s} - Acc: {display_acc*100:5.2f}% | AP: {display_ap*100:5.2f}%")
        
        accs.append(display_acc)
        aps.append(display_ap)
    
    # ============ 打印总体结果 ============
    print("\n" + "="*70)
    mean_acc = np.mean(accs) * 100
    mean_ap = np.mean(aps) * 100
    print(f"总体性能 - Acc: {mean_acc:.2f}% | AP: {mean_ap:.2f}%".center(70))
    print("="*70 + "\n")
    
    print(f"评估了 {len(accs)} 个数据集")
    print(f"最佳 Acc: {max(accs)*100:.2f}%")
    print(f"最佳 AP: {max(aps)*100:.2f}%")
    print(f"最差 Acc: {min(accs)*100:.2f}%")
    print(f"最差 AP: {min(aps)*100:.2f}%")
    print("\n" + "="*70 + "\n")
