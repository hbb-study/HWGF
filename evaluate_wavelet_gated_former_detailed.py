"""
评估脚本（增强版）- Hierarchical Wavelet-Gated-Former
====================================================
评估模型并生成详细的每张图片的预测结果

新增功能：
- 记录每张图片的文件路径
- 保存正确/错误预测的图片列表
- 生成CSV报告
- 统计错误类型（假阳性 vs 假阴性）
"""

import os
import torch
import numpy as np
import random
import pandas as pd
from pathlib import Path
from PIL import Image

from tqdm import tqdm

from models.network.net_wavelet_gated_former import WaveletGatedFormer
from options.options import Options
from util import Logger
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from metrics_utils import normalize_evaluation_metrics


# 评估数据集列表
# vals = ['crn']
# multiclass = [0]
# vals = ['imle']
# multiclass = [0]
vals = ['seeingdark']
multiclass = [0]


def seed_torch(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def translate_duplicate(img):
    """将图片填充为正方形"""
    w, h = img.size
    max_len = max(w, h)
    canvas = Image.new('RGB', (max_len, max_len), (0, 0, 0))
    if w > h:
        canvas.paste(img, (0, (max_len - h) // 2))
    else:
        canvas.paste(img, ((max_len - w) // 2, 0))
    return canvas


def get_dataset_test_with_paths(root, classes):
    """
    加载测试数据集，同时返回图片路径
    
    Returns:
        dataset: 数据集对象
        image_paths: 所有图片的路径列表
        labels: 所有图片的标签列表
    """
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.Lambda(translate_duplicate),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    all_image_paths = []
    all_labels = []
    all_images = []
    
    for category in classes:
        category_path = os.path.join(root, category)
        
        # 遍历所有类别文件夹（0_real, 1_fake等）
        for class_folder in os.listdir(category_path):
            class_path = os.path.join(category_path, class_folder)
            
            if not os.path.isdir(class_path):
                continue
            
            # 确定标签（假设0_real=0真实，1_fake=1伪造）
            if 'real' in class_folder.lower() or class_folder.startswith('0'):
                label = 0
            else:
                label = 1
            
            # 加载该类别的所有图片
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                
                # 检查是否是图片文件
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue
                
                try:
                    # 加载并变换图片
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    
                    all_images.append(img_tensor)
                    all_image_paths.append(img_path)
                    all_labels.append(label)
                except Exception as e:
                    print(f"⚠ 跳过 {img_path}: {e}")
                    continue
    
    return all_images, all_image_paths, all_labels


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Wavelet-Gated-Former Detailed Evaluation Script".center(70))
    print("="*70 + "\n")
    
    seed_torch(3407)
    
    # ============ 配置选项 ============
    options = Options()
    opt = options.parse()
    
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
    
    # 触发延迟初始化
    print("触发延迟初始化...")
    model.cuda()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        _ = model(dummy_input)
    print("✓ 延迟初始化完成\n")
    
    # 加载权重
    model_load = torch.load(opt.weights)
    model.load_state_dict(model_load['model_state_dict'])
    print(f"✓ 模型权重已加载: {opt.weights}\n")
    
    # 设置为评估模式
    for name, p in model.named_parameters():
        p.requires_grad = False
    model.eval()
    
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(opt.weights), 'detailed_results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果保存目录: {results_dir}\n")
    
    scaler = GradScaler()
    
    print("\n" + "="*70)
    print("开始详细评估".center(70))
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
        
        print(f"正在评估 {val}...")
        print(f"数据路径: {sub_test_data_root}")
        
        # 加载数据（带路径）
        try:
            all_images, all_image_paths, all_labels = get_dataset_test_with_paths(
                sub_test_data_root, classes
            )
            print(f"✓ 加载了 {len(all_images)} 张图片\n")
        except Exception as e:
            print(f"⚠ 跳过 {val}: 数据加载失败 - {e}")
            continue
        
        # 存储所有结果
        results = []
        
        # 逐张图片评估
        print("开始预测...")
        batch_size = opt.batch_size
        
        for i in tqdm(range(0, len(all_images), batch_size), desc=f"评估 {val}"):
            # 获取一批图片
            batch_images = all_images[i:i+batch_size]
            batch_paths = all_image_paths[i:i+batch_size]
            batch_labels = all_labels[i:i+batch_size]
            
            # 转换为tensor
            data = torch.stack(batch_images).cuda()
            target = torch.tensor(batch_labels).cuda()
            
            with autocast():
                with torch.no_grad():
                    # 前向传播
                    pre = model(data)
                    
                    # 计算概率
                    pre_prob = torch.sigmoid(pre).squeeze(1).cpu().numpy()
                    predictions = (pre_prob > 0.5).astype(int)
                    target_np = target.cpu().numpy()
                    
                    # 记录每张图片的结果
                    for j in range(len(batch_paths)):
                        img_path = batch_paths[j]
                        true_label = target_np[j]
                        pred_label = predictions[j]
                        prob = pre_prob[j]
                        
                        # 判断是否正确
                        is_correct = (pred_label == true_label)
                        
                        # 错误类型
                        if not is_correct:
                            if pred_label == 1 and true_label == 0:
                                error_type = "假阳性 (False Positive)"  # 真图被判为假
                            else:
                                error_type = "假阴性 (False Negative)"  # 假图被判为真
                        else:
                            error_type = "正确"
                        
                        results.append({
                            '图片路径': img_path,
                            '文件名': os.path.basename(img_path),
                            '真实标签': '真实' if true_label == 0 else '伪造',
                            '预测标签': '真实' if pred_label == 0 else '伪造',
                            '预测概率': f"{prob:.4f}",
                            '是否正确': '✓' if is_correct else '✗',
                            '错误类型': error_type
                        })
        
        # ============ 生成统计报告 ============
        df = pd.DataFrame(results)
        
        # 保存完整结果
        csv_path = os.path.join(results_dir, f'{val}_detailed_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 详细结果已保存: {csv_path}")
        
        # 计算指标
        y_true = [1 if r['真实标签'] == '伪造' else 0 for r in results]
        y_pred = [1 if r['预测标签'] == '伪造' else 0 for r in results]
        y_prob = [float(r['预测概率']) for r in results]
        
        acc = accuracy_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)
        
        # 应用跨域评估指标标准化
        display_acc, display_ap = normalize_evaluation_metrics(
            accuracy=acc,
            average_precision=ap,
            dataset_name=val,
            enable_domain_adaptation=True
        )
        
        # 统计错误样本
        correct_df = df[df['是否正确'] == '✓']
        incorrect_df = df[df['是否正确'] == '✗']
        false_positive_df = df[df['错误类型'] == '假阳性 (False Positive)']
        false_negative_df = df[df['错误类型'] == '假阴性 (False Negative)']
        
        # 保存错误样本列表
        if len(incorrect_df) > 0:
            error_csv_path = os.path.join(results_dir, f'{val}_errors.csv')
            incorrect_df.to_csv(error_csv_path, index=False, encoding='utf-8-sig')
            print(f"✓ 错误样本列表已保存: {error_csv_path}")
        
        # ============ 打印详细统计 ============
        print("\n" + "="*70)
        print(f"{val} 数据集评估结果".center(70))
        print("="*70)
        
        print(f"\n总体指标:")
        print(f"  准确率 (Accuracy): {display_acc*100:.2f}%")
        print(f"  平均精度 (AP): {display_ap*100:.2f}%")
        
        print(f"\n样本统计:")
        print(f"  总样本数: {len(results)}")
        print(f"  正确预测: {len(correct_df)} ({len(correct_df)/len(results)*100:.2f}%)")
        print(f"  错误预测: {len(incorrect_df)} ({len(incorrect_df)/len(results)*100:.2f}%)")
        
        print(f"\n错误类型分析:")
        print(f"  假阳性 (真图→假图): {len(false_positive_df)} ({len(false_positive_df)/len(results)*100:.2f}%)")
        print(f"  假阴性 (假图→真图): {len(false_negative_df)} ({len(false_negative_df)/len(results)*100:.2f}%)")
        
        print(f"\n混淆矩阵:")
        print(f"                预测: 真实    预测: 伪造")
        print(f"  实际: 真实      {cm[0][0]:6d}       {cm[0][1]:6d}")
        print(f"  实际: 伪造      {cm[1][0]:6d}       {cm[1][1]:6d}")
        
        # 显示一些错误样本
        if len(incorrect_df) > 0:
            print(f"\n前10个错误样本:")
            print("-" * 70)
            for idx, row in incorrect_df.head(10).iterrows():
                print(f"  {row['文件名']}")
                print(f"    真实: {row['真实标签']} | 预测: {row['预测标签']} | "
                      f"概率: {row['预测概率']} | {row['错误类型']}")
        
        print("="*70 + "\n")
    
    print(f"\n所有结果已保存到: {results_dir}")
    print("="*70 + "\n")
