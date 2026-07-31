import torch
import torch.nn as nn
import os
import sys
import json

# 将父目录（项目根目录）加入系统路径，以便导入同项目的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_process.dataloader import create_dataloaders  # 自定义数据加载函数
from torchvision import models                           # 用于加载模型架构


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """
    训练模型一个 epoch。

    参数:
        model:      PyTorch 模型
        loader:     训练 DataLoader
        criterion:  损失函数
        optimizer:  优化器
        device:     计算设备 ('cuda' / 'cpu')
        scaler:     混合精度梯度缩放器 (GradScaler)，若为 None 则使用常规精度训练

    返回:
        (平均损失, 训练准确率百分比)
    """
    model.train()  # 设置为训练模式（启用 Dropout、BatchNorm 统计量更新等）
    running_loss = 0.0
    correct = 0
    total = 0
    
    for img, lbl in loader:
        # 将数据和标签移动到指定设备（如 GPU）
        img, lbl = img.to(device), lbl.to(device)
        
        # 清空上一轮的梯度
        optimizer.zero_grad()
        
        # 前向传播
        out = model(img)
        loss = criterion(out, lbl)
        
        # 反向传播：支持混合精度训练
        if scaler is not None:
            # 使用 GradScaler 对损失进行缩放，防止梯度下溢
            scaler.scale(loss).backward()
            # scaler.step() 会自动在必要时还原梯度，然后调用 optimizer.step()
            scaler.step(optimizer)
            scaler.update()  # 更新缩放因子
        else:
            loss.backward()
            optimizer.step()
        
        # 累加损失（乘以当前 batch 的样本数，用于最后计算整个 epoch 的平均损失）
        running_loss += loss.item() * img.size(0)
        
        # 统计预测正确的样本数
        _, pred = torch.max(out, 1)   # 取概率最大的类别索引
        total += lbl.size(0)
        correct += (pred == lbl).sum().item()
    
    # 返回该 epoch 的平均损失和准确率（百分比）
    return running_loss / total, 100 * correct / total


def validate(model, loader, criterion, device):
    """
    在验证集上评估模型性能，不更新参数。

    参数:
        model:     PyTorch 模型
        loader:    验证 DataLoader
        criterion: 损失函数
        device:    计算设备

    返回:
        (平均损失, 验证准确率百分比)
    """
    model.eval()  # 设置为评估模式（关闭 Dropout、固定 BatchNorm 统计量等）
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度计算，节省显存，加速推理
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            loss = criterion(out, lbl)
            
            running_loss += loss.item() * img.size(0)
            _, pred = torch.max(out, 1)
            total += lbl.size(0)
            correct += (pred == lbl).sum().item()
    
    return running_loss / total, 100 * correct / total


def run_baseline(data_root, train_folder='train', num_classes=10, epochs=10,
                 device='cuda', patience=5, model_name='resnet18',
                 num_workers=4, model_save_dir=None,
                 result_save_dir=None, dataset_name=None):
    """
    从头训练一个基线模型（无预训练权重），不使用任何迁移学习策略。
    训练过程中会保存最佳模型并记录训练历史到 JSON 文件。

    参数:
        data_root:       数据集根目录，其中应包含 'train' 或自定义 train_folder 及 'val' 文件夹
        train_folder:    训练集所在的子文件夹名称（如 'train', 'train_5shot'）
        num_classes:     分类类别数
        epochs:          最大训练轮数
        device:          计算设备
        patience:        早停耐心值（连续多少个 epoch 验证准确率不提升即停止）
        model_name:      模型名称，目前仅支持 'resnet18'（基线固定使用 ResNet18 结构）
        num_workers:     数据加载使用的子进程数
        model_save_dir:  模型权重保存目录，None 时自动设为默认路径
        result_save_dir: 结果 JSON 保存目录，None 时自动构建（含 dataset_name）
        dataset_name:    数据集名称，用于构建默认结果目录（如 'rice'）

    返回:
        (best_acc, history) — 最佳验证准确率和训练历史字典
    """
    print(f"从头训练 Baseline | 数据: {data_root}/{train_folder}")
    
    # ---- 1. 创建数据加载器 ----
    train_loader, val_loader, _ = create_dataloaders(
        data_root=data_root,
        train_folder=train_folder,
        num_workers=num_workers
    )

    # ---- 2. 构建模型（resnet18，不使用预训练权重）----
    model = models.resnet18(weights=None)  # weights=None 表示随机初始化，不使用 ImageNet 预训练
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # 替换最后的全连接层，适配自定义类别数
    model = model.to(device)

    # ---- 3. 设置混合精度、损失函数和优化器 ----
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 所有参数都参与训练

    # ---- 4. 训练状态记录 ----
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # ---- 5. 训练循环 ----
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 记录每一轮的历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:2d}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # 验证准确率提升时保存最佳模型权重
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # 确定模型保存目录
            if model_save_dir is None:
                model_save_dir = os.path.join(r'D:\python_work\cs', 'models', 'saved')
            os.makedirs(model_save_dir, exist_ok=True)
            # 权重文件名包含 train_folder，避免不同实验互相覆盖
            weight_name = f'best_baseline_{model_name}_{train_folder}.pth'
            torch.save(model.state_dict(), os.path.join(model_save_dir, weight_name))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏸️ 早停触发！在第 {epoch+1} 个 epoch 停止训练。")
                break

    # ---- 6. 训练结束后，加载最优模型权重（回滚）----
    if model_save_dir is None:
        model_save_dir = os.path.join(r'D:\python_work\cs', 'models', 'saved')
    best_weight_path = os.path.join(model_save_dir, f'best_baseline_{model_name}_{train_folder}.pth')
    if os.path.exists(best_weight_path):
        model.load_state_dict(torch.load(best_weight_path, map_location=device))
        print(f"🔄 已回滚到最优模型权重 (准确率 {best_acc:.2f}%)")

    print(f"✅ 从头训练最佳验证准确率: {best_acc:.2f}%")

    # ---- 7. 自动保存实验结果到 JSON 文件 ----
    if result_save_dir is None:
        # 默认路径：如果有 dataset_name，则保存到 results/<dataset_name>/
        if dataset_name:
            result_save_dir = os.path.join(r'D:\python_work\cs', 'results', dataset_name)
        else:
            result_save_dir = os.path.join(r'D:\python_work\cs', 'results')
    os.makedirs(result_save_dir, exist_ok=True)
    
    # JSON 文件名包含模型名、策略、训练子集
    json_path = os.path.join(result_save_dir, f'result_{model_name}_baseline_{train_folder}.json')
    result_data = {
        "model_name": model_name,
        "strategy": "baseline",          # 标记为从头训练
        "best_val_acc": best_acc,
        "history": history,
        "train_folder": train_folder     # 记录使用的训练子集名称
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2)
    print(f"📁 结果已保存至 {json_path}")

    return best_acc, history


# ==================== 测试入口 ====================
if __name__ == "__main__":
    # 简单测试：使用 GPU（如果可用）运行一次水稻数据集的全量从头训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_baseline(
        data_root=r'D:\python_work\cs\raw',     # 原始水稻数据集路径
        train_folder='train',                   # 使用完整训练集
        num_classes=10,
        epochs=10,
        device=device,
        model_name='resnet18',
        dataset_name='rice'                     # 数据集名称，用于结果保存目录
    )