import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm  # 用于显示进度条
import matplotlib.pyplot as plt  # 本文件中未使用，但保留以备扩展

# 将上一级目录（项目根目录）添加到系统路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.dataloader import create_dataloaders  # 自定义数据加载函数
from models.model_utils import get_model                # 自定义模型加载函数


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """
    训练模型一个 epoch。

    参数:
        model:       PyTorch 模型
        loader:      训练 DataLoader
        criterion:   损失函数
        optimizer:   优化器
        device:      设备 ('cuda' 或 'cpu')
        scaler:      混合精度训练的梯度缩放器（GradScaler），若为 None 则使用常规精度训练
    
    返回:
        (平均损失, 训练准确率百分比)
    """
    model.train()  # 设为训练模式（启用 Dropout、BatchNorm 更新等）
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用 tqdm 创建进度条，leave=False 表示完成后不保留进度条
    pbar = tqdm(loader, desc="训练中", leave=False)
    for img, lbl in pbar:
        # 将数据和标签移动到指定设备
        img, lbl = img.to(device), lbl.to(device)
        
        # 清空上一轮的梯度
        optimizer.zero_grad()
        
        # 前向传播
        out = model(img)
        loss = criterion(out, lbl)
        
        # 反向传播：支持混合精度训练
        if scaler is not None:
            # 使用 GradScaler 缩放损失并反向传播
            scaler.scale(loss).backward()
            # 使用 scaler 更新优化器参数（自动处理梯度缩放还原）
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规反向传播
            loss.backward()
            optimizer.step()
        
        # 累计损失（乘以当前批次样本数，以便后续求平均）
        running_loss += loss.item() * img.size(0)
        
        # 计算预测准确的样本数
        _, pred = torch.max(out, 1)  # 取最大概率的类别
        total += lbl.size(0)
        correct += (pred == lbl).sum().item()
        
        # 更新进度条显示当前 batch 的 loss 和累计准确率
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")
    
    # 返回该 epoch 的平均损失和准确率
    return running_loss / total, 100 * correct / total


def validate(model, loader, criterion, device):
    """
    在验证集上评估模型性能。

    参数:
        model:      PyTorch 模型
        loader:     验证 DataLoader
        criterion:  损失函数
        device:     设备

    返回:
        (平均损失, 验证准确率百分比)
    """
    model.eval()  # 设为评估模式（关闭 Dropout、固定 BatchNorm 统计量等）
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="验证中", leave=False)
    with torch.no_grad():  # 禁用梯度计算，节省显存和计算时间
        for img, lbl in pbar:
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            loss = criterion(out, lbl)
            
            running_loss += loss.item() * img.size(0)
            _, pred = torch.max(out, 1)
            total += lbl.size(0)
            correct += (pred == lbl).sum().item()
            
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")
    
    return running_loss / total, 100 * correct / total


def run_experiment(data_root, train_folder, strategy, model_name='resnet18',
                   num_classes=10, batch_size=32, epochs=10, lr=0.001,
                   device='cuda', num_workers=6, patience=5,
                   result_save_dir=None, model_save_dir=None):
    """
    运行一次完整的迁移学习实验。
    包含训练循环、早停机制、保存最佳模型权重和训练历史记录。

    参数:
        data_root:      数据集根目录（包含 train 和 val 文件夹，或通过 train_folder 指定子文件夹）
        train_folder:   训练集所在的子文件夹名称（如 "train", "train_5shot"）
        strategy:       微调策略，传递给 get_model（如 "freeze_all"）
        model_name:     模型名称（'resnet18' 等）
        num_classes:    分类类别数
        batch_size:     批次大小
        epochs:         最大训练轮数
        lr:             学习率
        device:         计算设备
        num_workers:    数据加载子进程数
        patience:       早停耐心值：若连续 patience 个 epoch 验证准确率未提升则提前停止
        result_save_dir: 结果文件保存目录，None 时使用默认路径
        model_save_dir:  模型权重保存目录，None 时使用默认路径
    
    返回:
        (最佳验证准确率, 训练历史字典)
    """
    print(f"\n{'='*60}")
    print(f"策略: {strategy} | 数据: {data_root}/{train_folder}")
    print(f"{'='*60}")

    # ---- 1. 创建 DataLoader ----
    train_loader, val_loader, _ = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        train_folder=train_folder,
        num_workers=num_workers
    )
    
    # ---- 2. 构建模型并应用微调策略 ----
    model = get_model(num_classes=num_classes, strategy=strategy, model_name=model_name)
    model = model.to(device)
    
    # ---- 3. 设置混合精度训练（仅 CUDA 设备） ----
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    # ---- 4. 定义损失函数和优化器 ----
    criterion = nn.CrossEntropyLoss()
    # 只优化需要梯度的参数（根据策略冻结了部分参数）
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # ---- 5. 训练准备：最佳准确率记录、早停计数、历史记录 ----
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # ---- 6. 训练循环 ----
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练一个 epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"训练 Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | 验证 Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # ---- 早停与模型保存逻辑 ----
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            # 确定模型保存目录，如果未指定则使用默认路径
            if model_save_dir is None:
                model_save_dir = os.path.join(r'D:\python_work\cs', 'models', 'saved')
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 保存最佳模型权重
            save_name = f'best_{strategy}_{model_name}_{train_folder}.pth'
            torch.save(model.state_dict(), os.path.join(model_save_dir, save_name))
            print(f"  >>> 保存新最佳模型 (准确率 {best_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  → 验证准确率未提升 ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"⏸️ 早停触发！在第 {epoch+1} 个 epoch 停止训练。")
                break
    
    # ---- 7. 训练结束后，加载最优模型权重（回滚） ----
    if model_save_dir is None:
        model_save_dir = os.path.join(r'D:\python_work\cs', 'models', 'saved')
    best_weight_path = os.path.join(model_save_dir, f'best_{strategy}_{model_name}_{train_folder}.pth')
    if os.path.exists(best_weight_path):
        model.load_state_dict(torch.load(best_weight_path, map_location=device))
        print(f"🔄 已回滚到最优模型权重 (准确率 {best_acc:.2f}%)")
    
    print(f"\n✅ 完成！最佳验证准确率: {best_acc:.2f}%")
    
    # ---- 8. 保存实验结果到 JSON 文件 ----
    if result_save_dir is None:
        result_save_dir = os.path.join(r'D:\python_work\cs', 'results')
    import json as _json
    result_json_path = os.path.join(result_save_dir,
                                   f'result_{model_name}_{strategy}_{train_folder}.json')
    # 构造要保存的数据字典
    result_data = {
        'model_name': model_name,
        'strategy': strategy,
        'best_val_acc': best_acc,
        'history': history
    }
    with open(result_json_path, 'w', encoding='utf-8') as _f:
        _json.dump(result_data, _f, ensure_ascii=False, indent=2)
    print(f"📁 结果已保存至 {result_json_path}")
    
    return best_acc, history


# ==================== 测试入口 ====================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    base = r"D:\python_work\cs"

    # 示例1：在全量训练集上用三种微调策略运行 resnet18 实验
    for strat in ["freeze_all", "fine_tune_last", "full_fine_tune"]:
        run_experiment(data_root=os.path.join(base, "raw"),
                       train_folder="train",
                       strategy=strat,
                       epochs=10,
                       batch_size=32,
                       device=device)

    # 示例2：在 5-shot 小样本数据上运行 freeze_all 策略实验
    run_experiment(data_root=os.path.join(base, "data_small"),
                   train_folder="train_5shot",
                   strategy="freeze_all",
                   epochs=10,
                   batch_size=32,
                   device=device)
