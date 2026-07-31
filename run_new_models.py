import os, sys
# 将项目根目录添加到系统路径，方便导入自定义模块
sys.path.append(r"D:\python_work\cs")
import torch

# 从自定义模块导入实验运行函数
from train.train_migration import run_experiment          # 迁移学习实验
from train.train_baseline import run_baseline             # 从头训练基线实验
from data_process.build_small_dataset import build_all_small_datasets  # 生成小样本数据集

# ===== 全局配置 =====
BASE = r"D:\python_work\cs"                # 项目根目录
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择 GPU 或 CPU
EPOCHS = 10                                # 训练轮数
BATCH_SIZE = 32                            # 批次大小
NUM_WORKERS = 4                            # 数据加载子进程数

# ===== 任务列表（按顺序执行）=====
# 每个任务包含数据集名称、原始数据文件夹、类别数、要测试的模型列表
TASKS = [
    {"dataset": "rice",   "raw_dir": "raw",       "num_classes": 10, "models": ["resnet18", "resnet50", "efficientnet_v2_s"]},
    {"dataset": "rose",   "raw_dir": "raw_rose",  "num_classes": 6,  "models": ["resnet18", "resnet50", "efficientnet_v2_s"]},
    {"dataset": "tomato", "raw_dir": "raw_tomato", "num_classes": 9,  "models": ["resnet18", "resnet50", "efficientnet_v2_s"]},
]

# 定义小样本数据集的规模：名称和对应的训练集子文件夹名
SMALL_SCALES = [
    ("5-shot", "train_5shot"),          # 每类 5 张
    ("10-shot", "train_10shot"),        # 每类 10 张
    ("20%", "train_20percent"),         # 每类随机抽取 20%
]

# 全量数据迁移学习采用的微调策略
FULL_STRATEGIES = ["freeze_all", "fine_tune_last", "full_fine_tune"]
# 小样本迁移学习采用的微调策略
SMALL_STRATEGIES = ["freeze_all", "fine_tune_last", "full_fine_tune"]

def run_task(dataset, raw_dir, num_classes, model_name):
    """
    针对一个特定数据集和模型，运行所有预定的实验组合。
    包括：全量数据迁移学习、小样本迁移学习、全量数据从头训练、小样本从头训练（仅ResNet18）。

    参数:
        dataset:     数据集名称（如 'rice'）
        raw_dir:     原始数据文件夹名（如 'raw'）
        num_classes: 分类类别数
        model_name:  使用的模型名称（如 'resnet18'）
    """
    # 全量数据集的路径
    data_root = os.path.join(BASE, raw_dir)
    
    # 结果保存路径：results/数据集名称
    result_save_dir = os.path.join(BASE, 'results', dataset)
    # 模型保存路径：models/saved_数据集名称
    model_dir = os.path.join(BASE, 'models', f'saved_{dataset}')
    os.makedirs(result_save_dir, exist_ok=True)  # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    # ----- 1. 全量数据迁移学习 -----
    for s in FULL_STRATEGIES:
        run_experiment(
            data_root=data_root, train_folder="train",  # 使用原始训练集
            strategy=s, model_name=model_name,
            num_classes=num_classes,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            device=DEVICE, num_workers=NUM_WORKERS,
            result_save_dir=result_save_dir,
            model_save_dir=model_dir)
    
    # ----- 2. 小样本迁移学习 -----
    small_base = os.path.join(BASE, f'data_small_{dataset}')  # 小样本数据集的根目录
    for _, tf in SMALL_SCALES:   # 遍历 5-shot, 10-shot, 20%
        for s in SMALL_STRATEGIES:
            run_experiment(
                data_root=small_base, train_folder=tf,  # 使用对应的小样本训练集
                strategy=s, model_name=model_name,
                num_classes=num_classes,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                device=DEVICE, num_workers=NUM_WORKERS,
                result_save_dir=result_save_dir,
                model_save_dir=model_dir)
    
    # ----- 3. 全量数据从头训练（无预训练权重的基线） -----
    run_baseline(
        data_root=data_root, train_folder="train",
        num_classes=num_classes, epochs=EPOCHS,
        device=DEVICE,
        model_save_dir=model_dir,
        model_name=model_name,
        result_save_dir=result_save_dir,
        dataset_name=dataset)
    
    # ----- 4. 小样本从头训练（仅对 ResNet18 进行，其他模型跳过以节省时间） -----
    if model_name == "resnet18":
        for _, tf in SMALL_SCALES:
            run_baseline(
                data_root=small_base,
                train_folder=tf,
                num_classes=num_classes,
                epochs=EPOCHS,
                device=DEVICE,
                model_name=model_name,
                num_workers=NUM_WORKERS,
                model_save_dir=model_dir,
                result_save_dir=result_save_dir,
                dataset_name=dataset)
    
    print(f"\n{'='*60}")
    print(f"{dataset} - {model_name} 全部实验完成！")
    print(f"{'='*60}")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 按顺序执行所有任务
    for task in TASKS:
        dataset = task["dataset"]
        raw_dir = task["raw_dir"]
        num_classes = task["num_classes"]
        models = task["models"]
        
        # 检查并生成小样本数据集（如果尚未生成）
        small_dir = os.path.join(BASE, f'data_small_{dataset}')
        # 通过检查 5-shot 文件夹是否存在来判断是否需要生成
        if not os.path.exists(os.path.join(small_dir, 'train_5shot')):
            build_all_small_datasets(
                base_raw_path=os.path.join(BASE, raw_dir),
                output_base=small_dir
            )
        
        # 遍历该任务下的所有模型，逐一运行实验
        for model_name in models:
            run_task(dataset, raw_dir, num_classes, model_name)
