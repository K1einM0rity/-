"""
补充小样本下的全参数微调（full_fine_tune）实验
自动遍历三个数据集（水稻、玫瑰、番茄）、三个小样本规模（5-shot,10-shot,20%）、三个模型（resnet18,resnet50,efficientnet_v2_s）
结果将保存到 results/{dataset}/ 和 models/saved_{dataset}/ 中，与原有实验完全兼容
"""

import os
import sys
import torch

sys.path.append(r"D:\python_work\cs")

from train.train_migration import run_experiment

# ==================== 配置 ====================
BASE_DIR = r"D:\python_work\cs"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
NUM_WORKERS = 4

# 默认批量大小（EfficientNetV2-S 全参数微调显存占用大，单独降低）
DEFAULT_BATCH_SIZE = 32
BATCH_SIZE_FOR_EFFICIENTNET = 16

# 数据集配置（注意小样本数据文件夹名称与 run_base_small.py 一致）
DATASETS = {
    "rice": {
        "data_small": os.path.join(BASE_DIR, "data_small_rice"),
        "num_classes": 10
    },
    "rose": {
        "data_small": os.path.join(BASE_DIR, "data_small_rose"),
        "num_classes": 6
    },
    "tomato": {
        "data_small": os.path.join(BASE_DIR, "data_small_tomato"),
        "num_classes": 9
    }
}

# 模型列表
MODELS = ["resnet18", "resnet50", "efficientnet_v2_s"]

# 小样本训练子文件夹名称
SMALL_SCALES = [
    ("5-shot", "train_5shot"),
    ("10-shot", "train_10shot"),
    ("20%", "train_20percent")
]

STRATEGY = "full_fine_tune"

def run_small_full_finetune():
    print("=" * 70)
    print("开始补充小样本全参数微调实验")
    print(f"策略: {STRATEGY}")
    print(f"设备: {DEVICE}")
    print("=" * 70)

    for dataset_name, cfg in DATASETS.items():
        data_root = cfg["data_small"]
        num_classes = cfg["num_classes"]

        if not os.path.exists(data_root):
            print(f"\n❌ 跳过 {dataset_name}：小样本数据路径不存在 {data_root}")
            continue

        result_save_dir = os.path.join(BASE_DIR, "results", dataset_name)
        model_save_dir = os.path.join(BASE_DIR, "models", f"saved_{dataset_name}")
        os.makedirs(result_save_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)

        for scale_name, train_folder in SMALL_SCALES:
            train_path = os.path.join(data_root, train_folder)
            if not os.path.exists(train_path):
                print(f"   ⚠️ 跳过 {dataset_name} - {scale_name}：训练文件夹不存在 {train_path}")
                continue

            for model_name in MODELS:
                # 针对 EfficientNetV2-S 降低 batch_size 防止显存溢出
                if model_name == "efficientnet_v2_s":
                    batch_size = BATCH_SIZE_FOR_EFFICIENTNET
                else:
                    batch_size = DEFAULT_BATCH_SIZE

                print(f"\n{'=' * 60}")
                print(f"正在运行: {dataset_name} | {scale_name} | {model_name} | {STRATEGY} (batch_size={batch_size})")
                print(f"{'=' * 60}")

                try:
                    best_acc, history = run_experiment(
                        data_root=data_root,
                        train_folder=train_folder,
                        strategy=STRATEGY,
                        model_name=model_name,
                        num_classes=num_classes,
                        epochs=EPOCHS,
                        batch_size=batch_size,
                        device=DEVICE,
                        num_workers=NUM_WORKERS,
                        patience=5,
                        result_save_dir=result_save_dir,
                        model_save_dir=model_save_dir
                    )
                    print(f"✅ 完成: {dataset_name} - {scale_name} - {model_name}，最佳验证准确率: {best_acc:.2f}%")
                except Exception as e:
                    print(f"❌ 错误: {dataset_name} - {scale_name} - {model_name}")
                    print(f"   异常信息: {e}")
                    # 继续运行下一个任务
                    continue

    print("\n" + "=" * 70)
    print("所有小样本全参数微调实验执行完毕！")
    print("结果已保存到 results/{dataset}/ 目录下")
    print("=" * 70)

if __name__ == "__main__":
    run_small_full_finetune()