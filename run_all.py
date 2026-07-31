"""
统一实验启动脚本（答辩用，无需重新运行）
包含：
- 全量数据：所有策略（freeze_all, fine_tune_last, full_fine_tune）
- 小样本数据（5-shot,10-shot,20%）：
    * 迁移策略：freeze_all, fine_tune_last, full_fine_tune
    * 从头训练（baseline）：仅 ResNet18
"""

import os
import sys
sys.path.append(r"D:\python_work\cs")
import torch
from train.train_migration import run_experiment
from train.train_baseline import run_baseline
from data_process.build_small_dataset import build_all_small_datasets

# ==================== 全局配置 ====================
BASE = r"D:\python_work\cs"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 4

# 针对 EfficientNetV2-S 降低 batch_size（避免显存溢出，答辩时不需要实际跑，但保留配置）
BATCH_SIZE_EFFICIENTNET = 16


DATASETS = [
    {"name": "rice",   "raw_dir": "raw",        "num_classes": 10},
    {"name": "rose",   "raw_dir": "raw_rose",   "num_classes": 6},
    {"name": "tomato", "raw_dir": "raw_tomato", "num_classes": 9},
]

SMALL_SCALES = [
    ("5-shot", "train_5shot"),
    ("10-shot", "train_10shot"),
    ("20%", "train_20percent"),
]
FULL_STRATEGIES = ["freeze_all", "fine_tune_last", "full_fine_tune"]
SMALL_TRANSFER_STRATEGIES = ["freeze_all", "fine_tune_last", "full_fine_tune"]

BASELINE_MODEL = "resnet18"

# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("统一实验启动脚本（仅用于展示代码结构，无需重新运行）")
    print("=" * 70)

    for ds in DATASETS:
        dataset_name = ds["name"]
        raw_dir = ds["raw_dir"]
        num_classes = ds["num_classes"]

        # 1. 准备小样本数据文件夹（如果不存在则生成）
        small_base = os.path.join(BASE, f"data_small_{dataset_name}")
        if not os.path.exists(os.path.join(small_base, "train_5shot")):
            print(f"生成小样本数据: {dataset_name}")
            build_all_small_datasets(
                base_raw_path=os.path.join(BASE, raw_dir),
                output_base=small_base
            )

        # 2. 全量数据实验
        print(f"\n--- 全量数据: {dataset_name} ---")
        data_root_full = os.path.join(BASE, raw_dir)
        result_dir = os.path.join(BASE, "results", dataset_name)
        model_dir = os.path.join(BASE, "models", f"saved_{dataset_name}")

        for strategy in FULL_STRATEGIES:
            # 每个模型都跑
            for model_name in ["resnet18", "resnet50", "efficientnet_v2_s"]:
                batch = BATCH_SIZE_EFFICIENTNET if model_name == "efficientnet_v2_s" else BATCH_SIZE
                print(f"  全量: {model_name} - {strategy}")
                run_experiment(
                    data_root=data_root_full,
                    train_folder="train",
                    strategy=strategy,
                    model_name=model_name,
                    num_classes=num_classes,
                    epochs=EPOCHS,
                    batch_size=batch,
                    device=DEVICE,
                    num_workers=NUM_WORKERS,
                    result_save_dir=result_dir,
                    model_save_dir=model_dir
                )

        # 3. 小样本迁移学习实验（包括全参数微调）
        print(f"\n--- 小样本迁移学习: {dataset_name} ---")
        for scale_name, train_folder in SMALL_SCALES:
            for model_name in ["resnet18", "resnet50", "efficientnet_v2_s"]:
                for strategy in SMALL_TRANSFER_STRATEGIES:
                    batch = BATCH_SIZE_EFFICIENTNET if model_name == "efficientnet_v2_s" else BATCH_SIZE
                    print(f"  小样本 {scale_name}: {model_name} - {strategy}")
                    run_experiment(
                        data_root=small_base,
                        train_folder=train_folder,
                        strategy=strategy,
                        model_name=model_name,
                        num_classes=num_classes,
                        epochs=EPOCHS,
                        batch_size=batch,
                        device=DEVICE,
                        num_workers=NUM_WORKERS,
                        result_save_dir=result_dir,
                        model_save_dir=model_dir
                    )

        # 4. 小样本从头训练（仅 ResNet18）
        print(f"\n--- 小样本从头训练 (ResNet18): {dataset_name} ---")
        for scale_name, train_folder in SMALL_SCALES:
            print(f"  baseline {scale_name}")
            run_baseline(
                data_root=small_base,
                train_folder=train_folder,
                num_classes=num_classes,
                epochs=EPOCHS,
                device=DEVICE,
                model_name=BASELINE_MODEL,
                num_workers=NUM_WORKERS,
                model_save_dir=model_dir,
                result_save_dir=result_dir,
                dataset_name=dataset_name
            )

    print("\n" + "=" * 70)
    print("所有实验任务已定义完毕（实际运行已注释，仅为展示结构）")
    print("=" * 70)

if __name__ == "__main__":
    # 注意：由于训练耗时极长，此脚本默认不会真正运行训练。
    # 如果需要重新训练，请取消下面一行的注释，并确保有足够的时间。
    # main()
    print("统一启动脚本已加载。如需重新运行全部实验，请取消 main() 注释。")