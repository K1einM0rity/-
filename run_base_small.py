"""
批量运行三个数据集的小样本 ResNet18 从头训练
会自动生成 JSON 到 results/{dataset}/ 目录下
"""
import os
import sys
sys.path.append(r"D:\python_work\cs")

import torch
from train.train_baseline import run_baseline

BASE = r"D:\python_work\cs"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据集配置
DATASETS = {
    "rice": {
        "data_root": os.path.join(BASE, "data_small_rice"),
        "num_classes": 10
    },
    "rose": {
        "data_root": os.path.join(BASE, "data_small_rose"),
        "num_classes": 6
    },
    "tomato": {
        "data_root": os.path.join(BASE, "data_small_tomato"),
        "num_classes": 9
    }
}

# 小样本训练子文件夹名称
SMALL_TRAIN_FOLDERS = ["train_5shot", "train_10shot", "train_20percent"]

def main():
    print("="*60)
    print("批量运行小样本 ResNet18 从头训练")
    print("="*60)

    for dataset_name, cfg in DATASETS.items():
        data_root = cfg["data_root"]
        num_classes = cfg["num_classes"]

        # 模型权重保存目录（每个数据集独立）
        model_save_dir = os.path.join(BASE, "models", f"saved_{dataset_name}")
        os.makedirs(model_save_dir, exist_ok=True)

        # 结果 JSON 保存目录
        result_save_dir = os.path.join(BASE, "results", dataset_name)
        os.makedirs(result_save_dir, exist_ok=True)

        for train_folder in SMALL_TRAIN_FOLDERS:
            print(f"\n{'='*60}")
            print(f"数据集: {dataset_name} | 训练集: {train_folder}")
            print(f"{'='*60}")

            # 检查小样本数据是否存在
            train_path = os.path.join(data_root, train_folder)
            if not os.path.exists(train_path):
                print(f"❌ 跳过：路径不存在 {train_path}")
                continue

            best_acc, history = run_baseline(
                data_root=data_root,
                train_folder=train_folder,
                num_classes=num_classes,
                epochs=10,
                device=DEVICE,
                model_name="resnet18",
                num_workers=4,
                model_save_dir=model_save_dir,
                result_save_dir=result_save_dir,
                dataset_name=dataset_name
            )
            print(f"✅ {dataset_name} - {train_folder} 完成，最佳准确率: {best_acc:.2f}%")

    print("\n🎉 所有小样本从头训练完成！")

if __name__ == "__main__":
    main()