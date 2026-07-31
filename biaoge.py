"""
生成三个数据集的准确率表格图片
行：模型 + 策略组合
列：5-shot, 10-shot, 20%, Full
每个数据集生成一张表格图片，保存在 results/tables/ 目录下
"""

import os
import json
import matplotlib.pyplot as plt

# 修复中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = r"D:\python_work\cs"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["rice", "rose", "tomato"]
DATASET_NAMES = {
    "rice": "水稻病害数据集",
    "rose": "玫瑰病害数据集",
    "tomato": "番茄病害数据集"
}
MODELS = ["resnet18", "resnet50", "efficientnet_v2_s"]

# 迁移学习策略（所有模型都跑）
TRANSFER_STRATEGIES = ["freeze_all", "fine_tune_last", "full_fine_tune"]
# 从头训练只有 ResNet18 跑了
BASELINE_MODEL = "resnet18"

STRATEGY_DISPLAY = {
    "freeze_all": "冻结全部特征层",
    "fine_tune_last": "微调最后一层",
    "full_fine_tune": "全参数微调",
    "baseline": "从头训练"
}

SCALES = [
    ("5-shot", "train_5shot"),
    ("10-shot", "train_10shot"),
    ("20%", "train_20percent"),
    ("Full", "train")
]

def get_accuracy(dataset, model, strategy, scale_folder):
    dataset_results_dir = os.path.join(RESULTS_DIR, dataset)
    if scale_folder == "train":
        filename = f"result_{model}_{strategy}_train.json"
    else:
        filename = f"result_{model}_{strategy}_{scale_folder}.json"
    file_path = os.path.join(dataset_results_dir, filename)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                acc = data.get("best_val_acc", -1)
                return acc
        except Exception as e:
            print(f"读取失败: {file_path}, {e}")
    return None

def generate_table_for_dataset(dataset):
    rows = []
    row_labels = []
    
    for model in MODELS:
        # 添加迁移学习策略
        for strategy in TRANSFER_STRATEGIES:
            row_labels.append(f"{model}\n{STRATEGY_DISPLAY[strategy]}")
            row_data = []
            for scale_name, scale_folder in SCALES:
                acc = get_accuracy(dataset, model, strategy, scale_folder)
                if acc is None:
                    row_data.append("—")
                else:
                    row_data.append(f"{acc:.2f}%")
            rows.append(row_data)
        
        # 只有 ResNet18 添加 baseline
        if model == BASELINE_MODEL:
            row_labels.append(f"{model}\n{STRATEGY_DISPLAY['baseline']}")
            row_data = []
            for scale_name, scale_folder in SCALES:
                acc = get_accuracy(dataset, model, "baseline", scale_folder)
                if acc is None:
                    row_data.append("—")
                else:
                    row_data.append(f"{acc:.2f}%")
            rows.append(row_data)
    
    fig, ax = plt.subplots(figsize=(12, len(rows) * 0.5 + 2))
    ax.axis('off')
    columns = [s[0] for s in SCALES]
    table = ax.table(cellText=rows, rowLabels=row_labels, colLabels=columns,
                     loc='center', cellLoc='center', colWidths=[0.15]*len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头行
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif j == -1:  # 行标签列
            cell.set_facecolor('#E7E6E6')
            cell.set_text_props(fontweight='bold')
    
    ax.set_title(f"{DATASET_NAMES[dataset]} 各模型及策略准确率 (%)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{dataset}_accuracy_table.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存表格图片: {save_path}")

def main():
    print("=" * 60)
    print("生成三个数据集的准确率表格图片")
    print("=" * 60)
    for dataset in DATASETS:
        dataset_results_dir = os.path.join(RESULTS_DIR, dataset)
        if not os.path.exists(dataset_results_dir):
            print(f"⚠️ 跳过 {dataset}：results/{dataset} 文件夹不存在")
            continue
        generate_table_for_dataset(dataset)
    print("\n🎉 完成！表格图片保存在:", OUTPUT_DIR)

if __name__ == "__main__":
    main()