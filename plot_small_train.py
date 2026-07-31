"""
绘制小样本收敛速度对比图：每个数据集×每个小样本规模下，
最优迁移模型 vs 从头训练(ResNet18) 的验证准确率随epoch变化曲线
"""

import os
import json
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = r"D:\python_work\cs"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "convergence_curves")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据集列表及中文名
DATASETS = ["rice", "rose", "tomato"]
DATASET_NAMES = {"rice": "水稻", "rose": "玫瑰", "tomato": "番茄"}

# 小样本规模（文件夹名，显示名，用于文件/标题的显示名）
SMALL_SCALES = [
    ("train_5shot", "5-shot"),
    ("train_10shot", "10-shot"),
    ("train_20percent", "20%全量"),   # 显示时自动加“全量”
]

# 所有可能的模型和策略（用于扫描最优迁移模型）
MODELS = ["resnet18", "resnet50", "efficientnet_v2_s"]
STRATEGIES = ["freeze_all", "fine_tune_last", "full_fine_tune"]

def find_best_transfer_model(dataset, scale_folder):
    """返回该数据集该规模下最优迁移模型的 (model, strategy, history, best_acc)"""
    result_dir = os.path.join(RESULTS_DIR, dataset)
    best_acc = -1
    best_model = None
    best_strategy = None
    best_history = None
    for model in MODELS:
        for strategy in STRATEGIES:
            json_path = os.path.join(result_dir, f"result_{model}_{strategy}_{scale_folder}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                acc = data.get('best_val_acc', 0)
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_strategy = strategy
                    best_history = data.get('history')
    return best_model, best_strategy, best_history, best_acc

def get_baseline_history(dataset, scale_folder):
    """返回从头训练(ResNet18)的history和best_acc"""
    json_path = os.path.join(RESULTS_DIR, dataset, f"result_resnet18_baseline_{scale_folder}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('history'), data.get('best_val_acc', 0)
    else:
        return None, None

def plot_curve(dataset, scale_folder, display_name):
    """绘制单张对比曲线，display_name 用于文件名和标题（如 '5-shot', '20%全量'）"""
    print(f"绘制 {dataset} - {display_name} ...")
    best_model, best_strategy, best_history, best_acc = find_best_transfer_model(dataset, scale_folder)
    baseline_history, baseline_acc = get_baseline_history(dataset, scale_folder)

    if best_history is None:
        print(f"  警告: 未找到 {dataset} - {scale_folder} 的最优迁移模型")
        return
    if baseline_history is None:
        print(f"  警告: 未找到 {dataset} - {scale_folder} 的 baseline")
        return

    best_val_acc = best_history.get('val_acc', [])
    baseline_val_acc = baseline_history.get('val_acc', [])
    if not best_val_acc or not baseline_val_acc:
        print(f"  警告: 缺少 val_acc 数据")
        return

    epochs_best = list(range(1, len(best_val_acc)+1))
    epochs_baseline = list(range(1, len(baseline_val_acc)+1))

    # 构造图片标题和文件名
    title = f"{DATASET_NAMES[dataset]}{display_name}数据集-识别性能对比"
    filename = f"{title}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_best, best_val_acc, 
             label=f"最优迁移 ({best_model} - {best_strategy}) [Best: {best_acc:.1f}%]", 
             color='#2ca02c', marker='o', markersize=4, linewidth=2)
    plt.plot(epochs_baseline, baseline_val_acc, 
             label=f"从头训练 (ResNet18) [Best: {baseline_acc:.1f}%]",
             color='#d62728', marker='s', markersize=4, linewidth=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('验证准确率 (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  已保存: {save_path}")

def main():
    print("=" * 60)
    print("开始生成小样本识别性能对比曲线...")
    for dataset in DATASETS:
        for scale_folder, display_name in SMALL_SCALES:
            plot_curve(dataset, scale_folder, display_name)
    print("\n✅ 完成！图片保存在:", OUTPUT_DIR)

if __name__ == "__main__":
    main()