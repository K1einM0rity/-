"""
小样本 vs 全量 准确率对比可视化（带策略标注）
自动扫描 results/{dataset}/ 下的所有 result_*.json 文件，
提取每个模型在每种数据规模下的最优验证准确率及其对应策略，
绘制分组柱状图并标注策略，最后保存图表和 CSV。
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
BASE_DIR = r"D:\python_work\cs"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "small_vs_full")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["rice", "rose", "tomato"]
MODELS = ["resnet18", "resnet50", "efficientnet_v2_s"]

SCALES = [
    ("5-shot", "train_5shot", True),
    ("10-shot", "train_10shot", True),
    ("20%", "train_20percent", True),
    ("Full", "train", False),
]

# 小样本实验包含的策略（全参数微调已补充）
SMALL_STRATEGIES = ["freeze_all", "fine_tune_last", "full_fine_tune", "baseline", "Baseline"]
# 全量实验同样包含这些策略
FULL_STRATEGIES = ["freeze_all", "fine_tune_last", "full_fine_tune", "baseline", "Baseline"]

# 策略名称的显示简写（用于柱状图标注）
STRATEGY_DISPLAY = {
    "freeze_all": "冻结",
    "fine_tune_last": "微调最后",
    "full_fine_tune": "全参数微调",
    "baseline": "从头训练",
    "Baseline": "从头训练"
}

def get_best_acc_and_strategy(dataset, model, scale_folder, is_small):
    """
    返回 (best_acc, best_strategy)
    """
    dataset_results_dir = os.path.join(RESULTS_DIR, dataset)
    if not os.path.exists(dataset_results_dir):
        return None, None

    strategies = SMALL_STRATEGIES if is_small else FULL_STRATEGIES
    best_acc = -1.0
    best_strategy = None

    for strategy in strategies:
        if scale_folder == "train":
            pattern = f"result_{model}_{strategy}_train.json"
        else:
            pattern = f"result_{model}_{strategy}_{scale_folder}.json"
        file_path = os.path.join(dataset_results_dir, pattern)

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    acc = data.get("best_val_acc", -1)
                    if acc > best_acc:
                        best_acc = acc
                        best_strategy = strategy
            except Exception as e:
                print(f"  读取失败 {file_path}: {e}")

    # 兜底搜索（防止文件名大小写或额外后缀）
    if best_acc < 0:
        search_pattern = os.path.join(dataset_results_dir, f"*{model}*{scale_folder}*.json")
        files = glob.glob(search_pattern)
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    acc = data.get("best_val_acc", -1)
                    if acc > best_acc:
                        best_acc = acc
                        best_strategy = data.get("strategy", "unknown")
            except:
                pass

    return best_acc if best_acc >= 0 else None, best_strategy

def plot_dataset(dataset):
    print(f"\n处理数据集: {dataset}")

    # 数据结构：{model: [(acc, strategy), ...]} 顺序对应 SCALES
    data = {model: [] for model in MODELS}

    for scale_name, scale_folder, is_small in SCALES:
        for model in MODELS:
            acc, strategy = get_best_acc_and_strategy(dataset, model, scale_folder, is_small)
            if acc is None:
                print(f"  ⚠️ 未找到 {dataset} - {model} - {scale_name} 的结果")
                acc = 0
                strategy = "无数据"
            data[model].append((acc, strategy))

    # 绘图
    x = np.arange(len(SCALES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, model in enumerate(MODELS):
        acc_vals = [item[0] for item in data[model]]
        strategies = [item[1] for item in data[model]]

        if max(acc_vals) == 0:
            continue

        offset = (i - 1) * width
        bars = ax.bar(x + offset, acc_vals, width, label=model, color=colors[i])

        # 为每个柱子添加准确率数值和策略标注
        for bar, acc_val, strat in zip(bars, acc_vals, strategies):
            if acc_val <= 0:
                continue
            # 准确率数值放在柱子顶部
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{acc_val:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
            # 策略标注放在准确率数值上方（更靠上）或柱子内部
            # 如果柱子高度足够（>15），放在柱子内部顶部；否则放在准确率数值上方
            if bar.get_height() > 15:
                # 放入柱子内部顶部
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                        STRATEGY_DISPLAY.get(strat, strat), ha='center', va='top',
                        fontsize=7, color='white', fontweight='bold')
            else:
                # 放在准确率数值上方（柱子外部）
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                        STRATEGY_DISPLAY.get(strat, strat), ha='center', va='bottom',
                        fontsize=7, color='darkred')

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in SCALES])
    ax.set_ylabel("验证准确率 (%)")
    ax.set_title(f"{dataset.upper()} 数据集 - 小样本 vs 全量最优策略准确率对比\n(柱子顶部数值为准确率，标注为达到该准确率的最佳策略)")
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{dataset}_small_vs_full.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  已保存图表: {out_path}")

    # 保存 CSV（包含策略信息）
    csv_path = os.path.join(OUTPUT_DIR, f"{dataset}_small_vs_full.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        # 表头
        headers = ["Model"] + [s[0] for s in SCALES] + [f"{s[0]}_strategy" for s in SCALES]
        f.write(",".join(headers) + "\n")
        for model in MODELS:
            row = [model]
            for acc, strat in data[model]:
                row.append(f"{acc:.2f}" if acc > 0 else "")
            for acc, strat in data[model]:
                row.append(STRATEGY_DISPLAY.get(strat, strat) if strat else "")
            f.write(",".join(row) + "\n")
    print(f"  已保存CSV (含策略): {csv_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("小样本 vs 全量 准确率对比可视化（带策略标注）")
    print("=" * 60)
    for ds in DATASETS:
        ds_results_dir = os.path.join(RESULTS_DIR, ds)
        if not os.path.exists(ds_results_dir):
            print(f"⚠️ 跳过 {ds}：results/{ds} 文件夹不存在")
            continue
        plot_dataset(ds)
    print("\n✅ 完成！图表保存在:", OUTPUT_DIR)