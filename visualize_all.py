"""
可视化脚本：支持水稻/番茄/玫瑰三个数据集，自动遍历并分别生成图表
"""
import os, sys, json, glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE = r"D:\python_work\cs"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 数据集配置 =====
DATASETS = ["rice", "rose", "tomato"]

DATASET_CONFIG = {
    "rice": {
        "data_root": os.path.join(BASE, "raw"),
        "num_classes": 10,
        "class_names": ['Blast', 'BLB', 'BLS', 'BPB', 'BS', 'DH', 'DM', 'Healthy', 'Hispa', 'Tungro'],
    },
    "rose": {
        "data_root": os.path.join(BASE, "raw_rose"),
        "num_classes": 6,
        "class_names": ['Rose_D04', 'Rose_H', 'Rose_P01', 'Rose_P02', 'Rose_R01', 'Rose_R02'],
    },
    "tomato": {
        "data_root": os.path.join(BASE, "raw_tomato"),
        "num_classes": 9,
        "class_names": ['Tomato_D01_ulcer', 'Tomato_D04_leaf_fungus', 'Tomato_D05_septoria_leaf_spot', 'Tomato_D07_deer_virus', 'Tomato_D08_Yellow_Leaf_Curl_Virus', 'Tomato_D09_powdery_mildew', 'Tomato_Healthy', 'Tomato_P03_leaf_miner', 'Tomato_P05_blueworms'],
    },
}

sys.path.append(BASE)
from models.model_utils import get_model
from data_process.dataloader import create_dataloaders

def process_dataset(DATASET):
    """处理单个数据集的所有可视化任务"""
    print(f"\n{'='*60}")
    print(f"正在处理数据集: {DATASET}")
    print(f"{'='*60}")
    
    cfg = DATASET_CONFIG[DATASET]
    DATA_ROOT = cfg["data_root"]
    NUM_CLASSES = cfg["num_classes"]
    CLASS_NAMES = cfg["class_names"]
    
    output_dir = os.path.join(BASE, 'results', DATASET)
    os.makedirs(output_dir, exist_ok=True)

    def find_best_model():
        json_files = glob.glob(os.path.join(BASE, 'results', DATASET, 'result_*.json'))
        if not json_files:
            return {}
        best_per_model = {}
        for fp in json_files:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            model = data['model_name']
            acc = data['best_val_acc']
            if model not in best_per_model or acc > best_per_model[model]['acc']:
                best_per_model[model] = {
                    'acc': acc,
                    'strategy': data['strategy'],
                    'train_folder': data.get('train_folder', 'train'),
                    'json_path': fp
                }
        return best_per_model

    def plot_best_curves(best_models):
        for model, info in best_models.items():
            with open(info['json_path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            history = data['history']
            strategy = info['strategy']
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='训练损失')
            plt.plot(history['val_loss'], label='验证损失')
            plt.xlabel('轮次'); plt.ylabel('损失'); plt.legend()
            plt.title(f'{model} - 最优策略 ({strategy}) 损失')
            plt.grid(True, alpha=0.3)
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='训练准确率')
            plt.plot(history['val_acc'], label='验证准确率')
            plt.xlabel('轮次'); plt.ylabel('准确率 (%)'); plt.legend()
            plt.title(f'{model} - 最优策略 ({strategy}) 准确率')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'training_curves_{model}_best.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"📈 {model} 最优策略训练曲线已保存")

    def evaluate_and_plot(model_name, strategy, model_path):
        model = get_model(num_classes=NUM_CLASSES, strategy=strategy, model_name=model_name)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        _, loader, _ = create_dataloaders(data_root=DATA_ROOT, batch_size=32, train_folder="train", num_workers=0)
        all_labels, all_preds = [], []
        with torch.no_grad():
            for img, lbl in loader:
                img, lbl = img.to(DEVICE), lbl.to(DEVICE)
                out = model(img)
                _, pred = torch.max(out, 1)
                all_labels.extend(lbl.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
        
        # 打印到终端
        print(f"\n{model_name} 最优模型 ({strategy}) 评估准确率: {acc:.4f}")
        print(report)
        
        # 保存为 txt 文件
        report_path = os.path.join(output_dir, f'classification_report_{model_name}.txt')
        with open(report_path, 'w', encoding='utf-8') as rf:
            rf.write(f"模型: {model_name}  策略: {strategy}\n")
            rf.write(f"验证准确率: {acc:.4f}\n")
            rf.write("=" * 55 + "\n")
            rf.write(report)
        print(f"📝 分类报告已保存至: {report_path}")
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'{model_name} - 混淆矩阵 ({strategy})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'), dpi=150)
        plt.close()
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'{model_name} - 归一化混淆矩阵 ({strategy})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}_normalized.png'), dpi=150)
        plt.close()
        return acc

    def plot_comparison():
        all_data = {}
        for mn in ["resnet18", "resnet50", "efficientnet_v2_s"]:
            for s in ["freeze_all", "fine_tune_last", "full_fine_tune", "baseline"]:
                if s == "Baseline":
                    fp = os.path.join(BASE, 'results', DATASET, f'result_{mn}_baseline_train.json')
                else:
                    fp = os.path.join(BASE, 'results', DATASET, f'result_{mn}_{s}_train.json')
                if os.path.exists(fp):
                    with open(fp, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                        key = f"{d['model_name']}-{d['strategy']}"
                        all_data[key] = d['best_val_acc']
        if not all_data:
            return
        plt.figure(figsize=(14, 6))
        sorted_items = sorted(all_data.items(), key=lambda x: x[1])
        names = [item[0] for item in sorted_items]
        vals = [item[1] for item in sorted_items]
        plt.bar(names, vals, color=['steelblue', 'orange', 'green', 'red', 'purple', 'gray'][:len(names)])
        plt.ylabel("验证准确率 (%)")
        plt.title(f"{DATASET} - 模型对比")
        plt.ylim(0, 105)
        for i, v in enumerate(vals):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_core.png'), dpi=150)
        plt.close()
        print("✅ 核心对比柱状图已保存")

    # --- 主程序处理当前数据集 ---
    best_models = find_best_model()
    if not best_models:
        print(f"❌ 数据集 {DATASET} 没有找到任何实验结果，跳过")
        return

    print(f"📊 检测到 {len(best_models)} 个模型的最优策略")
    plot_best_curves(best_models)

    for model_name, info in best_models.items():
        strategy = info['strategy']
        train_folder = info['train_folder']
        
        # --- 专门为 evaluate_and_plot 准备的正确路径查找逻辑 ---
        model_path = None
        
        # 1. 尝试最基本的完整路径（包含 train_folder）
        model_path = os.path.join(BASE, 'models', f'saved_{DATASET}', f'best_{strategy}_{model_name}_{train_folder}.pth')
        
        # 2. 如果没找到（比如是小样本的最优结果），尝试后备的简单路径
        if not os.path.exists(model_path):
            fallback_path = os.path.join(BASE, 'models', f'saved_{DATASET}', f'best_{strategy}_{model_name}_train.pth')
            if os.path.exists(fallback_path):
                model_path = fallback_path
        
        # 3. 如果还是没找到，尝试自动搜索
        if not os.path.exists(model_path):
            search_pattern = os.path.join(BASE, 'models', f'saved_{DATASET}', f'best_{strategy}_{model_name}_*.pth')
            candidates = glob.glob(search_pattern)
            if candidates:
                model_path = candidates[0]  # 取第一个匹配的文件
                print(f"   通过自动搜索找到模型文件: {model_path}")
        
        # 执行评估或报错
        if model_path and os.path.exists(model_path):
            evaluate_and_plot(model_name, strategy, model_path)
        else:
            # 列出所有可能的路径供调试
            all_models_in_dir = glob.glob(os.path.join(BASE, 'models', f'saved_{DATASET}', f'best_*_{model_name}_*.pth'))
            print(f"⚠️ 无法找到 {model_name} 的模型文件。")
            print(f"   尝试过的主要路径: {os.path.join(BASE, 'models', f'saved_{DATASET}', f'best_{strategy}_{model_name}_{train_folder}.pth')}")
            if all_models_in_dir:
                print(f"   该目录下存在的相关文件: {all_models_in_dir}")
            else:
                print(f"   该目录下没有任何 {model_name} 的文件。请检查训练是否完好。")

    plot_comparison()
    print(f"🎉 数据集 {DATASET} 全部图表生成完毕！")

if __name__ == "__main__":
    print("=" * 50)
    print("开始生成全部数据集的可视化图表...")
    print("=" * 50)
    for dataset in DATASETS:
        result_dir = os.path.join(BASE, 'results', dataset)
        if not os.path.exists(result_dir):
            print(f"⚠️ 数据集 {dataset} 的结果文件夹不存在，跳过")
            continue
        process_dataset(dataset)
    print("\n🎉🎉🎉 所有数据集的可视化图表已全部生成！🎉🎉🎉")