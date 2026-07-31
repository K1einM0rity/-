import os
import sys
import json
import glob
import argparse
from PIL import Image
import torch
sys.path.append(r"D:\python_work\cs")
from models.model_utils import get_model
from data_process.dataloader import get_val_transforms
BASE_DIR = r"D:\python_work\cs"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据集类别名称（必须与训练时的 ImageFolder 顺序一致）
CLASS_NAMES = {
    "rice": ['Blast', 'BLB', 'BLS', 'BPB', 'BS', 'DH', 'DM', 'Healthy', 'Hispa', 'Tungro'],
    "rose": ['Rose_D04', 'Rose_H', 'Rose_P01', 'Rose_P02', 'Rose_R01', 'Rose_R02'],
    "tomato": ['Tomato_D01_ulcer', 'Tomato_D04_leaf_fungus', 'Tomato_D05_septoria_leaf_spot',
               'Tomato_D07_deer_virus', 'Tomato_D08_Yellow_Leaf_Curl_Virus', 'Tomato_D09_powdery_mildew',
               'Tomato_Healthy', 'Tomato_P03_leaf_miner', 'Tomato_P05_blueworms']
}

ALL_MODELS = ["resnet18", "resnet50", "efficientnet_v2_s"]

# ==================== 核心函数 ====================
def find_global_best_model(dataset):
    """扫描该数据集下所有模型的所有实验，返回全局最优的 (model_name, strategy, weight_path, best_acc)"""
    results_dir = os.path.join(BASE_DIR, "results", dataset)
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"未找到结果目录: {results_dir}")

    best_acc = -1
    best_model = None
    best_strategy = None
    best_weight_path = None

    for model_name in ALL_MODELS:
        json_files = glob.glob(os.path.join(results_dir, f"result_{model_name}_*.json"))
        for fp in json_files:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            acc = data.get('best_val_acc', 0)
            if acc > best_acc:
                best_acc = acc
                best_model = model_name
                best_strategy = data.get('strategy')
                train_folder = data.get('train_folder', 'train')
                model_dir = os.path.join(BASE_DIR, "models", f"saved_{dataset}")
                weight_candidate = os.path.join(model_dir, f"best_{best_strategy}_{best_model}_{train_folder}.pth")
                if not os.path.exists(weight_candidate):
                    weight_candidate = os.path.join(model_dir, f"best_{best_strategy}_{best_model}_train.pth")
                if os.path.exists(weight_candidate):
                    best_weight_path = weight_candidate
                else:
                    best_acc = -1
                    continue

    if best_weight_path is None:
        raise RuntimeError(f"未找到任何可用的模型权重文件，请检查 {results_dir} 和 {BASE_DIR}/models/saved_{dataset}")

    print(f"\n🌟 自动选择全局最优模型:")
    print(f"   数据集: {dataset}")
    print(f"   模型: {best_model}")
    print(f"   策略: {best_strategy}")
    print(f"   验证准确率: {best_acc:.2f}%")
    print(f"   权重文件: {os.path.basename(best_weight_path)}\n")
    return best_model, best_strategy, best_weight_path, best_acc

def load_model(dataset, model_name=None, strategy=None, weight_path=None):
    if model_name is None or strategy is None or weight_path is None:
        model_name, strategy, weight_path, _ = find_global_best_model(dataset)
    num_classes = len(CLASS_NAMES[dataset])
    model = get_model(num_classes=num_classes, strategy=strategy, model_name=model_name)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model, model_name, strategy

def predict_single_image(model, image_path, class_names):
    transform = get_val_transforms()
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    pred_class = class_names[pred_idx.item()]
    confidence = confidence.item() * 100
    all_probs = {class_names[i]: prob.item()*100 for i, prob in enumerate(probs[0])}
    return pred_class, confidence, all_probs

def predict_folder(model, folder_path, class_names):
    """批量预测文件夹内所有图片，只打印结果，不保存任何文件"""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"路径不是文件夹: {folder_path}")
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    if not files:
        print("文件夹中没有找到图片文件。")
        return
    print(f"\n共找到 {len(files)} 张图片，开始预测...\n")
    for fname in files:
        img_path = os.path.join(folder_path, fname)
        try:
            pred_class, conf, _ = predict_single_image(model, img_path, class_names)
            print(f"{fname}: {pred_class} ({conf:.2f}%)")
        except Exception as e:
            print(f"处理 {fname} 失败: {e}")




# ==================== 主程序 ====================
if __name__ == "__main__":

    DEFAULT_CONFIG = {
        "dataset": "rose",           # 可选: rice, rose, tomato
        "image": r"data_small_rose/val/Rose_D04/0003732_09103539.png",     # 单张图片路径（留空则使用folder模式）
        "folder": r""                # 批量预测文件夹路径（如果image非空，优先使用image）
    }

    if len(sys.argv) > 1:

        parser = argparse.ArgumentParser(description="农作物病害预测脚本（自动选择最优模型）")
        parser.add_argument("--dataset", type=str, required=True, choices=["rice", "rose", "tomato"])
        parser.add_argument("--model", type=str, choices=ALL_MODELS, help="手动指定模型（不指定则自动选择全局最优）")
        parser.add_argument("--strategy", type=str, choices=["freeze_all", "fine_tune_last", "full_fine_tune"])
        parser.add_argument("--weight", type=str)
        parser.add_argument("--image", type=str)
        parser.add_argument("--folder", type=str)
        args = parser.parse_args()

        if not args.image and not args.folder:
            print("错误: 请指定 --image 或 --folder")
            sys.exit(1)

        dataset = args.dataset
        image_path = args.image
        folder_path = args.folder

        if args.model:

            print("命令行手动指定模型功能请使用自动模式（去掉 --model）")
            sys.exit(1)

        try:
            model, model_name, strategy = load_model(dataset)
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
    else:

        if not DEFAULT_CONFIG["image"] and not DEFAULT_CONFIG["folder"]:
            print("请在 DEFAULT_CONFIG 中设置 image 或 folder 路径")
            sys.exit(1)
        dataset = DEFAULT_CONFIG["dataset"]
        image_path = DEFAULT_CONFIG["image"]
        folder_path = DEFAULT_CONFIG["folder"]
        try:
            model, model_name, strategy = load_model(dataset)
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)

    class_names = CLASS_NAMES[dataset]


    if image_path and os.path.exists(image_path):
        pred, conf, all_probs = predict_single_image(model, image_path, class_names)
        print("\n" + "="*50)
        print(f"图片: {image_path}")
        print(f"预测类别: {pred}")
        print(f"置信度: {conf:.2f}%")
        print("="*50)
        print("\n所有类别概率:")
        for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls:20s}: {prob:6.2f}%")
    elif folder_path and os.path.isdir(folder_path):
        predict_folder(model, folder_path, class_names)
    else:
        print("未找到有效的图片或文件夹路径，请检查 DEFAULT_CONFIG 或命令行参数")
        sys.exit(1)