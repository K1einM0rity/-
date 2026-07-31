"""
探索三个数据集（水稻、玫瑰、番茄）：
- 统计每个类别的训练集、验证集图片数量
- 为每个数据集生成一张样图（每个类别显示一张图片）
- 图片上标注：中文/完整类别名 + 训练样本数 + 验证样本数
"""

import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# ==================== 配置 ====================
BASE_DIR = r"D:\python_work\cs"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 数据集配置（含中文名称映射）
DATASETS = {
    "rice": {
        "path": os.path.join(BASE_DIR, "raw"),
        "display_names": {
            "Blast": "稻瘟病",
            "BLB": "细菌性条斑病",
            "BLS": "褐斑病",
            "BPB": "细菌性叶枯病",
            "BS": "胡麻斑病",
            "DH": "稻瘿蚊",
            "DM": "霜霉病",
            "Healthy": "健康",
            "Hispa": "稻铁甲虫",
            "Tungro": "东格鲁病",
        },
        "title": "水稻病害数据集",
    },
    "rose": {
        "path": os.path.join(BASE_DIR, "raw_rose"),
        "display_names": {
            "Rose_D04": "黑斑病",
            "Rose_H": "健康",
            "Rose_P01": "白粉病",
            "Rose_P02": "灰霉病",
            "Rose_R01": "锈病",
            "Rose_R02": "炭疽病",
        },
        "title": "玫瑰病害数据集",
    },
    "tomato": {
        "path": os.path.join(BASE_DIR, "raw_tomato"),
        "display_names": {
            "Tomato_D01_ulcer": "溃疡病",
            "Tomato_D04_leaf_fungus": "叶霉病",
            "Tomato_D05_septoria_leaf_spot": "斑枯病",
            "Tomato_D07_deer_virus": "鹿病毒病",
            "Tomato_D08_Yellow_Leaf_Curl_Virus": "黄化曲叶病毒病",
            "Tomato_D09_powdery_mildew": "白粉病",
            "Tomato_Healthy": "健康",
            "Tomato_P03_leaf_miner": "潜叶蛾",
            "Tomato_P05_blueworms": "青虫",
        },
        "title": "番茄病害数据集",
    },
}

def get_display_name(class_name, dataset_name):
    """返回显示用的中文名称（如果映射存在），否则返回原始类名"""
    mapping = DATASETS[dataset_name].get("display_names", {})
    return mapping.get(class_name, class_name)

def explore_dataset(dataset_name):
    """处理单个数据集，生成样图和统计"""
    print(f"\n{'='*50}")
    print(f"正在处理数据集: {dataset_name.upper()}")
    print(f"{'='*50}")
    
    data_root = DATASETS[dataset_name]["path"]
    train_path = os.path.join(data_root, "train")
    val_path = os.path.join(data_root, "val")
    
    if not os.path.exists(train_path):
        print(f"❌ 训练集路径不存在: {train_path}")
        return
    
    # 统计训练集各类别图片数
    train_counts = {}
    for class_name in os.listdir(train_path):
        class_dir = os.path.join(train_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        n_images = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        train_counts[class_name] = n_images
    
    # 统计验证集各类别图片数（如果存在）
    val_counts = {}
    if os.path.exists(val_path):
        for class_name in os.listdir(val_path):
            class_dir = os.path.join(val_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            n_images = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            val_counts[class_name] = n_images
    else:
        print(f"⚠️ 验证集路径不存在: {val_path}，将只统计训练集")
    
    # 打印统计表
    print("\n📊 类别统计（训练集 + 验证集）：")
    print(f"{'中文名称':<20} {'训练集':<8} {'验证集':<8} {'总计':<8}")
    total_train = 0
    total_val = 0
    for class_name in sorted(train_counts.keys()):
        train_cnt = train_counts.get(class_name, 0)
        val_cnt = val_counts.get(class_name, 0)
        total_train += train_cnt
        total_val += val_cnt
        display = get_display_name(class_name, dataset_name)
        print(f"{display:<20} {train_cnt:<8} {val_cnt:<8} {train_cnt+val_cnt:<8}")
    print(f"{'总计':<20} {total_train:<8} {total_val:<8} {total_train+total_val:<8}")
    
    # 生成样图（每个类别取第一张训练图片）
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    full_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    
    # 收集每个类别的第一张图片
    first_image_per_class = {}
    shown_labels = set()
    for img_tensor, label in full_dataset:
        class_name = full_dataset.classes[label]
        if class_name not in first_image_per_class:
            first_image_per_class[class_name] = img_tensor
            shown_labels.add(class_name)
        if len(shown_labels) == len(full_dataset.classes):
            break
    
    # 绘制样图网格
    class_names = sorted(first_image_per_class.keys())
    n_classes = len(class_names)
# 根据类别数动态选择列数，让排列更接近正方形，避免最后一行空缺过多
    if n_classes <= 6:
        n_cols = 3  # 玫瑰6类 → 2行×3列，完美填满
    elif n_classes <= 9:
        n_cols = 3  # 番茄9类 → 3行×3列，完美填满
    else:
        n_cols = 5  # 水稻10类 → 2行×5列

    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # 增加间距控制，防止标题重叠
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.tight_layout()
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols > 1 else [axes]
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    for i, class_name in enumerate(class_names):
        img_tensor = first_image_per_class[class_name]
        img_np = img_tensor.permute(1, 2, 0).numpy().clip(0, 1)
        ax = axes[i]
        ax.imshow(img_np)
        display_name = get_display_name(class_name, dataset_name)
        train_cnt = train_counts.get(class_name, 0)
        val_cnt = val_counts.get(class_name, 0)
        ax.set_title(f"{display_name}\n训练:{train_cnt}  验证:{val_cnt}", fontsize=9)
        ax.axis('off')
    
    # 隐藏多余的子图
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(DATASETS[dataset_name]["title"], fontsize=16, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"class_samples_{dataset_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 样图已保存: {save_path}")

if __name__ == "__main__":
    print("开始探索三个数据集...")
    for ds in DATASETS.keys():
        explore_dataset(ds)
    print("\n🎉 全部完成！样图保存在 results/ 目录下。")