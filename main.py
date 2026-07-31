import torch                # PyTorch深度学习框架核心库
import torchvision          # PyTorch的计算机视觉工具包
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt   # 画图用的，类似MATLAB的plot
import numpy as np          # 科学计算库，处理数组/矩阵
import os                   # 操作系统接口，处理文件路径
from PIL import Image       # Python图像处理库，打开图片用的

# ==================== 第一部分：环境测试 ====================
def test_pytorch_env():
    print("环境测试")
    # 1. 查看PyTorch版本
    print(f"\n1️⃣ PyTorch版本: {torch.__version__}")
    print(f"   torchvision版本: {torchvision.__version__}")
    
    # 2. 检测是否有GPU可用（这决定了你训练模型是10分钟还是10小时）
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"2️⃣ ✅ 检测到GPU: {gpu_name}")
        print(f"   训练时可以使用GPU加速！")
    else:
        device = "cpu"
        print("2️⃣ ⚠️ 未检测到GPU，将使用CPU训练（会慢一些，但不影响学习）")
    
    # 3. 尝试加载一个预训练模型（这是迁移学习的核心能力）
    try:
        # ResNet18是一个经典的卷积神经网络，有1800万参数                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        # weights='DEFAULT' 意思是下载在ImageNet上训练好的权重
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        print("3️⃣ ✅ 成功加载ResNet18预训练模型")
        print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"3️⃣ ❌ 模型加载失败: {e}")
        print("   请检查网络连接，可能需要下载模型文件")
        return False
    
    print("\n🎉 环境测试全部通过！你的电脑已经准备好做计算机视觉了！")
    return True

def explore_rice_dataset(data_root=r"D:\python_work\cs\raw"):
    
    train_path = os.path.join(data_root, "train")
    # 1. 检查文件夹是否存在
    if not os.path.exists(train_path):
        print(f"❌ 错误：找不到路径 {train_path}")
        print("   请确认你的数据放在 D:\\python_work\\cs\\raw\\train 下")
        return
    
    # 2. 统计每个类别的图片数量
    print("\n📊 类别统计：")
    class_counts = {}
    total_images = 0
    for class_name in os.listdir(train_path): 
            class_dir = os.path.join(train_path, class_name)
            n_images = len([f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')])
            class_counts[class_name] = n_images
            total_images += n_images
            print(f"   {class_name}: {n_images} 张")
    
    print(f"\n📌 总图片数: {total_images}")
    print(f"📌 类别总数: {len(class_counts)}")
    
    # 3. 用ImageFolder加载数据集（这是PyTorch的标准做法）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # 把图片统一缩放到224x224（经典CNN输入尺寸）
        transforms.ToTensor()            # 把PIL图片转为PyTorch张量，像素值从0-255缩放到0-1
    ])
    
    dataset = datasets.ImageFolder(root=train_path, transform=transform)
    
    print(f"\n✅ ImageFolder加载成功")
    print(f"   类别映射关系: {dataset.class_to_idx}")
    print(f"   （例如：'Blast' 对应标签 {dataset.class_to_idx['Blast']}）")
    
    # 4. 可视化：每个类别显示一张样图
    print("\n📷 正在生成样图展示...")
    
    class_names = dataset.classes
    n_classes = len(class_names)
    
    # 动态计算画布大小（每行5张图）
    n_cols = 5
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()
    
    # 设置中文字体（防止中文乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 遍历数据集，找到每个类别的第一张图并显示
    shown_labels = set()
    for img_tensor, label in dataset:
        if label not in shown_labels:
            shown_labels.add(label)
            ax = axes[label]
            
            # img_tensor的形状是(C, H, W)，matplotlib需要(H, W, C)
            # permute(1,2,0) 就是把通道维移到最后
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            ax.imshow(img_np)
            ax.set_title(f"{class_names[label]}\n({class_counts[class_names[label]]}张)", fontsize=10)
            ax.axis('off')
            
        if len(shown_labels) == n_classes:
            break
    
    # 隐藏多余的子图
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("水稻病害数据集 - 各类别样图", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存图片到results文件夹（以后写论文用）
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/class_samples.png", dpi=150, bbox_inches='tight')
    print("✅ 样图已保存到 results/class_samples.png")
    plt.show()
    
    return dataset, class_names

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 第一步：测试环境
    if not test_pytorch_env():
        print("\n❌ 环境测试未通过，请先解决报错再继续。")
        exit(1)
    
    # 第二步：探索数据
    dataset, class_names = explore_rice_dataset()
    