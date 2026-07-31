import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

def get_train_transforms():
    """
    构建训练集的数据预处理流水线。
    训练集通常会应用数据增强，以提升模型的泛化能力。
    """
    train_pipeline = transforms.Compose([
        # 随机旋转，角度范围 ±15 度
        transforms.RandomRotation(degrees=15),
        # 以 50% 的概率进行水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        # 随机调整颜色属性：亮度、对比度、饱和度和色调
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # 将图像缩放到 224x224，适配常见的 CNN 输入尺寸（如 ResNet、VGG）
        transforms.Resize(size=(224, 224)),
        # 将 PIL Image 或 numpy.ndarray 转换为 torch.Tensor，并将像素值从 [0,255] 归一化到 [0,1]
        transforms.ToTensor(),
        # 使用 ImageNet 数据集的均值和标准差进行标准化，使数据分布接近标准正态分布
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_pipeline

def get_val_transforms():
    """
    构建验证集的数据预处理流水线。
    验证集不需要数据增强，只需缩放、转张量和标准化，保持与训练集一致的数值范围。
    """
    val_pipeline = transforms.Compose([
        # 缩放到 224x224
        transforms.Resize(size=(224, 224)),   
        # 转为张量，像素值归一化到 [0,1]
        transforms.ToTensor(),                 
        # 标准化，使用与训练集完全相同的均值和标准差
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  
    ])
    
    return val_pipeline



def create_dataloaders(data_root=r"D:\python_work\cs\raw",
                       batch_size=32,
                       val_split=0.2,
                       num_workers=6,
                       train_folder="train"):
    """
    创建训练集和验证集的 DataLoader。

    参数:
        data_root: 数据集根目录，默认指向 D:\python_work\cs\raw
        batch_size: 每个批次加载的样本数
        val_split: 当没有独立的验证集文件夹时，从训练集中划分多少比例作为验证集
        num_workers: 数据加载时使用的子进程数量，提高加载效率
        train_folder: 训练集所在的文件夹名称，默认为 "train"
    
    返回:
        train_loader, val_loader, class_names
    """
    # 构建训练集路径
    train_path = os.path.join(data_root, train_folder)
    # 构建验证集路径（优先使用独立的 "val" 文件夹）
    val_path = os.path.join(data_root, "val")
    
    # 检查训练集路径是否存在，若不存在则抛出异常
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ 训练集路径不存在：{train_path}")
    
    # 获取训练和验证的预处理流程
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # 加载完整的训练数据集（此时包含所有训练样本，之后可能从中切分验证集）
    full_train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    # 获取类别名称列表（按字母顺序排序）
    class_names = full_train_dataset.classes
    
    # 判断是否存在独立的验证集文件夹
    if os.path.exists(val_path):
        # 如果存在 "val" 文件夹，则直接使用它作为验证集
        val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)
        train_dataset = full_train_dataset  # 训练集保持不变
    else:
        # 否则从训练集中按比例切分出一部分作为验证集
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        # 使用随机切分，并固定随机种子（42）以确保每次运行得到相同的划分
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    # 创建训练 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱数据顺序，帮助模型泛化
        num_workers=num_workers,  # 多进程加载
        pin_memory=True if torch.cuda.is_available() else False  # 如果有 GPU，将数据锁定在内存中，加速传输
    )
    
    # 创建验证 DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证时不需要打乱，通常按顺序评估
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 打印创建成功信息，显示批次数量
    print(f"✅ DataLoader 创建成功！训练批次：{len(train_loader)}，验证批次：{len(val_loader)}")
    return train_loader, val_loader, class_names


def test_dataloader():
    """
    测试 DataLoader 是否正常工作。
    打印一个批次的数据形状、标签、以及归一化后的像素统计信息。
    """
    # 使用较小的 batch_size 进行快速测试
    train_loader, val_loader, class_names = create_dataloaders(batch_size=4)
    # 从训练 DataLoader 中获取一个批次的数据
    images, labels = next(iter(train_loader))
    
    print(f"\n📦 一个批次的数据形状：")
    print(f"   images 形状: {images.shape}")   # 期望 [batch_size, 3, 224, 224]
    print(f"   labels 形状: {labels.shape}")
    print(f"   labels 内容: {labels.tolist()}")   # 将标签张量转为列表打印，便于阅读
    
    # 检查归一化后的数值范围（因为做了 Normalize，可能出现负值）
    print(f"\n🎨 归一化后的像素值统计：")
    print(f"   最小值: {images.min().item():.3f}")
    print(f"   最大值: {images.max().item():.3f}")
    print(f"   均值: {images.mean().item():.3f}")
    print(f"   （由于做了 Normalize，数值可能为负数，这是正常的）")
    
    print("\n🎉 DataLoader 测试通过！")



if __name__ == "__main__":
    # 当脚本直接运行时，执行 DataLoader 的测试函数
    test_dataloader()