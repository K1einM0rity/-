import os
import shutil
import random
from tqdm import tqdm  # 用于显示进度条

# ==================== 第一部分：核心抽取函数 ====================

def create_small_dataset(source_train_dir, target_dir, samples_per_class=None, percent=None):
    """
    从源训练集中抽取部分样本，生成一个小样本数据集。

    参数:
        source_train_dir: 源训练集文件夹路径，该文件夹下每个子文件夹代表一个类别。
        target_dir: 目标小样本数据集存放路径，将按类别创建子文件夹并复制图片。
        samples_per_class: 每个类别复制的图片数量（绝对数量），与 percent 互斥，二选一。
        percent: 每个类别复制的图片比例（0~1之间的浮点数），与 samples_per_class 互斥，二选一。
    
    异常:
        FileNotFoundError: 如果源训练集路径不存在。
        ValueError: 如果 samples_per_class 和 percent 都未指定。
    """
    # 检查源训练集路径是否存在
    if not os.path.exists(source_train_dir):
        raise FileNotFoundError(f"❌ 源训练集路径不存在：{source_train_dir}")
    
    # 获取所有类别名称（文件夹名）
    class_names = [d for d in os.listdir(source_train_dir) 
                   if os.path.isdir(os.path.join(source_train_dir, d))]
    print(f"📊 检测到 {len(class_names)} 个类别：{class_names}")
    
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)
    print(f"📁 目标文件夹已创建：{target_dir}")
    
    total_copied = 0  # 记录总共复制的图片数
    
    # 遍历每个类别，使用 tqdm 显示进度条
    for class_name in tqdm(class_names, desc="抽取进度"):
        src_class_dir = os.path.join(source_train_dir, class_name)
        
        # 列出该类别下所有图片文件（根据常见后缀过滤）
        all_images = [f for f in os.listdir(src_class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        
        # 随机打乱图片顺序（使用固定随机种子以保证可复现）
        random.seed(42)
        random.shuffle(all_images)
        
        # 确定需要抽取的图片数量
        if samples_per_class is not None:
            n_sample = samples_per_class
        elif percent is not None:
            n_sample = int(len(all_images) * percent)
        else:
            raise ValueError("❌ 必须指定 samples_per_class 或 percent 中的一个！")
        
        # 如果源图片总数少于需要抽取的数量，则全部复制，并给出警告
        if len(all_images) < n_sample:
            print(f"⚠️ 警告：类别 {class_name} 仅有 {len(all_images)} 张图，少于需要的 {n_sample} 张，将全部复制。")
            n_sample = len(all_images)
        
        # 选取前 n_sample 张（因为已经打乱，相当于随机选取）
        selected_images = all_images[:n_sample]
        
        # 创建目标类别文件夹
        dst_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        # 逐张复制图片，使用 shutil.copy2 保留元数据
        for img_name in selected_images:
            src_img_path = os.path.join(src_class_dir, img_name)
            dst_img_path = os.path.join(dst_class_dir, img_name)
            shutil.copy2(src_img_path, dst_img_path)
            total_copied += 1
            
    print(f"\n✅ 小样本数据集构建完成！共复制 {total_copied} 张图片到 {target_dir}")


# ==================== 第二部分：批量生成不同规模的数据集 ====================

def build_all_small_datasets(base_raw_path=r"D:\python_work\cs\raw",
                             output_base=r"D:\python_work\cs\data_small"):
    """
    自动生成多个不同规模的小样本训练集，并复制一份完整的验证集。
    
    参数:
        base_raw_path: 原始完整数据集根目录，里面应包含 "train" 和 "val" 子文件夹。
        output_base: 生成的小样本数据集存放根目录。
    """

    # 构建源训练集和源验证集的完整路径
    src_train = os.path.join(base_raw_path, "train")
    src_val = os.path.join(base_raw_path, "val")

    # 检查路径是否存在
    if not os.path.exists(src_train):
        raise FileNotFoundError(f"❌ 源训练集不存在：{src_train}")
    if not os.path.exists(src_val):
        raise FileNotFoundError(f"❌ 源验证集不存在：{src_val}")
    
    # 定义需要生成的小样本数据集配置
    configs = [
        {"name": "train_5shot", "samples_per_class": 5},          # 每个类别 5 张
        {"name": "train_10shot", "samples_per_class": 10},        # 每个类别 10 张
        {"name": "train_20percent", "percent": 0.20},             # 每个类别随机抽取 20%
    ]
    
    # 遍历配置，逐个生成小样本训练集
    for cfg in configs:
        print(f"\n🔄 正在生成：{cfg['name']} ...")
        target_train_dir = os.path.join(output_base, cfg["name"])

        # 根据配置中指定的参数调用 create_small_dataset
        if "samples_per_class" in cfg:
            create_small_dataset(
                source_train_dir=src_train,
                target_dir=target_train_dir,
                samples_per_class=cfg["samples_per_class"]
            )
        else:
            create_small_dataset(
                source_train_dir=src_train,
                target_dir=target_train_dir,
                percent=cfg["percent"]
            )
    
    # === 4. 复制验证集（所有实验共用同一份验证集） ===
    print(f"\n🔄 正在复制验证集到 {output_base}/val ...")
    target_val_dir = os.path.join(output_base, "val")
    
    # 如果目标验证集文件夹已存在，先清空，确保每次生成都是全新的、无残留文件
    if os.path.exists(target_val_dir):
        shutil.rmtree(target_val_dir)
        print(f"   已清空旧的验证集文件夹")
    
    # 递归复制整个源验证集文件夹到目标位置
    shutil.copytree(src_val, target_val_dir)
    print(f"✅ 验证集复制完成：{target_val_dir}")
    
    print("\n" + "=" * 60)
    print("🎉 所有小样本数据集生成完毕！")
    print(f"   数据存放根目录：{output_base}")
    print("=" * 60)


# ==================== 第三部分：测试函数 ====================

def verify_small_dataset(data_root=r"D:\python_work\cs\data_small\train_5shot"):
    """
    验证生成的小样本数据集结构是否正确。
    遍历数据集下的每个类别文件夹，统计并打印每个类别的图片数量及总数。
    
    参数：
        data_root : str
            要检查的数据集路径（类别子文件夹所在目录）。
    """
    print(f"\n📊 验证数据集：{data_root}")
    
    if not os.path.exists(data_root):
        print("❌ 路径不存在！")
        return
    
    # 获取所有类别文件夹
    class_names = [d for d in os.listdir(data_root) 
                   if os.path.isdir(os.path.join(data_root, d))]
    
    total = 0
    # 按类别名字母顺序遍历，保证输出有序
    for cls in sorted(class_names):
        cls_path = os.path.join(data_root, cls)
        # 统计该类别下的图片文件数量
        n_images = len([f for f in os.listdir(cls_path) if f.lower().endswith('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')])
        total += n_images
        print(f"   {cls}: {n_images} 张")
    
    print(f"   总计: {total} 张")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 执行批量生成：生成 5-shot、10-shot、20% 三个小样本训练集并复制验证集
    build_all_small_datasets()
    
    # 验证生成的 5-shot 数据集
    verify_small_dataset(r"D:\python_work\cs\data_small\train_5shot")
    # 验证生成的 20% 数据集
    verify_small_dataset(r"D:\python_work\cs\data_small\train_20percent")
