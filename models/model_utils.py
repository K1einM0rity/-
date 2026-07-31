import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=10, strategy="freeze_all", model_name="resnet18"):
    """
    加载预训练模型并根据指定的微调策略修改最后的分类层。

    参数:
        num_classes: 分类任务的类别数，用于替换原始模型的输出层。
        strategy: 微调策略，可选：
            - "freeze_all":   冻结除最后全连接层（分类器）以外的所有层参数。
            - "fine_tune_last": 仅解冻最后几个残差块（ResNet 的 layer4）和分类层，适合在小数据集上微调少量高层特征。
            - "full_fine_tune":  所有层都可训练，从头微调整个模型。
        model_name: 预训练模型名称，当前支持：
            - "resnet18"
            - "resnet50"
            - "efficientnet_v2_s"

    返回:
        model: 配置好的 PyTorch 模型对象。
    
    异常:
        ValueError: 如果模型名称或策略不受支持。
    """
    # ---- 1. 根据模型名称加载对应的预训练模型，并替换最后的分类层 ----
    if model_name == "resnet18":
        # 使用 ImageNet 默认权重加载 ResNet-18
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # ResNet 的全连接层在 model.fc，获取其输入特征维度
        num_features = model.fc.in_features
        # 替换为新的全连接层，输出 num_classes 个类别
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "resnet50":
        # 加载 ResNet-50
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = model.fc.in_features                                                                                                                                                                                                                                              
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "efficientnet_v2_s":
        # 加载 EfficientNetV2-S
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        # EfficientNet 的分类器是一个 Sequential，最后一层是 [1]，即 nn.Linear
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}。目前仅支持 'resnet18', 'resnet50', 'efficientnet_v2_s'。")

    # ---- 2. 首先将所有参数的 requires_grad 设置为 True（为后续选择性冻结做准备） ----
    for param in model.parameters():
        param.requires_grad = True

    # ---- 3. 根据微调策略设置哪些参数需要梯度更新 ----
    if strategy == "freeze_all":
        # 策略1：冻结全部特征提取层，只训练最后新增的分类层
        for name, param in model.named_parameters():
            # 如果参数名中不包含 "fc"（全连接层）且不包含 "classifier"，则冻结
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
    elif strategy == "fine_tune_last":
        # 策略2：解冻最后一个特征块（如 ResNet 的 layer4）和分类层，其余冻结
        for name, param in model.named_parameters():
            # 只有参数名包含 "layer4"、"fc" 或 "classifier" 的层才训练
            if "layer4" not in name and "fc" not in name and "classifier" not in name:
                param.requires_grad = False
    elif strategy == "full_fine_tune":
        # 策略3：全量微调，什么都不做（因为已经全部设为 True）
        pass
    else:
        raise ValueError(f"不支持的策略: {strategy}。可选: 'freeze_all', 'fine_tune_last', 'full_fine_tune'")

    # ---- 4. 统计可训练参数数量与总参数量，便于观察不同策略的影响 ----
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"✅ 模型配置完成：")
    print(f"   模型：{model_name}")
    print(f"   策略：{strategy}")
    print(f"   类别数：{num_classes}")
    print(f"   可训练参数：{trainable_params:,} / {total_params:,} "
          f"({trainable_params/total_params*100:.1f}%)")

    return model


# ==================== 测试函数 ====================

def test_get_model():
    """
    测试 get_model 函数，分别用三种策略创建模型，并打印可训练参数比例。
    """
    print("=" * 60)
    print("【第四课 - 模型加载与策略测试】")
    print("=" * 60)

    strategies = ["freeze_all", "fine_tune_last", "full_fine_tune"]

    for strat in strategies:
        print(f"\n--- 测试策略: {strat} ---")
        # 使用 ResNet-18 作为测试模型
        model = get_model(num_classes=10, strategy=strat, model_name="resnet18")
        print(f"    模型类型: {type(model).__name__}")

    print("\n🎉 所有策略测试通过！")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    test_get_model()
