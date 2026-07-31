# 农作物病害图像分类 —— 基于迁移学习的小样本识别研究

毕业设计项目：使用 ImageNet 预训练的深度卷积网络，通过不同的迁移学习微调策略，在水稻、玫瑰、番茄三个作物病害数据集上进行图像分类，并系统评估小样本（few-shot）条件下的识别性能。

## 研究内容

- **数据集**：水稻 Rice（10 类）、玫瑰 Rose（6 类）、番茄 Tomato（9 类）三个独立的作物病害数据集
- **模型**：ResNet18、ResNet50、EfficientNetV2-S（均为 ImageNet 预训练权重）
- **微调策略**：
  - `freeze_all`：冻结全部特征提取层，仅训练最后的分类层
  - `fine_tune_last`：解冻最后一个特征块（如 ResNet 的 `layer4`）+ 分类层
  - `full_fine_tune`：全参数微调
- **小样本实验**：在每个数据集上分别构造 5-shot、10-shot、20% 三种规模的训练子集
- **基线对比**：ResNet18 在相同数据规模下从零开始训练（baseline）

核心问题：在标注样本稀缺时，不同迁移策略相对从头训练能带来多大提升，以及哪种策略在何种数据规模下最优。

## 项目结构

```
cs/
├── main.py                    # 环境测试 + 数据集探索（各类别样图、统计）
├── run_all.py                 # 统一实验入口（定义全部实验矩阵，需取消注释才真正运行）
├── run_base_small.py          # 小样本 baseline（从头训练）实验脚本
├── run_new_models.py          # 新模型（EfficientNetV2-S 等）实验脚本
├── run_new_small_all.py       # 小样本 + 全模型组合实验脚本
├── small_vs_full.py           # 小样本 vs 全量数据对比分析
├── predict.py                 # 预测脚本：自动扫描 results/ 选择全局最优模型
├── visualize_all.py           # 生成收敛曲线、精度对比图
├── plot_small_train.py        # 小样本训练过程可视化
├── exporl_data.py             # 数据探索辅助脚本
├── biaoge.py                  # 实验表格生成脚本
├── data_process/
│   ├── dataloader.py          # DataLoader 与数据增强 transforms
│   └── build_small_dataset.py # 从全量数据构造 5-shot/10-shot/20% 子集
├── train/
│   ├── train_migration.py     # 迁移学习训练主逻辑（含三种策略）
│   └── train_baseline.py      # 从头训练 baseline 逻辑
├── models/
│   └── model_utils.py         # 加载预训练模型 + 三种微调策略 + 参数量统计
└── results/                   # 实验结果（精度表 CSV、收敛曲线、对比图、论文插图）
```

## 环境与安装

- Python ≥ 3.9
- PyTorch ≥ 2.1（带 CUDA 版本，建议按本机环境从 [pytorch.org](https://pytorch.org) 安装）

```bash
pip install -r requirements.txt
```

> torch / torchvision 请使用 PyTorch 官方命令安装（CUDA 版本以本机为准），普通 `pip install torch` 默认安装 CPU 版。例如 CUDA 12.6：
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```

## 数据准备

原始数据集**不包含在本仓库**（GitHub 单文件 100MB 限制 + 数据集版权），需自行下载并整理为如下结构：

```
raw/                      # 或 raw_rose/、raw_tomato/
├── train/
│   ├── class_1/
│   │   ├── xxx.jpg
│   │   └── ...
│   └── class_2/
└── val/
    ├── class_1/
    └── class_2/
```

各数据集的类别名见 [predict.py](predict.py) 中的 `CLASS_NAMES`。小样本子集可由 [build_small_dataset.py](data_process/build_small_dataset.py) 从全量数据自动生成。

## 使用方法

```bash
# 1. 环境测试 + 数据探索
python main.py

# 2. 训练（run_all.py 默认只展示实验结构，需取消注释 main() 后执行）
#    python run_all.py

# 3. 单张图片预测（自动选择全局最优模型）
python predict.py --dataset rice --image path/to/image.jpg

# 4. 批量预测文件夹
python predict.py --dataset tomato --folder path/to/folder

# 5. 生成结果图表
python visualize_all.py
```

## 实验结果

**各数据集最优精度**（三种模型 × 三种策略中的最优验证准确率，baseline 为 ResNet18 从头训练）：

| 数据集 | 类别数 | 5-shot | 10-shot | 20% | 全量 |
|--------|:------:|:------:|:-------:|:---:|:----:|
| 水稻 Rice | 10 | 29.0% | 38.0% | 81.2% | **95.1%** |
| 玫瑰 Rose | 6  | 68.1% | 78.8% | 94.2% | **98.2%** |
| 番茄 Tomato | 9 | 62.9% | 71.7% | 87.7% | **96.8%** |

**主要结论**（基于 `results/` 中的完整精度表）：

1. **迁移学习优势显著**：全量数据下 ResNet50 `fine_tune_last` 在三个数据集上均达到最优（水稻 95.1% / 玫瑰 98.2% / 番茄 96.8%），相比从头训练的 baseline（水稻全量约 69.9%）提升约 25 个百分点。
2. **`fine_tune_last` 策略整体最优**：在大多数数据规模下优于全量冻结与全参数微调，尤其在小样本场景。
3. **ResNet50 表现最稳定**：三个数据集、各个数据规模下均保持竞争力。
4. **数据规模是关键瓶颈**：5-shot 条件下即便最优模型精度也偏低（水稻仅 29.0%），说明极少量样本下病害识别的挑战性；20% 以上数据时精度即可接近全量水平。

完整实验数据见 `results/tables/`，可视化图表见 `results/`。

## 说明

- 各脚本内存在针对作者机器的绝对路径（如 `D:\python_work\cs`），clone 到其他机器后需按本机路径调整。
- 模型权重文件（`models/saved*/`，约 10GB）与数据集（`raw*/`、`data_small*/`）体积过大且不入库，见 `.gitignore`。
