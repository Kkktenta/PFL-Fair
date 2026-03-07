# 基于个性化联邦学习的公平性模型实验指南

本指南提供完整的步骤说明，帮助您复现"基于个性化联邦学习的公平性模型"实验。

---

## 📋 实验概述

### 研究目标
实现并比较四个联邦学习算法在Adult数据集上的性能和公平性表现。

### 数据集
- **Adult (UCI成人收入数据集)**
  - 二分类任务：预测个人年收入是否超过$50K
  - 包含敏感属性（性别、种族等）用于公平性评估
  - 14个特征，2个类别
  - 适合研究联邦学习中的公平性问题

### 基线算法

1. **FedAvg** - 标准联邦平均算法（基线）
2. **FedALA** - 自适应局部聚合（个性化）
3. **FairFed** - 公平性感知联邦学习
4. **FedALAFair** - 结合个性化和公平性的新模型（本研究提出）

---

## 🚀 快速开始

### 环境要求

```bash
# Python >= 3.8
# PyTorch >= 1.9.0
# CUDA (可选，用于GPU加速)

# 安装依赖
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install h5py matplotlib seaborn
```

### 步骤 1: 生成 Adult 数据集

```bash
# 切换到项目根目录
cd /Users/juiklo/Workspace/Graduation_Thesis_Code/PFLlib

# 给脚本添加执行权限
chmod +x generate_adult_data.sh

# 运行数据集生成脚本
./generate_adult_data.sh
```

**预期输出：**
- 数据集将被下载到 `dataset/Adult/rawdata/`
- 处理后的数据分配到 `dataset/Adult/train/` 和 `dataset/Adult/test/`
- 配置文件保存在 `dataset/Adult/config.json`

**说明：**
- 数据集将被分配给20个客户端
- 使用Dirichlet分布创建non-IID数据划分
- 模拟真实的联邦学习场景

---

### 步骤 2: 运行实验

```bash
# 给实验脚本添加执行权限
chmod +x run_experiments.sh

# 运行所有四个基线算法
./run_experiments.sh
```

**实验配置：**
- 全局轮数：100轮
- 本地训练轮数：5轮
- 批大小：32
- 学习率：0.01
- 客户端数量：20
- 每个实验重复3次

**预期时间：**
- 每个算法约15-30分钟（取决于硬件）
- 总计约1-2小时完成所有实验

**实验进度监控：**
脚本会显示每个算法的训练进度，包括：
- 当前轮数
- 训练/测试准确率
- 训练损失
- 公平性指标（FairFed和FedALAFair）
- 每轮耗时

---

### 步骤 3: 分析结果

```bash
# 运行结果分析脚本
python analyze_results.py
```

**生成的可视化内容：**

1. **test_acc_comparison.png** - 测试准确率随训练轮数的变化
2. **train_acc_comparison.png** - 训练准确率随训练轮数的变化
3. **train_loss_comparison.png** - 训练损失随训练轮数的变化
4. **final_comparison.png** - 最终性能对比柱状图

**统计摘要：**
- 每个算法的平均性能
- 标准差（多次运行的稳定性）
- 最大性能
- 最终轮性能

所有结果保存在 `analysis_results/` 目录。

---

## 📊 结果解读

### 性能指标

1. **测试准确率 (Test Accuracy)**
   - 模型泛化能力指标
   - 越高越好
   - 预期：FedALA和FedALAFair应该表现更好（个性化优势）

2. **公平性指标 (Demographic Parity)**
   - 衡量模型对不同群体的公平性
   - 越接近0越公平
   - 预期：FairFed和FedALAFair应该表现更好（公平性约束）

### 预期结果趋势

```
准确率排序（预期）：
FedALAFair ≥ FedALA > FairFed ≥ FedAvg

公平性排序（预期）：
FairFed ≥ FedALAFair > FedAvg ≥ FedALA

综合表现（准确率 + 公平性）：
FedALAFair（最佳平衡） > FairFed > FedALA > FedAvg
```

### 关键发现点

1. **FedAvg vs FedALA**
   - 对比：个性化的影响
   - FedALA应该在non-IID数据上表现更好

2. **FedAvg vs FairFed**
   - 对比：公平性约束的影响
   - FairFed的准确率可能略低，但公平性更好

3. **FedALA vs FedALAFair**
   - 对比：在个性化基础上加入公平性约束
   - FedALAFair应该在准确率和公平性之间取得最佳平衡

---

## 🔧 高级配置

### 调整实验参数

编辑 `run_experiments.sh` 文件，可以修改以下参数：

```bash
# 基础参数
NUM_CLIENTS=20          # 客户端数量
BATCH_SIZE=32           # 批大小
LOCAL_EPOCHS=5          # 本地训练轮数
GLOBAL_ROUNDS=100       # 全局通信轮数
LEARNING_RATE=0.01      # 学习率

# 公平性参数
FAIRNESS_LAMBDA=0.1     # 公平性损失权重（0.05-0.5推荐范围）
SENSITIVE_ATTR_IDX=12   # 敏感属性索引（性别列）

# ALA参数
ETA=1.0                 # ALA学习率
RAND_PERCENT=80         # 随机采样百分比
LAYER_IDX=2             # ALA层索引
```

### 单独运行某个算法

```bash
cd system

# 示例：只运行FedALAFair
python main.py \
    -data Adult \
    -m AdultMLP \
    -algo FedALAFair \
    -gr 100 \
    -did 0 \
    -go my_experiment \
    -dev cuda \
    -nc 20 \
    -jr 1.0 \
    -ls 5 \
    -lbs 32 \
    -lr 0.01 \
    -ncl 2 \
    -t 1 \
    -eg 5 \
    -fl 0.1 \
    -sai 12 \
    -et 1.0 \
    -s 80 \
    -p 2
```

---

## 📂 项目结构

```
PFLlib/
├── dataset/
│   ├── generate_Adult.py           # Adult数据集生成脚本
│   └── Adult/                       # 生成的数据集
│       ├── train/                   # 训练数据
│       ├── test/                    # 测试数据
│       └── config.json              # 配置文件
├── system/
│   ├── main.py                      # 主程序（已修改支持新算法）
│   ├── flcore/
│   │   ├── clients/
│   │   │   ├── clientfairfed.py    # FairFed客户端
│   │   │   └── clientalafair.py    # FedALAFair客户端
│   │   ├── servers/
│   │   │   ├── serverfairfed.py    # FairFed服务器
│   │   │   └── serveralafair.py    # FedALAFair服务器
│   │   └── trainmodel/
│   │       └── models.py            # 模型定义（包含AdultMLP）
│   └── results/                     # 实验结果（自动生成）
├── generate_adult_data.sh           # 数据集生成脚本
├── run_experiments.sh               # 实验运行脚本
├── analyze_results.py               # 结果分析脚本
├── analysis_results/                # 分析结果（自动生成）
└── FAIRNESS_EXPERIMENT_GUIDE.md     # 本指南
```

---

## 🐛 常见问题

### 1. 数据集下载失败

**问题：** wget下载Adult数据集时失败

**解决方案：**
```bash
# 手动下载数据集
cd dataset/Adult/rawdata
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

# 或者使用curl
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

### 2. CUDA内存不足

**问题：** GPU内存不足错误

**解决方案：**
```bash
# 方案1：减少批大小
BATCH_SIZE=16  # 在run_experiments.sh中修改

# 方案2：使用CPU
DEVICE="cpu"   # 在run_experiments.sh中修改
```

### 3. 结果文件未找到

**问题：** analyze_results.py报告找不到结果文件

**解决方案：**
- 确保实验已完全运行完成
- 检查 `system/results/` 目录是否有 `.h5` 文件
- 验证文件命名格式：`Adult_[算法]_fairness_experiment_[指标].h5`

### 4. 依赖包问题

**问题：** 导入错误或版本冲突

**解决方案：**
```bash
# 创建新的conda环境
conda create -n pfllib python=3.8
conda activate pfllib

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn h5py matplotlib seaborn ujson
```

---

## 📝 实验检查清单

在提交论文/报告前，请确保：

- [ ] 所有四个算法都成功运行完成
- [ ] 每个算法至少运行3次（用于统计显著性）
- [ ] 生成了所有可视化结果
- [ ] 记录了最终性能指标和公平性指标
- [ ] 分析了算法间的性能差异
- [ ] 讨论了准确率-公平性权衡
- [ ] 验证了FedALAFair的优势

---

## 📚 参考实现

### FedAvg
- 标准实现：`system/flcore/servers/serveravg.py`
- 客户端：`system/flcore/clients/clientavg.py`

### FedALA  
- 服务器：`system/flcore/servers/serverala.py`
- 客户端：`system/flcore/clients/clientala.py`
- ALA核心：`system/utils/ALA.py`

### FairFed
- 服务器：`system/flcore/servers/serverfairfed.py`
- 客户端：`system/flcore/clients/clientfairfed.py`

### FedALAFair (新模型)
- 服务器：`system/flcore/servers/serveralafair.py`
- 客户端：`system/flcore/clients/clientalafair.py`
- 结合了ALA的自适应聚合和FairFed的公平性约束

---

## 🎯 论文撰写建议

### 实验部分应包含：

1. **数据集描述**
   - Adult数据集特征
   - 数据划分策略（non-IID Dirichlet）
   - 敏感属性说明

2. **实验设置**
   - 超参数配置
   - 硬件环境
   - 评估指标

3. **结果与分析**
   - 性能对比表格
   - 准确率曲线图
   - 公平性指标对比
   - 统计显著性检验

4. **消融研究** (可选)
   - 不同fairness_lambda值的影响
   - 不同客户端数量的影响
   - 不同数据异质性程度的影响

---

## 💡 扩展实验建议

1. **超参数调优**
   - 调整fairness_lambda (0.05, 0.1, 0.2, 0.5)
   - 调整学习率
   - 调整本地训练轮数

2. **公平性指标扩展**
   - Equalized Odds
   - Equal Opportunity
   - 按种族分组的公平性

3. **数据异质性分析**
   - 不同alpha值的Dirichlet分布
   - 完全non-IID (pathological non-IID)
   - 标签偏移 vs 特征偏移

4. **可扩展性测试**
   - 增加客户端数量 (50, 100)
   - 更长的训练时间
   - 不同参与率

---

## 🤝 如需帮助

如果在实验过程中遇到问题，可以：

1. 查看PFLlib官方文档：http://www.pfllib.com
2. 检查本指南的"常见问题"部分
3. 查看代码注释和文档字符串
4. 参考已有的算法实现作为模板

---

## ✅ 实验完成标准

实验成功完成的标志：

1. ✅ 数据集成功生成（20个客户端的训练和测试数据）
2. ✅ 四个算法都成功运行100轮
3. ✅ 生成了准确率和损失曲线
4. ✅ FedALAFair的表现验证了设计思路
5. ✅ 可以清晰展示准确率-公平性权衡

---

**祝实验顺利！如有问题，请参考本指南或查看代码注释。**
