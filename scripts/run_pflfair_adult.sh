#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 用 Adult 数据集跑 PFL-Fair 实验（FairFed + FedALA）
# 运行方式：bash scripts/run_pflfair_adult.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../system"

# ── 设备检测 ──────────────────────────────────────────────────────────────────
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    DEVICE_ID="0"
else
    DEVICE="cpu"
    DEVICE_ID="0"
fi
echo "Using device: $DEVICE"

# ── 公共超参数（与其他算法保持一致）─────────────────────────────────────────
DATASET="Adult"
NUM_CLIENTS=20
NUM_CLASSES=2
MODEL="DNN"
GLOBAL_ROUNDS=100
LOCAL_EPOCHS=5
BATCH_SIZE=32
LR=0.001
JOIN_RATIO=1.0
SENSITIVE_ATTR_IDX=12  # Adult 数据集 sex 列索引

# ── FairFed 专属参数 ───────────────────────────────────────────────────────────
FAIRNESS_LAMBDA=0.1    # β：公平性权重调整步长

# ── FedALA 专属参数 ───────────────────────────────────────────────────────────
ETA=1.0          # 自适应本地聚合的学习率缩放因子
RAND_PERCENT=80  # ALA 随机采样百分比
LAYER_IDX=2      # 参与自适应聚合的层编号（从输出层往前数）

python3 main.py \
    -data  "$DATASET"           \
    -nc    "$NUM_CLIENTS"       \
    -ncl   "$NUM_CLASSES"       \
    -m     "$MODEL"             \
    -algo  "PFL-Fair"           \
    -gr    "$GLOBAL_ROUNDS"     \
    -ls    "$LOCAL_EPOCHS"      \
    -lbs   "$BATCH_SIZE"        \
    -lr    "$LR"                \
    -jr    "$JOIN_RATIO"        \
    -dev   "$DEVICE"            \
    -did   "$DEVICE_ID"         \
    -fl    "$FAIRNESS_LAMBDA"   \
    -sai   "$SENSITIVE_ATTR_IDX"\
    -et    "$ETA"               \
    -s     "$RAND_PERCENT"      \
    -p     "$LAYER_IDX"         \
    -eg    5                    \
    -t     1                    \
    -go    "pflfair_adult_run1"
