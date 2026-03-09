#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 用 Adult 数据集跑 FairFed 实验
# 运行方式：在项目根目录执行  bash run_fairfed_adult.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/system"

# ── 设备检测 ──────────────────────────────────────────────────────────────────
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    DEVICE_ID="0"
else
    DEVICE="cpu"
    DEVICE_ID="0"
fi
echo "Using device: $DEVICE"

# ── 超参数 ────────────────────────────────────────────────────────────────────
DATASET="Adult"
NUM_CLIENTS=20
NUM_CLASSES=2
MODEL="DNN"         # DNN(14, 64, 2)；也可改成 AdultMLP
GLOBAL_ROUNDS=100
LOCAL_EPOCHS=5
BATCH_SIZE=32
LR=0.001
JOIN_RATIO=1.0      # 全部客户端参与

# FairFed 专属参数
FAIRNESS_LAMBDA=0.1   # β：公平性权重调整步长
SENSITIVE_ATTR_IDX=12 # Adult 数据集 sex 列在特征矩阵中的索引

python main.py \
    -data  "$DATASET"           \
    -nc    "$NUM_CLIENTS"       \
    -ncl   "$NUM_CLASSES"       \
    -m     "$MODEL"             \
    -algo  "FairFed"            \
    -gr    "$GLOBAL_ROUNDS"     \
    -ls    "$LOCAL_EPOCHS"      \
    -lbs   "$BATCH_SIZE"        \
    -lr    "$LR"                \
    -jr    "$JOIN_RATIO"        \
    -dev   "$DEVICE"            \
    -did   "$DEVICE_ID"         \
    -fl    "$FAIRNESS_LAMBDA"   \
    -sai   "$SENSITIVE_ATTR_IDX"\
    -eg    5                    \
    -t     1                    \
    -go    "fairfed_adult_run1"
