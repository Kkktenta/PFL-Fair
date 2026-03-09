#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 用 Adult 数据集跑 FedAvg 实验
# 运行方式：在项目根目录执行  bash run_fedavg_adult.sh
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

# ── 超参数（与 run_fairfed_adult.sh 保持一致）────────────────────────────────
DATASET="Adult"
NUM_CLIENTS=20
NUM_CLASSES=2
MODEL="DNN"
GLOBAL_ROUNDS=100
LOCAL_EPOCHS=5
BATCH_SIZE=32
LR=0.001
JOIN_RATIO=1.0

python main.py \
    -data  "$DATASET"       \
    -nc    "$NUM_CLIENTS"   \
    -ncl   "$NUM_CLASSES"   \
    -m     "$MODEL"         \
    -algo  "FedAvg"         \
    -gr    "$GLOBAL_ROUNDS" \
    -ls    "$LOCAL_EPOCHS"  \
    -lbs   "$BATCH_SIZE"    \
    -lr    "$LR"            \
    -jr    "$JOIN_RATIO"    \
    -dev   "$DEVICE"        \
    -did   "$DEVICE_ID"     \
    -eg    5                \
    -t     1                \
    -go    "fedavg_adult_run1"
