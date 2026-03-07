#!/bin/bash

# ===============================================================
# Fairness-aware Federated Learning Experiments on Adult Dataset
# ===============================================================
# This script runs all baseline algorithms:
#   1. FedAvg - Standard Federated Averaging
#   2. FedALA - Adaptive Local Aggregation for Personalization
#   3. FairFed - Fairness-aware Federated Learning
#   4. FedALAFair - Combined Personalization + Fairness
# ===============================================================

# Experiment Configuration
DATASET="Adult"
MODEL="AdultMLP"  # Specialized MLP for Adult dataset
NUM_CLASSES=2
NUM_CLIENTS=20
BATCH_SIZE=32
LOCAL_EPOCHS=5
GLOBAL_ROUNDS=100
LEARNING_RATE=0.01
JOIN_RATIO=1.0
EVAL_GAP=5
DEVICE="cuda"
DEVICE_ID="0"

# Fairness parameters
FAIRNESS_LAMBDA=0.1  # Weight for fairness loss
SENSITIVE_ATTR_IDX=12  # Index of 'sex' attribute in Adult dataset features

# ALA parameters (for FedALA and FedALAFair)
ETA=1.0
RAND_PERCENT=80
LAYER_IDX=2

# Result directory
GOAL="fairness_experiment"
TIMES=3  # Run each experiment 3 times for statistical significance

echo "=========================================="
echo "Fairness-aware FL Experiments - Adult Dataset"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Number of Clients: $NUM_CLIENTS"
echo "Global Rounds: $GLOBAL_ROUNDS"
echo "Fairness Lambda: $FAIRNESS_LAMBDA"
echo "=========================================="
echo ""

cd system

# ===============================================================
# Experiment 1: FedAvg (Baseline)
# ===============================================================
echo ""
echo "=========================================="
echo "Running Experiment 1/4: FedAvg"
echo "=========================================="
python main.py \
    -data $DATASET \
    -m $MODEL \
    -algo FedAvg \
    -gr $GLOBAL_ROUNDS \
    -did $DEVICE_ID \
    -go $GOAL \
    -dev $DEVICE \
    -nc $NUM_CLIENTS \
    -jr $JOIN_RATIO \
    -ls $LOCAL_EPOCHS \
    -lbs $BATCH_SIZE \
    -lr $LEARNING_RATE \
    -ncl $NUM_CLASSES \
    -t $TIMES \
    -eg $EVAL_GAP

echo "FedAvg completed!"
echo ""

# ===============================================================
# Experiment 2: FedALA (Personalization)
# ===============================================================
echo ""
echo "=========================================="
echo "Running Experiment 2/4: FedALA"
echo "=========================================="
python main.py \
    -data $DATASET \
    -m $MODEL \
    -algo FedALA \
    -gr $GLOBAL_ROUNDS \
    -did $DEVICE_ID \
    -go $GOAL \
    -dev $DEVICE \
    -nc $NUM_CLIENTS \
    -jr $JOIN_RATIO \
    -ls $LOCAL_EPOCHS \
    -lbs $BATCH_SIZE \
    -lr $LEARNING_RATE \
    -ncl $NUM_CLASSES \
    -t $TIMES \
    -eg $EVAL_GAP \
    -et $ETA \
    -s $RAND_PERCENT \
    -p $LAYER_IDX

echo "FedALA completed!"
echo ""

# ===============================================================
# Experiment 3: FairFed (Fairness)
# ===============================================================
echo ""
echo "=========================================="
echo "Running Experiment 3/4: FairFed"
echo "=========================================="
python main.py \
    -data $DATASET \
    -m $MODEL \
    -algo FairFed \
    -gr $GLOBAL_ROUNDS \
    -did $DEVICE_ID \
    -go $GOAL \
    -dev $DEVICE \
    -nc $NUM_CLIENTS \
    -jr $JOIN_RATIO \
    -ls $LOCAL_EPOCHS \
    -lbs $BATCH_SIZE \
    -lr $LEARNING_RATE \
    -ncl $NUM_CLASSES \
    -t $TIMES \
    -eg $EVAL_GAP \
    -fl $FAIRNESS_LAMBDA \
    -sai $SENSITIVE_ATTR_IDX

echo "FairFed completed!"
echo ""

# ===============================================================
# Experiment 4: FedALAFair (Personalization + Fairness)
# ===============================================================
echo ""
echo "=========================================="
echo "Running Experiment 4/4: FedALAFair"
echo "=========================================="
python main.py \
    -data $DATASET \
    -m $MODEL \
    -algo FedALAFair \
    -gr $GLOBAL_ROUNDS \
    -did $DEVICE_ID \
    -go $GOAL \
    -dev $DEVICE \
    -nc $NUM_CLIENTS \
    -jr $JOIN_RATIO \
    -ls $LOCAL_EPOCHS \
    -lbs $BATCH_SIZE \
    -lr $LEARNING_RATE \
    -ncl $NUM_CLASSES \
    -t $TIMES \
    -eg $EVAL_GAP \
    -fl $FAIRNESS_LAMBDA \
    -sai $SENSITIVE_ATTR_IDX \
    -et $ETA \
    -s $RAND_PERCENT \
    -p $LAYER_IDX

echo "FedALAFair completed!"
echo ""

# ===============================================================
# Summary
# ===============================================================
echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo ""
echo "Results are saved in: system/results/"
echo ""
echo "To analyze results, run:"
echo "  python ../analyze_results.py"
echo ""
echo "Key files:"
echo "  - *_test_acc.h5: Test accuracy over rounds"
echo "  - *_train_acc.h5: Training accuracy over rounds"
echo "  - *_train_loss.h5: Training loss over rounds"
echo ""
echo "Next steps:"
echo "  1. Check results in system/results/"
echo "  2. Run analyze_results.py for visualization"
echo "  3. Compare fairness metrics across algorithms"
echo ""
