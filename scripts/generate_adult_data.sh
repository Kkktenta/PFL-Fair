#!/bin/bash

# ===============================================================
# Adult Dataset Generation Script for Fairness-aware FL
# ===============================================================
# This script generates the Adult dataset with different
# partitioning strategies for federated learning experiments
# ===============================================================

echo "=========================================="
echo "Adult Dataset Generation"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

cd dataset

# Generate Adult dataset with non-IID Dirichlet partition
# This simulates realistic federated learning scenarios
# where different clients have different data distributions
echo "Generating Adult dataset (Non-IID, Dirichlet partition)..."
python generate_Adult.py noniid balance dir

echo ""
echo "=========================================="
echo "Dataset generation completed!"
echo "=========================================="
echo ""
echo "Dataset location: dataset/Adult/"
echo "  - train/: Training data for each client"
echo "  - test/: Test data for each client"
echo "  - config.json: Dataset configuration"
echo ""
echo "You can now run the experiments using run_experiments.sh"
echo ""
