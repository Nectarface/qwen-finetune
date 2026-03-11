#!/bin/bash
#
# QWEN 3.5 9B FULL FINE-TUNING
# Run on fresh RunPod pod (no persistent storage)
#

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           QWEN 3.5 9B FULL FINE-TUNING                        ║"
echo "║              Personal Knowledge Training                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

BASE="/workspace/qwen_training"
mkdir -p $BASE
cd $BASE

# Copy files if running from repo
if [ -f "$(dirname $0)/train_qwen.py" ]; then
    cp "$(dirname $0)"/*.py "$BASE/" 2>/dev/null || true
    cp "$(dirname $0)"/*.json "$BASE/" 2>/dev/null || true
    cp "$(dirname $0)"/*.txt "$BASE/" 2>/dev/null || true
    cp "$(dirname $0)"/*.jsonl "$BASE/" 2>/dev/null || true
fi

mkdir -p output logs

# ============================================
# STEP 1: Dependencies
# ============================================
echo "[1/4] Installing dependencies..."
pip install -q transformers datasets accelerate deepspeed torch safetensors huggingface-hub tqdm scipy bitsandbytes

# Try to install flash-attn (may fail on some systems)
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention not available, using default"

# ============================================
# STEP 2: GPU Check
# ============================================
echo ""
echo "[2/4] GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "     -> $NUM_GPUS GPUs available"

# ============================================
# STEP 3: Check Dataset
# ============================================
echo ""
echo "[3/4] Checking dataset..."

DATASET="$BASE/final_training_file.jsonl"

# Decompress if only gzipped version exists
if [ ! -f "$DATASET" ] && [ -f "$DATASET.gz" ]; then
    echo "     -> Decompressing training data..."
    gunzip -k "$DATASET.gz"
fi

if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo ""
    echo "Please upload your training file:"
    echo "  final_training_file.jsonl or final_training_file.jsonl.gz"
    echo ""
    echo "Then run: bash GO.sh"
    exit 1
fi

LINES=$(wc -l < "$DATASET")
SIZE=$(du -h "$DATASET" | cut -f1)
echo "     -> Found: $DATASET"
echo "     -> Examples: $LINES"
echo "     -> Size: $SIZE"

# ============================================
# STEP 4: Run Training
# ============================================
echo ""
echo "[4/4] Starting full fine-tuning on $NUM_GPUS GPUs..."
echo ""

# Use DeepSpeed for distributed training
deepspeed --num_gpus=$NUM_GPUS train_qwen.py \
    --dataset "$DATASET" \
    --output "$BASE/output/qwen3.5-9b-finetuned" \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 2e-5 \
    --max_length 2048

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                 FINE-TUNING COMPLETE!                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Model saved to: $BASE/output/qwen3.5-9b-finetuned"
echo ""
ls -la $BASE/output/
echo ""
