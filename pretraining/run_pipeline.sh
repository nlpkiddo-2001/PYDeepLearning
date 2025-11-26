#!/bin/bash
set -e # Exit on error

# Configuration
DATA_ROOT="./data"
CHECKPOINT_ROOT="./checkpoints"
TOKENIZER_PATH="$DATA_ROOT/tokenizer.json"
NUM_GPUS=8

# Ensure directories exist
mkdir -p "$CHECKPOINT_ROOT"

echo "=================================================================="
echo "STEP 1: DATA PACKING"
echo "=================================================================="

# 1. Pretraining Data
echo "Packing Pretraining Data..."
python src/data_packing.py \
    --input "$DATA_ROOT/pretraining" \
    --output_dir "$DATA_ROOT/pretraining_bin" \
    --tokenizer_path "$TOKENIZER_PATH"

# 2. Midtraining Data
echo "Packing Midtraining Data..."
python src/data_packing.py \
    --input "$DATA_ROOT/midtraining" \
    --output_dir "$DATA_ROOT/midtraining_bin" \
    --tokenizer_path "$TOKENIZER_PATH"

# 3. Finetuning Data
echo "Packing Finetuning Data..."
# Handles single file or directory
python src/data_packing.py \
    --input "$DATA_ROOT/finetuning" \
    --output_dir "$DATA_ROOT/finetuning_bin" \
    --tokenizer_path "$TOKENIZER_PATH"


echo "=================================================================="
echo "STEP 2: PRE-TRAINING"
echo "=================================================================="
torchrun --nproc_per_node=$NUM_GPUS src/main.py \
    --stage "pretrain" \
    --data_dir "$DATA_ROOT/pretraining_bin" \
    --output_dir "$CHECKPOINT_ROOT/pretrain" \
    --batch_size 32 \
    --grad_accum 2 \
    --context_window 4096 \
    --learning_rate 4e-4 \
    --max_steps 50000 \
    --save_every 1000 \
    --use_muon

echo "=================================================================="
echo "STEP 3: MID-TRAINING"
echo "=================================================================="
# Resume from best pretrain checkpoint
PRETRAIN_CKPT="$CHECKPOINT_ROOT/pretrain/step_50000.pt"
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "Warning: Pretrain checkpoint not found at $PRETRAIN_CKPT. Using latest if available."
    PRETRAIN_CKPT="$CHECKPOINT_ROOT/pretrain/latest.pt"
fi

torchrun --nproc_per_node=$NUM_GPUS src/main.py \
    --stage "midtrain" \
    --data_dir "$DATA_ROOT/midtraining_bin" \
    --output_dir "$CHECKPOINT_ROOT/midtrain" \
    --resume_from "$PRETRAIN_CKPT" \
    --batch_size 32 \
    --grad_accum 2 \
    --context_window 4096 \
    --learning_rate 4e-5 \
    --max_steps 5000 \
    --save_every 500 \
    --use_muon

echo "=================================================================="
echo "STEP 4: FINE-TUNING"
echo "=================================================================="
MIDTRAIN_CKPT="$CHECKPOINT_ROOT/midtrain/step_5000.pt"
if [ ! -f "$MIDTRAIN_CKPT" ]; then
    MIDTRAIN_CKPT="$CHECKPOINT_ROOT/midtrain/latest.pt"
fi

torchrun --nproc_per_node=$NUM_GPUS src/main.py \
    --stage "finetune" \
    --data_dir "$DATA_ROOT/finetuning_bin" \
    --output_dir "$CHECKPOINT_ROOT/finetune" \
    --resume_from "$MIDTRAIN_CKPT" \
    --batch_size 16 \
    --grad_accum 4 \
    --context_window 2048 \
    --learning_rate 1e-5 \
    --max_steps 1000 \
    --save_every 100 \
    --use_muon

echo "=================================================================="
echo "STEP 5: EXPORT TO HUGGING FACE"
echo "=================================================================="
FINETUNE_CKPT="$CHECKPOINT_ROOT/finetune/final_model.pt"
if [ ! -f "$FINETUNE_CKPT" ]; then
    FINETUNE_CKPT="$CHECKPOINT_ROOT/finetune/latest.pt"
fi

python scripts/convert_to_hf.py \
    --checkpoint "$FINETUNE_CKPT" \
    --output_dir "$CHECKPOINT_ROOT/***REMOVED***model"

echo "PIPELINE COMPLETE!"
