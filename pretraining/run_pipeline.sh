#!/bin/bash
set -e # Exit on error


NUM_GPUS=8

mkdir -p ./checkpoints/pretrain
mkdir -p ./checkpoints/midtrain
mkdir -p ./checkpoints/finetune

echo "=================================================================="
echo "LLM Training Pipeline - YAML Configuration Mode"
echo "=================================================================="
echo "This pipeline will run all three training stages:"
echo "  1. Pre-training (from scratch)"
echo "  2. Mid-training (continued training on domain data)"
echo "  3. Fine-tuning (supervised instruction tuning)"
echo ""
echo "Each stage uses its own YAML config file:"
echo "  - config.pretrain.yaml"
echo "  - config.midtrain.yaml"
echo "  - config.finetune.yaml"
echo "=================================================================="
echo ""

if [ ! -f "config.pretrain.yaml" ]; then
    echo "Error: config.pretrain.yaml not found!"
    exit 1
fi

if [ ! -f "config.midtrain.yaml" ]; then
    echo "Error: config.midtrain.yaml not found!"
    exit 1
fi

if [ ! -f "config.finetune.yaml" ]; then
    echo "Error: config.finetune.yaml not found!"
    exit 1
fi

echo "=================================================================="
echo "STEP 1: PRE-TRAINING"
echo "=================================================================="
echo "Using config: config.pretrain.yaml"
echo "Starting pre-training from scratch..."
echo ""

torchrun --nproc_per_node=$NUM_GPUS src/main.py --config config.pretrain.yaml

echo ""
echo "Pre-training complete!"
echo ""

echo "=================================================================="
echo "STEP 2: MID-TRAINING"
echo "=================================================================="
echo "Using config: config.midtrain.yaml"
echo "Note: Make sure config.midtrain.yaml has the correct 'resume_from' path"
echo ""

LATEST_PRETRAIN=$(ls -t ./checkpoints/pretrain/step_*.pt 2>/dev/null | head -1 || echo "")
if [ -n "$LATEST_PRETRAIN" ]; then
    echo "Found latest pretrain checkpoint: $LATEST_PRETRAIN"
    echo "You may want to update config.midtrain.yaml to use this checkpoint"
else
    echo "Warning: No pretrain checkpoint found. Make sure to set resume_from in config.midtrain.yaml"
fi
echo ""

torchrun --nproc_per_node=$NUM_GPUS src/main.py --config config.midtrain.yaml

echo ""
echo "Mid-training complete!"
echo ""

echo "=================================================================="
echo "STEP 3: FINE-TUNING"
echo "=================================================================="
echo "Using config: config.finetune.yaml"
echo "Note: Make sure config.finetune.yaml has the correct 'resume_from' path"
echo ""

LATEST_MIDTRAIN=$(ls -t ./checkpoints/midtrain/step_*.pt 2>/dev/null | head -1 || echo "")
if [ -n "$LATEST_MIDTRAIN" ]; then
    echo "Found latest midtrain checkpoint: $LATEST_MIDTRAIN"
    echo "You may want to update config.finetune.yaml to use this checkpoint"
else
    echo "Warning: No midtrain checkpoint found. Make sure to set resume_from in config.finetune.yaml"
fi
echo ""

torchrun --nproc_per_node=$NUM_GPUS src/main.py --config config.finetune.yaml

echo ""
echo "Fine-tuning complete!"
echo ""

echo "=================================================================="
echo "STEP 4: EXPORT TO HUGGING FACE (Optional)"
echo "=================================================================="
FINETUNE_CKPT=$(ls -t ./checkpoints/finetune/step_*.pt 2>/dev/null | head -1 || echo "")

if [ -z "$FINETUNE_CKPT" ]; then
    echo "No finetune checkpoint found. Skipping HF export."
else
    echo "Found checkpoint: $FINETUNE_CKPT"
    
    if [ -f "scripts/convert_to_hf.py" ]; then
        echo "Converting to Hugging Face format..."
        python scripts/convert_to_hf.py \
            --checkpoint "$FINETUNE_CKPT" \
            --output_dir "./checkpoints/hf_model"
        echo "Model exported to ./checkpoints/hf_model"
    else
        echo "scripts/convert_to_hf.py not found. Skipping HF export."
    fi
fi

echo ""
echo "=================================================================="
echo "PIPELINE COMPLETE!"
echo "=================================================================="
echo "Checkpoints saved in:"
echo "  - ./checkpoints/pretrain/"
echo "  - ./checkpoints/midtrain/"
echo "  - ./checkpoints/finetune/"
echo "=================================================================="

