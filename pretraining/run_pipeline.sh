#!/bin/bash
set -e # Exit on error


NUM_GPUS=8

mkdir -p ./checkpoints/pretrain
mkdir -p ./checkpoints/midtrain
mkdir -p ./checkpoints/finetune


echo "=================================================================="
echo "LLM Training Pipeline - Automated & Configurable"
echo "=================================================================="
echo "This pipeline runs: Pre-training -> Mid-training -> Fine-tuning"
echo ""

# Ensure configs exist
for config in config.pretrain.yaml config.midtrain.yaml config.finetune.yaml; do
    if [ ! -f "$config" ]; then
        echo "Error: $config not found!"
        exit 1
    fi
done

# --- PRE-TRAINING ---
echo "=================================================================="
echo "STEP 1: PRE-TRAINING"
echo "=================================================================="

# Check for pre-training data
if [ ! -d "./data/pretraining" ] || [ -z "$(ls -A ./data/pretraining)" ]; then
    echo "Pre-training data not found. Downloading..."
    # Note: data_pretraining.py seems to be designed for local dry-runs or full downloads.
    # Assuming standard usage without args downloads full set or as configured in script.
    python scripts/data_pretraining.py --output_dir ./data/pretraining
else
    echo "Pre-training data found."
fi

echo "Starting pre-training..."
# Check for existing checkpoint to potentially resume (handled by main.py but good to log)
LATEST_PRETRAIN=$(ls -t ./checkpoints/pretrain/step_*.pt 2>/dev/null | head -1 || echo "")
if [ -n "$LATEST_PRETRAIN" ]; then
    echo "Resuming pre-training from: $LATEST_PRETRAIN"
fi

torchrun --nproc_per_node=$NUM_GPUS src/main.py --config config.pretrain.yaml

echo "Pre-training complete!"
echo ""


# --- MID-TRAINING ---
echo "=================================================================="
echo "STEP 2: MID-TRAINING"
echo "=================================================================="

# Check for mid-training data
# Midtraining data is scattered in subfolders, check specifically for one key folder or if base exists
if [ ! -d "./data/midtraining" ]; then
    echo "Mid-training data not found. Downloading..."
    python src/download_datasets.py --stage midtrain --config config.midtrain.yaml --max_samples 100000
else
    echo "Mid-training data found."
fi

# Resume logic: Midtraining MUST resume from Pretraining result if starting fresh mid-train
# or resume from its own checkpoint if continuing.
# main.py handles 'auto' resume, but we need to ensure the transition from pretrain -> midtrain works.
# If midtrain checkpoint exists, main.py picks it up.
# If NOT, main.py looks for pretrain checkpoint because config.midtrain.yaml has resume_from: "auto".
# We just enforce execution.

echo "Starting mid-training..."
torchrun --nproc_per_node=$NUM_GPUS src/main.py --config config.midtrain.yaml

echo "Mid-training complete!"
echo ""


# --- FINE-TUNING ---
echo "=================================================================="
echo "STEP 3: FINE-TUNING"
echo "=================================================================="

# Check for SFT data
if [ ! -d "./data/sft" ] || [ -z "$(ls -A ./data/sft)" ]; then
    echo "SFT data not found. Downloading..."
    python src/download_datasets.py --stage sft --output_dir ./data
else
    echo "SFT data found."
fi

echo "Starting fine-tuning..."
torchrun --nproc_per_node=$NUM_GPUS src/main.py --config config.finetune.yaml

echo "Fine-tuning complete!"
echo ""


# --- HF EXPORT ---
echo "=================================================================="
echo "STEP 4: EXPORT TO HUGGING FACE (Optional)"
echo "=================================================================="
FINETUNE_CKPT=$(ls -t ./checkpoints/finetune/step_*.pt 2>/dev/null | head -1 || echo "")

if [ -n "$FINETUNE_CKPT" ] && [ -f "scripts/convert_to_hf.py" ]; then
    echo "Exporting model from $FINETUNE_CKPT..."
    python scripts/convert_to_hf.py \
        --checkpoint "$FINETUNE_CKPT" \
        --output_dir "./checkpoints/hf_model"
    echo "Model exported to ./checkpoints/hf_model"
else
    echo "Skipping export (Checkpoint or script missing)."
fi

echo ""
echo "=================================================================="
echo "PIPELINE SUMMARY"
echo "=================================================================="
echo "Completed all stages."


