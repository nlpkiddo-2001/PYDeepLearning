# H100 LLM Training Pipeline

A high-performance, modular Large Language Model (LLM) training pipeline optimized for H100 GPUs. This project supports the full lifecycle of LLM training, including Pre-training, Mid-training (Continual Pre-training), and Fine-tuning.

## Features

- **Multi-Stage Training**: Seamlessly transition between Pre-training, Mid-training, and Fine-tuning stages.
- **Distributed Training**: Built on PyTorch DDP (`DistributedDataParallel`) for efficient multi-GPU scaling.
- **Optimized for H100**: 
    - BF16 Mixed Precision training.
    - `torch.compile` ready (via `torch.set_float32_matmul_precision`).
    - Optimized for high throughput (Tokens/sec tracking).
- **Advanced Optimization**:
    - **Optimizers**: Support for **AdamW** and **Muon** (Momentum Orthogonalized) optimizers.
    - **Schedulers**: Cosine, Linear, and WSD (Warmup-Stable-Decay) learning rate schedules.
    - **Dynamic Batching**: Support for `batch_size_schedule` to increase batch size during training.
- **Robust Checkpointing**:
    - Auto-resume functionality.
    - Safe checkpoint loading (compatible with PyTorch 2.6+ security features).
    - Checkpoint rotation (keeps only the latest 3 checkpoints to save space).
- **Configurable**: Fully YAML-based configuration for all training parameters.

## Directory Structure

```
.
├── config.pretrain.yaml    # Configuration for pre-training
├── config.midtrain.yaml    # Configuration for mid-training
├── config.finetune.yaml    # Configuration for fine-tuning
├── run_pipeline.sh         # Master script to run the full pipeline
├── src/
│   ├── main.py             # Entry point for training
│   ├── trainer.py          # Trainer class handling the training loop
│   ├── model.py            # GPT model architecture definition
│   ├── dataloader.py       # Distributed data loading logic
│   └── muon.py             # Muon optimizer implementation
├── scripts/                # Helper scripts (e.g., HF conversion)
└── checkpoints/            # Directory where models are saved
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Dependencies**:
    Ensure you have PyTorch installed with CUDA support.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project requires PyTorch 2.0+ for optimal performance.*

## Usage

### 1. Configuration

The pipeline uses YAML files for configuration. You can customize `config.pretrain.yaml`, `config.midtrain.yaml`, and `config.finetune.yaml` to suit your needs.

**Example `config.pretrain.yaml`:**
```yaml
stage: "pretrain"
data_dir: "./data/pretraining"
output_dir: "./checkpoints/pretrain"
model:
  dim: 2048
  n_layers: 20
  n_heads: 32
batch_size: 8
grad_accum: 16
learning_rate: 6.0e-4
use_muon: false
```

### 2. Running the Pipeline

To run the entire pipeline (Pre-train -> Mid-train -> Fine-tune) sequentially:

```bash
./run_pipeline.sh
```

This script will:
1.  Run Pre-training using `config.pretrain.yaml`.
2.  Run Mid-training using `config.midtrain.yaml` (auto-detecting the latest pre-trained checkpoint).
3.  Run Fine-tuning using `config.finetune.yaml` (auto-detecting the latest mid-trained checkpoint).
4.  (Optional) Export the final model to Hugging Face format.

### 3. Running Individual Stages

You can also run specific stages manually using `torchrun`:

**Pre-training:**
```bash
torchrun --nproc_per_node=8 src/main.py --config config.pretrain.yaml
```

**Mid-training (Resuming from Pre-train):**
Ensure `resume_from` in `config.midtrain.yaml` points to your pre-trained checkpoint, or use the auto-resume logic in `run_pipeline.sh`.
```bash
torchrun --nproc_per_node=8 src/main.py --config config.midtrain.yaml
```

## Advanced Configuration

### Muon Optimizer
To use the Muon optimizer (often converges faster for certain architectures):
```yaml
use_muon: true
```

### Batch Size Scheduling
To dynamically increase batch size (simulating larger batch sizes via gradient accumulation):
```yaml
optimization:
  batch_size_schedule:
    strategy: "doubling"
    start_batch_size: 100
    interval_steps: 1000
```

## License

[MIT License](LICENSE)
