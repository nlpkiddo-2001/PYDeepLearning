# Project Structure

```
llm-from-scratch/
│
├── data/
│   └── raw_dataset/
│
├── src/
│   ├── model.py         # The language model architecture (e.g., GPT-like)
│   ├── tokenizer.py     # Code for training and using our tokenizer
│   ├── dataset.py       # Data loading and preprocessing logic
│   ├── train.py         # The main script to run the training loop
│   └── config.py        # All hyperparameters and configuration settings
│
├── notebooks/
│   └── 01_data_exploration.ipynb # For initial exploration of our dataset
│
├── scripts/
│   └── download_dataset.sh # A shell script to download the dataset
│
├── checkpoints/
│   └── # Directory to save model weights during training
│
├── requirements.txt     # To manage Python package dependencies
└── README.md            # Project documentation
```



This repo is currently focusing on training a model in a single GPU machine. in the future it will be scaled to train on multiple instance :) Happy Coding
