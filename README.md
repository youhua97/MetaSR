# Bridging NIP and MLM: A Unified Meta-Learning Framework for Sequential Recommendation

This repository is the official implementation of the paper "Bridging NIP and MLM: A Unified Meta-Learning Framework for Sequential Recommendation." This project implements a meta-learning-enhanced BERT-based sequential recommendation system that bridges the gap between the Next Item Prediction and Masked Language Modeling paradigms.

## Requirements

### Environment

- Python 3.7+
- PyTorch 1.8.0+
- CUDA 10.2+ (for GPU support)

### Dependencies

Install the required packages using pip:

```bash
pip install torch>=1.8.0
pip install numpy
pip install pandas
pip install tqdm
pip install tensorboard
pip install scikit-learn
```

## Dataset Preparation

### Supported Datasets

1. **MovieLens-1M**: Place the dataset files in `Data/ml-1m/`
   - `ratings.dat`
   - `movies.dat`
   - `users.dat`

2. **MovieLens-20M**: Place the dataset files in `Data/ml-20m/`
   - `ratings.csv`
   - `movies.csv`

3. **Amazon Beauty**: Place the dataset files in `Data/beauty/`
   - `ratings_Beauty.csv`

4. **Amazon Toys**: Place the dataset files in `Data/toys/`
   - Rating files in CSV format

### Data Preprocessing

The framework automatically preprocesses raw data with the following settings:
- Minimum user interactions: 5 (configurable via `--min_uc`)
- Minimum item interactions: 5 (configurable via `--min_sc`)
- Evaluation split: Leave-one-out (configurable via `--split`)

Preprocessed data will be cached in `Data/preprocessed/` for faster subsequent runs.

## Usage

### Training

To train the model with default settings on the Beauty dataset:

```bash
python main.py --template train --dataset_code beauty
```

### Training with Custom Configurations

For MovieLens-1M dataset:

```bash
python main.py --template train --dataset_code ml-1m --device cuda --device_idx 0
```

For MovieLens-20M dataset:

```bash
python main.py --template train --dataset_code ml-20m --num_epochs 50
```

### Model Evaluation

To evaluate a trained model:

```bash
python test_meta.py --dataset_code beauty --test_model_path path/to/model.pth
```

## Project Structure

```
.
├── main.py                     # Main training script
├── test_meta.py               # Model evaluation script
├── options.py                 # Command-line argument parser
├── config.py                  # Global configuration settings
├── templates.py               # Dataset-specific configurations
├── utils.py                   # Utility functions
├── loggers.py                 # Training logging utilities
├── models/
│   ├── bert.py               # BERT recommendation model
│   ├── base.py               # Base model class
│   └── bert_modules/         # BERT components
│       ├── bert.py           # BERT encoder
│       ├── transformer.py    # Transformer blocks
│       ├── embedding/        # Embedding layers
│       └── attention/        # Attention mechanisms
├── trainers/
│   ├── bert.py               # BERT trainer implementation
│   ├── base.py               # Base trainer with meta-learning
│   ├── metanetwork.py        # Meta-learner network
│   └── utils.py              # Training utilities
├── dataloaders/
│   ├── bert.py               # BERT dataloader
│   ├── base.py               # Base dataloader
│   └── negative_samplers/    # Negative sampling strategies
├── datasets/
│   ├── base.py               # Dataset base class
│   ├── ml_1m.py              # MovieLens-1M dataset
│   ├── ml_20m.py             # MovieLens-20M dataset
│   ├── beauty.py             # Amazon Beauty dataset
│   └── utils.py              # Dataset utilities
└── Data/                      # Dataset directory
```
