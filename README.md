# Bridging Next Item Prediction and Masked Language Modeling

This repository contains the official implementation of the paper on bridging Next Item Prediction (NIP) and Masked Language Modeling (MLM) for sequential recommendation using meta-learning approaches.

## Abstract

This project implements a meta-learning enhanced BERT-based sequential recommendation system that bridges the gap between Next Item Prediction and Masked Language Modeling paradigms. The framework incorporates a meta-learner that adaptively weighs task-specific information to improve recommendation performance across diverse user behaviors.

## Key Features

- BERT-based sequential recommendation model with transformer architecture
- Meta-learning framework for adaptive task weighting
- Support for multiple benchmark datasets (MovieLens-1M, MovieLens-20M, Amazon Beauty, Amazon Toys)
- Flexible configuration system for different datasets and hyperparameters
- Comprehensive evaluation metrics including Recall@K and NDCG@K

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

### Basic Training

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

### Important Hyperparameters

- `--dataset_code`: Dataset to use (beauty, toys, ml-1m, ml-20m)
- `--bert_hidden_units`: Hidden dimension size (default: 256)
- `--bert_num_blocks`: Number of transformer layers (default: 2)
- `--bert_num_heads`: Number of attention heads (default: 2 for Beauty/Toys, 4 for MovieLens)
- `--bert_mask_prob`: Masking probability for MLM (default: 0.15)
- `--num_tasks_selected`: Number of tasks selected for meta-learning (default: 2-4)
- `--lr`: Local learning rate (default: 0.001)
- `--lr_global`: Global model learning rate (default: 0.001)
- `--lr_learner`: Meta-learner learning rate (default: 0.001)
- `--num_epochs`: Number of training epochs (default: 100)

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

## Model Architecture

### BERT-based Sequential Recommendation

The model consists of three main components:

1. **Embedding Layer**: Token embeddings combined with positional encodings
2. **Transformer Encoder**: Multi-layer transformer blocks with multi-head self-attention
3. **Prediction Head**: Linear layer for item prediction

### Meta-Learning Framework

The meta-learner network takes user and task embeddings as input and outputs task weights to adaptively combine different learning objectives:

- User-specific embeddings capture individual preferences
- Task embeddings represent different learning scenarios
- Adaptive weighting mechanism balances between tasks

## Experimental Results

The model is evaluated using the following metrics:

- Recall@K (K = 1, 5, 10, 20)
- NDCG@K (K = 1, 5, 10, 20)

Training logs and tensorboard summaries are saved in the `experiments/` directory.

## Configuration Details

### Dataset-specific Settings

#### Beauty Dataset
- Sequence length: 15
- Dropout: 0.3
- Selected tasks: 4
- Epochs: 100

#### MovieLens-1M
- Sequence length: 40
- Dropout: 0.2
- Selected tasks: 2
- Epochs: 100

#### MovieLens-20M
- Sequence length: 50
- Dropout: 0.0
- Selected tasks: 2
- Epochs: 50

## Output Directory Structure

```
experiments/
└── experiment_description_timestamp/
    ├── models/
    │   ├── best_acc_model.pth     # Best performing model
    │   └── checkpoint-*.pth        # Training checkpoints
    ├── logs/
    │   └── training.log            # Training logs
    └── tensorboard/                # Tensorboard logs
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{bridging-nip-mlm,
  title={Bridging Next Item Prediction and Masked Language Modeling for Sequential Recommendation},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work builds upon the following research:

- BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer
- Meta-learning approaches for recommendation systems
- Transformer architecture for sequential modeling

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## Notes

- Ensure sufficient GPU memory for large datasets (MovieLens-20M requires at least 16GB)
- The first run will preprocess the dataset, which may take several minutes
- Training progress can be monitored using tensorboard: `tensorboard --logdir experiments/`
- Checkpoint models are automatically saved during training
- The framework supports multi-GPU training via the `--num_gpu` parameter
