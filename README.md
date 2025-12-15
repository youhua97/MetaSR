# Bridging NIP and MLM: A Unified Meta-Learning Framework for Sequential Recommendation

This repository is the official implementation of the paper "Bridging NIP and MLM: A Unified Meta-Learning Framework for Sequential Recommendation." This project implements a meta-learning-enhanced BERT-based sequential recommendation system that bridges the gap between the Next Item Prediction and Masked Language Modeling paradigms.

## Project Structure

```
.
├── main.py                    # Main training script
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
└── Data/                     # Dataset directory
```

More coming soon ...
