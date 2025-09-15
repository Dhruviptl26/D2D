# D2D - Machine Learning Project

A comprehensive machine learning project framework for data-driven decision making.

## Overview

This project provides a complete machine learning pipeline including data preprocessing, model training, evaluation, and inference capabilities.

## Features

- Data loading and preprocessing utilities
- Multiple ML model implementations
- Model training and evaluation pipeline
- Prediction and inference system
- Configuration management
- Logging and monitoring
- CLI interface for easy usage

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python -m d2d.train --config config/default.yaml
```

3. Make predictions:
```bash
python -m d2d.predict --model models/trained_model.pkl --data data/test.csv
```

## Project Structure

```
d2d/
├── data/               # Data loading and preprocessing
├── models/             # ML model implementations
├── training/           # Training pipeline
├── evaluation/         # Model evaluation utilities
├── inference/          # Prediction and inference
├── utils/              # Common utilities
└── config/             # Configuration files
```

## Documentation

See the `docs/` directory for detailed documentation on each component.