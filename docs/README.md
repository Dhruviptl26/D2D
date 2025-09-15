# D2D Machine Learning Project - Documentation

## Overview

The D2D (Data-to-Decision) Machine Learning Project is a comprehensive framework for building, training, and deploying machine learning models. It provides a complete pipeline from data preprocessing to model inference.

## Features

- **Comprehensive Data Pipeline**: Data loading, cleaning, preprocessing, and splitting
- **Multiple ML Models**: Support for classification and regression with various algorithms
- **Model Training**: Automated training pipeline with configuration management
- **Model Evaluation**: Comprehensive metrics and visualization capabilities
- **Inference System**: Easy prediction and batch processing
- **CLI Interface**: Command-line tools for training and prediction
- **Extensible Architecture**: Modular design for easy customization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd D2D
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Generate Sample Data

```bash
python generate_sample_data.py --samples 1000 --features 10 --problem-type classification
```

### 2. Train a Model

```bash
python -m d2d.train --config d2d/config/default.yaml
```

### 3. Make Predictions

```bash
python -m d2d.predict --model models/trained_model.pkl --data data/test_data.csv --preprocessor models/trained_model_preprocessor.pkl
```

## Configuration

The system uses YAML configuration files to define training parameters. See `d2d/config/default.yaml` for an example configuration.

### Configuration Sections

- **data**: Data loading settings
- **preprocessing**: Data cleaning and transformation options
- **model**: Model type and hyperparameters
- **output**: File paths for saving models and results

## Architecture

### Core Modules

1. **d2d.data**: Data loading and preprocessing utilities
2. **d2d.models**: Machine learning model implementations
3. **d2d.training**: Training pipeline and workflow management
4. **d2d.evaluation**: Model evaluation and metrics
5. **d2d.inference**: Prediction and inference capabilities
6. **d2d.utils**: Utility functions and helpers

### Supported Models

#### Classification
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)

#### Regression
- Random Forest Regressor
- Linear Regression
- Support Vector Regression (SVR)

## API Reference

### DataLoader

```python
from d2d.data import DataLoader

loader = DataLoader()
data = loader.load_csv('data.csv')
info = loader.get_data_info()
```

### DataPreprocessor

```python
from d2d.data import DataPreprocessor

preprocessor = DataPreprocessor()
cleaned_data = preprocessor.clean_data(data)
encoded_data = preprocessor.encode_categorical(data, ['category_col'])
scaled_data = preprocessor.scale_features(data, ['numeric_col1', 'numeric_col2'])
```

### Model Training

```python
from d2d.training import train_from_config

results = train_from_config('config.yaml')
```

### Model Inference

```python
from d2d.inference import Predictor

predictor = Predictor('model.pkl', 'preprocessor.pkl')
predictions = predictor.predict(data)
probabilities = predictor.predict_proba(data)
```

## Examples

### Custom Model Training

```python
from d2d.data import DataLoader, DataPreprocessor
from d2d.models import ModelFactory
from d2d.evaluation import Evaluator

# Load and preprocess data
loader = DataLoader()
data = loader.load_csv('data.csv')

preprocessor = DataPreprocessor()
data = preprocessor.clean_data(data)
data = preprocessor.encode_categorical(data)
data = preprocessor.scale_features(data)

X_train, X_test, y_train, y_test = preprocessor.split_data(data, 'target')

# Train model
model = ModelFactory.create_model('classification', 'random_forest')
model.fit(X_train, y_train)

# Evaluate
evaluator = Evaluator()
predictions = model.predict(X_test)
metrics = evaluator.evaluate_classification(y_test, predictions)
```

### Batch Prediction

```python
from d2d.inference import Predictor

predictor = Predictor('model.pkl', 'preprocessor.pkl')

# Process multiple files
data_files = ['data1.csv', 'data2.csv', 'data3.csv']
for file in data_files:
    results = predictor.predict_from_file(file, f'predictions_{file}')
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions and support, please open an issue on the repository.