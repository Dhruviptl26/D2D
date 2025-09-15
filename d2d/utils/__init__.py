"""
Utility functions and helpers for D2D ML project.
"""

import logging
import os
import yaml
import json
import pickle
from typing import Any, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logging.info("Logging setup complete")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Error saving config to {config_path}: {e}")
        raise


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        logging.info(f"JSON data loaded from {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON from {filepath}: {e}")
        raise


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=2, default=str)
        logging.info(f"JSON data saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {e}")
        raise


def load_pickle(filepath: str) -> Any:
    """Load data from pickle file."""
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        logging.info(f"Pickle data loaded from {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading pickle from {filepath}: {e}")
        raise


def save_pickle(data: Any, filepath: str) -> None:
    """Save data to pickle file."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as file:
            pickle.dump(data, file)
        logging.info(f"Pickle data saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving pickle to {filepath}: {e}")
        raise


def create_directory(directory: str) -> None:
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory created: {directory}")


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(filepath)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def validate_data_types(data: pd.DataFrame, expected_types: Dict[str, str]) -> bool:
    """Validate data types of DataFrame columns."""
    for column, expected_type in expected_types.items():
        if column not in data.columns:
            logging.error(f"Column '{column}' not found in data")
            return False
            
        actual_type = str(data[column].dtype)
        if expected_type not in actual_type:
            logging.error(f"Column '{column}' has type {actual_type}, expected {expected_type}")
            return False
    
    logging.info("Data type validation passed")
    return True


def check_missing_values(data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    """Check for missing values and return columns with high missing rates."""
    missing_rates = data.isnull().sum() / len(data)
    high_missing = missing_rates[missing_rates > threshold]
    
    if len(high_missing) > 0:
        logging.warning(f"Columns with high missing rates: {high_missing.to_dict()}")
    else:
        logging.info("No columns with high missing rates found")
    
    return high_missing.to_dict()


def memory_usage_summary(data: pd.DataFrame) -> Dict[str, str]:
    """Get memory usage summary of DataFrame."""
    memory_usage = data.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    summary = {
        'total_memory': format_file_size(total_memory),
        'shape': data.shape,
        'columns': len(data.columns),
        'memory_per_column': {
            col: format_file_size(memory_usage[col]) 
            for col in data.columns
        }
    }
    
    logging.info(f"Memory usage: {summary['total_memory']} for shape {summary['shape']}")
    return summary


def generate_sample_data(n_samples: int = 1000, n_features: int = 10, 
                        problem_type: str = 'classification',
                        random_state: int = 42) -> pd.DataFrame:
    """Generate sample data for testing purposes."""
    np.random.seed(random_state)
    
    # Generate features
    data = {}
    for i in range(n_features):
        if i % 3 == 0:  # Categorical feature
            data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)
        else:  # Numeric feature
            data[f'num_feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Generate target
    if problem_type == 'classification':
        data['target'] = np.random.choice([0, 1], n_samples)
    else:  # regression
        data['target'] = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    logging.info(f"Generated sample data: {df.shape}")
    return df


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        logging.info(f"Started: {self.name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        logging.info(f"Completed: {self.name} in {elapsed:.2f} seconds")


def ensure_reproducibility(seed: int = 42) -> None:
    """Ensure reproducibility by setting random seeds."""
    import random
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set environment variable for TensorFlow/Keras
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        # Set PyTorch random seed if available
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        # Set TensorFlow random seed if available
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    logging.info(f"Random seeds set to {seed} for reproducibility")