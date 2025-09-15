"""
Test configuration file for pytest.
"""

import pytest
import warnings
import numpy as np

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set random seed for reproducible tests
np.random.seed(42)


@pytest.fixture(scope="session")
def suppress_sklearn_warnings():
    """Suppress sklearn warnings during tests."""
    import sklearn
    sklearn.set_config(assume_finite=True)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'filepath': 'test_data.csv'
        },
        'target_column': 'target',
        'test_size': 0.2,
        'random_state': 42,
        'preprocessing': {
            'drop_duplicates': True,
            'handle_missing': 'drop',
            'scaling_method': 'standard'
        },
        'model': {
            'problem_type': 'classification',
            'model_type': 'random_forest',
            'parameters': {
                'n_estimators': 10,
                'random_state': 42
            }
        },
        'output': {
            'model_path': 'test_model.pkl',
            'preprocessor_path': 'test_preprocessor.pkl'
        }
    }