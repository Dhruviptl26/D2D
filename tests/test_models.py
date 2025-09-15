"""
Test suite for D2D ML project models module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

from d2d.models import MLModel, ClassificationModel, RegressionModel, ModelFactory


class TestModelFactory:
    """Test cases for ModelFactory class."""
    
    def test_create_classification_model(self):
        """Test creating classification model."""
        model = ModelFactory.create_model('classification', 'random_forest')
        assert isinstance(model, ClassificationModel)
        assert model.model_type == 'random_forest'
        
    def test_create_regression_model(self):
        """Test creating regression model."""
        model = ModelFactory.create_model('regression', 'random_forest')
        assert isinstance(model, RegressionModel)
        assert model.model_type == 'random_forest'
        
    def test_invalid_problem_type(self):
        """Test creating model with invalid problem type."""
        with pytest.raises(ValueError, match="Unsupported problem type"):
            ModelFactory.create_model('invalid', 'random_forest')
            
    def test_get_available_models_classification(self):
        """Test getting available classification models."""
        models = ModelFactory.get_available_models('classification')
        expected = ['random_forest', 'logistic_regression', 'svm']
        assert models == expected
        
    def test_get_available_models_regression(self):
        """Test getting available regression models."""
        models = ModelFactory.get_available_models('regression')
        expected = ['random_forest', 'linear_regression', 'svm']
        assert models == expected


class TestClassificationModel:
    """Test cases for ClassificationModel class."""
    
    def test_init_random_forest(self):
        """Test initialization with random forest."""
        model = ClassificationModel('random_forest', n_estimators=50)
        assert model.model_type == 'random_forest'
        assert not model.is_fitted
        
    def test_init_logistic_regression(self):
        """Test initialization with logistic regression."""
        model = ClassificationModel('logistic_regression')
        assert model.model_type == 'logistic_regression'
        
    def test_init_invalid_model(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unsupported classification model"):
            ClassificationModel('invalid_model')
            
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        model = ClassificationModel('random_forest', n_estimators=10)
        
        # Test fitting
        fitted_model = model.fit(X, y)
        assert fitted_model.is_fitted
        assert fitted_model is model  # Should return self
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
        
    def test_predict_proba(self):
        """Test probability prediction."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        model = ClassificationModel('random_forest', n_estimators=10)
        model.fit(X, y)
        
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(X), 2)  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        
    def test_predict_without_fitting(self):
        """Test prediction without fitting the model."""
        model = ClassificationModel('random_forest')
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)


class TestRegressionModel:
    """Test cases for RegressionModel class."""
    
    def test_init_random_forest(self):
        """Test initialization with random forest."""
        model = RegressionModel('random_forest', n_estimators=50)
        assert model.model_type == 'random_forest'
        assert not model.is_fitted
        
    def test_init_linear_regression(self):
        """Test initialization with linear regression."""
        model = RegressionModel('linear_regression')
        assert model.model_type == 'linear_regression'
        
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100))
        
        model = RegressionModel('random_forest', n_estimators=10)
        
        # Test fitting
        fitted_model = model.fit(X, y)
        assert fitted_model.is_fitted
        
        # Test prediction
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)


class TestMLModelSaveLoad:
    """Test model saving and loading functionality."""
    
    def test_save_and_load_model(self):
        """Test saving and loading model."""
        # Create and train a model
        X = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        y = pd.Series(np.random.choice([0, 1], 50))
        
        model = ClassificationModel('random_forest', n_estimators=10)
        model.fit(X, y)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            model.save_model(tmp_path)
            
            # Create new model and load
            new_model = ClassificationModel('random_forest')
            new_model.load_model(tmp_path)
            
            assert new_model.is_fitted
            
            # Test that predictions are the same
            original_pred = model.predict(X)
            loaded_pred = new_model.predict(X)
            np.testing.assert_array_equal(original_pred, loaded_pred)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    def test_save_unfitted_model(self):
        """Test saving an unfitted model raises error."""
        model = ClassificationModel('random_forest')
        
        with pytest.raises(ValueError, match="Model must be fitted before saving"):
            model.save_model('test.pkl')


@pytest.fixture
def sample_classification_data():
    """Fixture providing sample classification data."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y = pd.Series(np.random.choice([0, 1], 100))
    return X, y


@pytest.fixture
def sample_regression_data():
    """Fixture providing sample regression data."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y = pd.Series(np.random.randn(100))
    return X, y


def test_classification_pipeline(sample_classification_data):
    """Test complete classification pipeline."""
    X, y = sample_classification_data
    
    # Test different classification models
    for model_type in ['random_forest', 'logistic_regression']:
        model = ModelFactory.create_model('classification', model_type, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in y.unique() for pred in predictions)


def test_regression_pipeline(sample_regression_data):
    """Test complete regression pipeline."""
    X, y = sample_regression_data
    
    # Test different regression models
    for model_type in ['random_forest', 'linear_regression']:
        model = ModelFactory.create_model('regression', model_type, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)