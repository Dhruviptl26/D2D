"""
Training pipeline for D2D ML models.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from d2d.data import DataLoader, DataPreprocessor
from d2d.models import ModelFactory, MLModel
from d2d.evaluation import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """Main training class for ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.evaluator = Evaluator()
        
    @classmethod
    def from_config_file(cls, config_path: str) -> 'Trainer':
        """Create trainer from configuration file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return cls(config)
        
    def load_data(self) -> pd.DataFrame:
        """Load training data."""
        data_config = self.config.get('data', {})
        filepath = data_config.get('filepath')
        
        if not filepath:
            raise ValueError("Data filepath not specified in config")
            
        if filepath.endswith('.csv'):
            data = self.data_loader.load_csv(filepath)
        elif filepath.endswith('.json'):
            data = self.data_loader.load_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
            
        logger.info(f"Loaded training data: {data.shape}")
        return data
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Preprocess the data."""
        preprocess_config = self.config.get('preprocessing', {})
        
        # Clean data
        data = self.preprocessor.clean_data(
            data,
            drop_duplicates=preprocess_config.get('drop_duplicates', True),
            handle_missing=preprocess_config.get('handle_missing', 'drop')
        )
        
        # Encode categorical variables
        categorical_columns = preprocess_config.get('categorical_columns')
        data = self.preprocessor.encode_categorical(data, categorical_columns)
        
        # Scale features
        numeric_columns = preprocess_config.get('numeric_columns')
        scaling_method = preprocess_config.get('scaling_method', 'standard')
        
        # Exclude target column from scaling
        target_column = self.config.get('target_column')
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if target_column in numeric_columns:
                numeric_columns.remove(target_column)
        elif target_column in numeric_columns:
            numeric_columns = [col for col in numeric_columns if col != target_column]
            
        data = self.preprocessor.scale_features(data, numeric_columns, scaling_method)
        
        # Split data
        target_column = self.config.get('target_column')
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            data, target_column, test_size, random_state
        )
        
        return X_train, X_test, y_train, y_test
        
    def create_model(self) -> MLModel:
        """Create the ML model."""
        model_config = self.config.get('model', {})
        problem_type = model_config.get('problem_type', 'classification')
        model_type = model_config.get('model_type', 'random_forest')
        model_params = model_config.get('parameters', {})
        
        self.model = ModelFactory.create_model(problem_type, model_type, **model_params)
        logger.info(f"Created {problem_type} model: {model_type}")
        return self.model
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> MLModel:
        """Train the model."""
        if self.model is None:
            self.create_model()
            
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
        return self.model
        
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate the trained model."""
        if self.model is None or not self.model.is_fitted:
            raise ValueError("Model must be trained before evaluation")
            
        predictions = self.model.predict(X_test)
        
        model_config = self.config.get('model', {})
        problem_type = model_config.get('problem_type', 'classification')
        
        if problem_type == 'classification':
            metrics = self.evaluator.evaluate_classification(y_test, predictions)
        else:
            metrics = self.evaluator.evaluate_regression(y_test, predictions)
            
        logger.info(f"Model evaluation completed: {metrics}")
        return metrics
        
    def save_model(self, filepath: Optional[str] = None) -> None:
        """Save the trained model."""
        if filepath is None:
            output_config = self.config.get('output', {})
            filepath = output_config.get('model_path', 'models/trained_model.pkl')
            
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(filepath)
        
        # Also save preprocessor
        preprocessor_path = filepath.replace('.pkl', '_preprocessor.pkl')
        import joblib
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
    def run_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("Starting training pipeline")
        
        # Load and preprocess data
        data = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        # Save model
        self.save_model()
        
        results = {
            'metrics': metrics,
            'data_shape': data.shape,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'config': self.config
        }
        
        logger.info("Training pipeline completed successfully")
        return results


def train_from_config(config_path: str) -> Dict[str, Any]:
    """Convenience function to train model from config file."""
    trainer = Trainer.from_config_file(config_path)
    return trainer.run_training_pipeline()