"""
Inference and prediction utilities for D2D ML project.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging

from d2d.models import MLModel
from d2d.data import DataPreprocessor

logger = logging.getLogger(__name__)


class Predictor:
    """Handle model inference and predictions."""
    
    def __init__(self, model_path: Optional[str] = None, 
                 preprocessor_path: Optional[str] = None):
        self.model = None
        self.preprocessor = None
        
        if model_path:
            self.load_model(model_path)
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)
            
    def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
            
    def load_preprocessor(self, preprocessor_path: str) -> None:
        """Load a fitted preprocessor."""
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        except Exception as e:
            logger.error(f"Error loading preprocessor from {preprocessor_path}: {e}")
            raise
            
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        if self.preprocessor is None:
            logger.warning("No preprocessor loaded, using raw data")
            return data
            
        try:
            processed_data = self.preprocessor.transform_new_data(data)
            logger.info(f"Preprocessed input data: {processed_data.shape}")
            return processed_data
        except Exception as e:
            logger.error(f"Error preprocessing input data: {e}")
            raise
            
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions on input data."""
        if self.model is None:
            raise ValueError("No model loaded. Use load_model() first.")
            
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            
        # Preprocess if preprocessor is available
        if self.preprocessor is not None:
            data = self.preprocess_input(data)
            
        try:
            predictions = self.model.predict(data)
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
            
    def predict_proba(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get prediction probabilities (for classification models)."""
        if self.model is None:
            raise ValueError("No model loaded. Use load_model() first.")
            
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            
        # Preprocess if preprocessor is available
        if self.preprocessor is not None:
            data = self.preprocess_input(data)
            
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(data)
                logger.info(f"Generated probability predictions for {len(probabilities)} samples")
                return probabilities
            else:
                raise ValueError("Model does not support probability predictions")
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {e}")
            raise
            
    def predict_from_file(self, filepath: str, 
                         output_path: Optional[str] = None,
                         exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Make predictions on data from a file."""
        # Load data
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            data = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
            
        logger.info(f"Loaded data from {filepath}: {data.shape}")
        
        # Remove excluded columns (like target column for prediction)
        if exclude_columns:
            data_for_prediction = data.drop(columns=[col for col in exclude_columns if col in data.columns])
        else:
            # Try to automatically detect and exclude common target column names
            common_targets = ['target', 'label', 'class', 'y']
            data_for_prediction = data.drop(columns=[col for col in common_targets if col in data.columns])
        
        # Make predictions
        predictions = self.predict(data_for_prediction)
        
        # Create results DataFrame
        results = data.copy()
        results['predictions'] = predictions
        
        # Add probabilities if available
        try:
            probabilities = self.predict_proba(data_for_prediction)
            if probabilities.ndim > 1:
                for i in range(probabilities.shape[1]):
                    results[f'probability_class_{i}'] = probabilities[:, i]
            else:
                results['probability'] = probabilities
        except:
            logger.info("Probability predictions not available")
            
        # Save results if output path specified
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
            
        return results
        
    def batch_predict(self, data_list: List[Union[pd.DataFrame, np.ndarray]]) -> List[np.ndarray]:
        """Make predictions on multiple datasets."""
        predictions = []
        
        for i, data in enumerate(data_list):
            try:
                pred = self.predict(data)
                predictions.append(pred)
                logger.info(f"Processed batch {i+1}/{len(data_list)}")
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {e}")
                predictions.append(None)
                
        return predictions
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {}
        
        if self.model is not None:
            info['model_type'] = type(self.model).__name__
            info['model_loaded'] = True
            
            # Get model-specific info
            if hasattr(self.model, 'feature_importances_'):
                info['has_feature_importance'] = True
                info['n_features'] = len(self.model.feature_importances_)
            else:
                info['has_feature_importance'] = False
                
            if hasattr(self.model, 'classes_'):
                info['n_classes'] = len(self.model.classes_)
                info['classes'] = self.model.classes_.tolist()
                
        else:
            info['model_loaded'] = False
            
        if self.preprocessor is not None:
            info['preprocessor_loaded'] = True
            info['feature_names'] = getattr(self.preprocessor, 'feature_names', [])
        else:
            info['preprocessor_loaded'] = False
            
        return info
        
    def explain_prediction(self, data: Union[pd.DataFrame, np.ndarray],
                          sample_idx: int = 0) -> Dict[str, Any]:
        """Provide basic explanation for a prediction (feature importance if available)."""
        if self.model is None:
            raise ValueError("No model loaded")
            
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            
        # Make prediction
        prediction = self.predict(data.iloc[[sample_idx]])
        
        explanation = {
            'sample_index': sample_idx,
            'prediction': prediction[0],
            'input_features': data.iloc[sample_idx].to_dict()
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            if self.preprocessor and hasattr(self.preprocessor, 'feature_names'):
                feature_names = self.preprocessor.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
                
            importance_dict = dict(zip(feature_names, self.model.feature_importances_))
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            explanation['feature_importance'] = sorted_importance[:10]  # Top 10 features
            
        return explanation


def load_predictor(model_path: str, preprocessor_path: Optional[str] = None) -> Predictor:
    """Convenience function to load a predictor with model and preprocessor."""
    return Predictor(model_path, preprocessor_path)


def predict_from_config(config: Dict[str, Any], input_data: pd.DataFrame) -> np.ndarray:
    """Make predictions using configuration settings."""
    model_path = config.get('model_path')
    preprocessor_path = config.get('preprocessor_path')
    
    if not model_path:
        raise ValueError("Model path not specified in config")
        
    predictor = Predictor(model_path, preprocessor_path)
    return predictor.predict(input_data)