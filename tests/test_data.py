"""
Test suite for D2D ML project data module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from d2d.data import DataLoader, DataPreprocessor


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.data is None
        
    @patch('pandas.read_csv')
    def test_load_csv_success(self, mock_read_csv):
        """Test successful CSV loading."""
        # Mock data
        mock_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_read_csv.return_value = mock_data
        
        loader = DataLoader()
        result = loader.load_csv('test.csv')
        
        assert result.equals(mock_data)
        assert loader.data.equals(mock_data)
        mock_read_csv.assert_called_once_with('test.csv')
        
    @patch('pandas.read_csv')
    def test_load_csv_error(self, mock_read_csv):
        """Test CSV loading with error."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_csv('nonexistent.csv')
            
    def test_get_data_info_no_data(self):
        """Test data info when no data is loaded."""
        loader = DataLoader()
        info = loader.get_data_info()
        assert info == {"error": "No data loaded"}
        
    def test_get_data_info_with_data(self):
        """Test data info with loaded data."""
        loader = DataLoader()
        test_data = pd.DataFrame({
            'A': [1, 2, None],
            'B': ['x', 'y', 'z']
        })
        loader.data = test_data
        
        info = loader.get_data_info()
        
        assert info['shape'] == (3, 2)
        assert info['columns'] == ['A', 'B']
        assert info['missing_values']['A'] == 1
        assert info['missing_values']['B'] == 0


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_init(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.scalers == {}
        assert preprocessor.encoders == {}
        assert preprocessor.feature_names == []
        
    def test_clean_data_drop_duplicates(self):
        """Test data cleaning with duplicate removal."""
        preprocessor = DataPreprocessor()
        data = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [4, 5, 5, 6]
        })
        
        cleaned = preprocessor.clean_data(data, drop_duplicates=True)
        
        assert len(cleaned) == 3  # One duplicate removed
        assert not cleaned.duplicated().any()
        
    def test_clean_data_handle_missing_drop(self):
        """Test data cleaning with missing value dropping."""
        preprocessor = DataPreprocessor()
        data = pd.DataFrame({
            'A': [1, 2, None, 3],
            'B': [4, None, 5, 6]
        })
        
        cleaned = preprocessor.clean_data(data, handle_missing='drop')
        
        assert len(cleaned) == 2  # Two rows with missing values removed
        assert not cleaned.isnull().any().any()
        
    def test_encode_categorical(self):
        """Test categorical encoding."""
        preprocessor = DataPreprocessor()
        data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C'],
            'numeric': [1, 2, 3, 4]
        })
        
        encoded = preprocessor.encode_categorical(data, ['category'])
        
        assert 'category' in preprocessor.encoders
        assert encoded['category'].dtype in ['int32', 'int64']
        assert set(encoded['category'].unique()) == {0, 1, 2}  # Three categories encoded
        
    def test_scale_features(self):
        """Test feature scaling."""
        preprocessor = DataPreprocessor()
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40]
        })
        
        scaled = preprocessor.scale_features(data, ['feature1', 'feature2'])
        
        assert 'features' in preprocessor.scalers
        # Check if scaled data has approximately zero mean and unit variance
        assert abs(scaled['feature1'].mean()) < 1e-10
        assert abs(scaled['feature1'].std() - 1.0) < 1e-10
        
    def test_split_data(self):
        """Test data splitting."""
        preprocessor = DataPreprocessor()
        data = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'target': [0, 1] * 50  # Binary target
        })
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            data, 'target', test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert set(X_train.columns) == {'feature1', 'feature2'}
        assert preprocessor.feature_names == ['feature1', 'feature2']


@pytest.fixture
def sample_data():
    """Fixture providing sample data for tests."""
    return pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'target': [0, 1, 0, 1, 1]
    })


def test_full_preprocessing_pipeline(sample_data):
    """Test complete preprocessing pipeline."""
    preprocessor = DataPreprocessor()
    
    # Clean data
    cleaned = preprocessor.clean_data(sample_data)
    
    # Encode categorical
    encoded = preprocessor.encode_categorical(cleaned, ['category'])
    
    # Scale features
    scaled = preprocessor.scale_features(encoded, ['numeric1', 'numeric2'])
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        scaled, 'target', test_size=0.2
    )
    
    assert len(X_train) + len(X_test) == len(sample_data)
    assert 'category' in preprocessor.encoders
    assert 'features' in preprocessor.scalers
    assert len(preprocessor.feature_names) == 3  # numeric1, numeric2, category