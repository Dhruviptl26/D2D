#!/usr/bin/env python3
"""
D2D Machine Learning Project Demo

This script demonstrates the complete machine learning workflow
from data generation to model training and prediction.
"""

import os
import sys
from pathlib import Path

# Add the project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from d2d.utils import setup_logging, generate_sample_data
from d2d.training import train_from_config
from d2d.inference import Predictor


def main():
    """Run the complete ML demonstration."""
    
    # Setup logging
    setup_logging('INFO')
    print("=" * 60)
    print("D2D MACHINE LEARNING PROJECT DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Generate training data
    print("\n1. Generating training data...")
    train_data = generate_sample_data(
        n_samples=1000, 
        n_features=12, 
        problem_type='classification'
    )
    
    # Save training data
    os.makedirs('demo_data', exist_ok=True)
    train_data.to_csv('demo_data/train_data.csv', index=False)
    print(f"✓ Training data saved: {train_data.shape}")
    
    # Step 2: Generate test data 
    print("\n2. Generating test data...")
    test_data = generate_sample_data(
        n_samples=200,
        n_features=12,
        problem_type='classification',
        random_state=123  # Different seed for test data
    )
    test_data.to_csv('demo_data/test_data.csv', index=False)
    print(f"✓ Test data saved: {test_data.shape}")
    
    # Step 3: Create demo configuration
    print("\n3. Creating training configuration...")
    demo_config = {
        'data': {'filepath': 'demo_data/train_data.csv'},
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
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        'output': {
            'model_path': 'demo_models/demo_model.pkl',
            'preprocessor_path': 'demo_models/demo_preprocessor.pkl'
        }
    }
    
    # Save configuration
    import yaml
    os.makedirs('demo_config', exist_ok=True)
    with open('demo_config/demo.yaml', 'w') as f:
        yaml.dump(demo_config, f, default_flow_style=False, indent=2)
    print("✓ Configuration created")
    
    # Step 4: Train model
    print("\n4. Training machine learning model...")
    results = train_from_config('demo_config/demo.yaml')
    
    print(f"✓ Model trained successfully!")
    print(f"  - Training accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"  - F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"  - Model saved to: {demo_config['output']['model_path']}")
    
    # Step 5: Make predictions
    print("\n5. Making predictions on test data...")
    predictor = Predictor(
        'demo_models/demo_model.pkl',
        'demo_models/demo_model_preprocessor.pkl'  # Use the correct preprocessor filename
    )
    
    predictions_df = predictor.predict_from_file(
        'demo_data/test_data.csv',
        'demo_results/predictions.csv'
    )
    
    print(f"✓ Predictions completed!")
    print(f"  - Test samples: {len(predictions_df)}")
    print(f"  - Predictions saved to: demo_results/predictions.csv")
    
    # Step 6: Show sample results
    print("\n6. Sample predictions:")
    sample_results = predictions_df[['target', 'predictions']].head(10)
    correct_predictions = (sample_results['target'] == sample_results['predictions']).sum()
    
    print(sample_results.to_string(index=False))
    print(f"\nAccuracy on first 10 samples: {correct_predictions}/10 = {correct_predictions/10:.1%}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("🎉 The D2D ML framework is working perfectly!")
    print("\nGenerated files:")
    print("  📊 Training data: demo_data/train_data.csv")
    print("  📊 Test data: demo_data/test_data.csv")
    print("  ⚙️  Configuration: demo_config/demo.yaml")
    print("  🤖 Trained model: demo_models/demo_model.pkl")
    print("  🔧 Preprocessor: demo_models/demo_preprocessor.pkl")
    print("  📈 Predictions: demo_results/predictions.csv")
    
    print(f"\nTo run predictions on new data:")
    print(f"python -m d2d.predict --model demo_models/demo_model.pkl \\")
    print(f"                      --data your_data.csv \\") 
    print(f"                      --preprocessor demo_models/demo_preprocessor.pkl")


if __name__ == '__main__':
    main()