#!/usr/bin/env python3
"""
Command-line interface for D2D ML prediction.
"""

import click
import logging
import pandas as pd
from pathlib import Path

from d2d.inference import Predictor
from d2d.utils import setup_logging


@click.command()
@click.option('--model', '-m', type=str, required=True,
              help='Path to trained model file')
@click.option('--data', '-d', type=str, required=True,
              help='Path to input data file (CSV or JSON)')
@click.option('--preprocessor', '-p', type=str, default=None,
              help='Path to preprocessor file (optional)')
@click.option('--output', '-o', type=str, default=None,
              help='Path to save predictions (optional)')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
@click.option('--show-probabilities', is_flag=True, default=False,
              help='Show prediction probabilities (for classification)')
def predict(model, data, preprocessor, output, log_level, show_probabilities):
    """Make predictions using a trained model."""
    
    # Setup logging
    setup_logging(log_level)
    
    try:
        # Validate inputs
        if not Path(model).exists():
            raise FileNotFoundError(f"Model file not found: {model}")
        
        if not Path(data).exists():
            raise FileNotFoundError(f"Data file not found: {data}")
        
        if preprocessor and not Path(preprocessor).exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor}")
        
        click.echo(f"Loading model from: {model}")
        
        # Create predictor
        predictor = Predictor(model, preprocessor)
        
        # Get model info
        model_info = predictor.get_model_info()
        click.echo(f"Model type: {model_info.get('model_type', 'Unknown')}")
        
        click.echo(f"Loading data from: {data}")
        
        # Make predictions
        results = predictor.predict_from_file(data, output)
        
        # Display results summary
        click.echo("\n" + "="*50)
        click.echo("PREDICTION COMPLETED")
        click.echo("="*50)
        
        click.echo(f"Input samples: {len(results)}")
        click.echo(f"Predictions generated: {len(results['predictions'])}")
        
        # Show sample predictions
        click.echo("\nSample predictions:")
        sample_size = min(5, len(results))
        for i in range(sample_size):
            pred_value = results['predictions'].iloc[i]
            click.echo(f"  Sample {i+1}: {pred_value}")
            
            if show_probabilities:
                prob_cols = [col for col in results.columns if col.startswith('probability')]
                if prob_cols:
                    prob_str = ", ".join([f"{col}: {results[col].iloc[i]:.3f}" for col in prob_cols])
                    click.echo(f"    Probabilities: {prob_str}")
        
        if output:
            click.echo(f"\nPredictions saved to: {output}")
        
        # Show prediction statistics
        predictions = results['predictions']
        if pd.api.types.is_numeric_dtype(predictions):
            click.echo(f"\nPrediction statistics:")
            click.echo(f"  Mean: {predictions.mean():.4f}")
            click.echo(f"  Std:  {predictions.std():.4f}")
            click.echo(f"  Min:  {predictions.min():.4f}")
            click.echo(f"  Max:  {predictions.max():.4f}")
        else:
            click.echo(f"\nPrediction distribution:")
            value_counts = predictions.value_counts()
            for value, count in value_counts.items():
                click.echo(f"  {value}: {count} ({count/len(predictions)*100:.1f}%)")
        
        click.echo("\nPrediction completed successfully!")
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    predict()