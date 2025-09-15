#!/usr/bin/env python3
"""
Command-line interface for D2D ML training.
"""

import click
import logging
from pathlib import Path

from d2d.training import train_from_config
from d2d.utils import setup_logging, load_config


@click.command()
@click.option('--config', '-c', type=str, required=True,
              help='Path to configuration YAML file')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
@click.option('--log-file', type=str, default=None,
              help='Path to log file (optional)')
def train(config, log_level, log_file):
    """Train a machine learning model using the specified configuration."""
    
    # Setup logging
    setup_logging(log_level, log_file)
    
    try:
        # Load and validate config
        if not Path(config).exists():
            raise FileNotFoundError(f"Configuration file not found: {config}")
        
        click.echo(f"Starting training with config: {config}")
        
        # Run training pipeline
        results = train_from_config(config)
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("TRAINING COMPLETED SUCCESSFULLY")
        click.echo("="*50)
        
        click.echo(f"Data shape: {results['data_shape']}")
        click.echo(f"Training set: {results['train_shape']}")
        click.echo(f"Test set: {results['test_shape']}")
        
        metrics = results['metrics']
        click.echo("\nModel Performance:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                click.echo(f"  {metric}: {value:.4f}")
        
        config_data = results['config']
        output_config = config_data.get('output', {})
        model_path = output_config.get('model_path', 'models/trained_model.pkl')
        
        click.echo(f"\nModel saved to: {model_path}")
        click.echo("Training pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    train()