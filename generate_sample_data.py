#!/usr/bin/env python3
"""
Generate sample data for D2D ML project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import click

from d2d.utils import generate_sample_data, setup_logging


@click.command()
@click.option('--samples', '-n', type=int, default=1000,
              help='Number of samples to generate')
@click.option('--features', '-f', type=int, default=10,
              help='Number of features to generate')
@click.option('--problem-type', type=click.Choice(['classification', 'regression']),
              default='classification', help='Type of ML problem')
@click.option('--output', '-o', type=str, default='data/sample_data.csv',
              help='Output file path')
@click.option('--seed', type=int, default=42,
              help='Random seed for reproducibility')
def generate_data(samples, features, problem_type, output, seed):
    """Generate sample data for testing D2D ML pipeline."""
    
    setup_logging()
    
    click.echo(f"Generating {samples} samples with {features} features")
    click.echo(f"Problem type: {problem_type}")
    click.echo(f"Random seed: {seed}")
    
    # Generate data
    data = generate_sample_data(
        n_samples=samples,
        n_features=features,
        problem_type=problem_type,
        random_state=seed
    )
    
    # Create output directory
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    data.to_csv(output, index=False)
    
    click.echo(f"\nData saved to: {output}")
    click.echo(f"Data shape: {data.shape}")
    click.echo(f"Columns: {list(data.columns)}")
    
    # Show data statistics
    click.echo("\nData summary:")
    click.echo(data.describe())
    
    click.echo("\nSample data generation completed!")


if __name__ == '__main__':
    generate_data()