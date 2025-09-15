"""
Model evaluation utilities for D2D ML project.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Model evaluation and metrics calculation."""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_classification(self, y_true: pd.Series, y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate classification model performance."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # ROC AUC if probabilities provided and binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
                y_pred_proba_pos = y_pred_proba[:, 1]
            else:
                y_pred_proba_pos = y_pred_proba
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba_pos)
        
        logger.info(f"Classification evaluation completed - Accuracy: {metrics['accuracy']:.4f}")
        return metrics
        
    def evaluate_regression(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate regression model performance."""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['mean_residual'] = np.mean(y_true - y_pred)
        metrics['std_residual'] = np.std(y_true - y_pred)
        
        logger.info(f"Regression evaluation completed - R2: {metrics['r2_score']:.4f}")
        return metrics
        
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
            
        return plt.gcf()
        
    def plot_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve for binary classification."""
        if len(np.unique(y_true)) != 2:
            raise ValueError("ROC curve is only available for binary classification")
            
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
            y_pred_proba_pos = y_pred_proba[:, 1]
        else:
            y_pred_proba_pos = y_pred_proba
            
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba_pos)
        auc_score = roc_auc_score(y_true, y_pred_proba_pos)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
            
        return plt.gcf()
        
    def plot_regression_results(self, y_true: pd.Series, y_pred: np.ndarray,
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot regression results (actual vs predicted)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted')
        ax1.grid(True)
        
        # Residuals
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regression plots saved to {save_path}")
            
        return fig
        
    def generate_evaluation_report(self, metrics: Dict[str, Any], 
                                 problem_type: str,
                                 save_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report."""
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Problem Type: {problem_type.title()}")
        report_lines.append("")
        
        if problem_type == 'classification':
            report_lines.append("CLASSIFICATION METRICS:")
            report_lines.append("-" * 25)
            report_lines.append(f"Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
            report_lines.append(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
            report_lines.append(f"Recall:    {metrics.get('recall', 'N/A'):.4f}")
            report_lines.append(f"F1-Score:  {metrics.get('f1_score', 'N/A'):.4f}")
            
            if 'roc_auc' in metrics:
                report_lines.append(f"ROC AUC:   {metrics['roc_auc']:.4f}")
                
        elif problem_type == 'regression':
            report_lines.append("REGRESSION METRICS:")
            report_lines.append("-" * 20)
            report_lines.append(f"R² Score: {metrics.get('r2_score', 'N/A'):.4f}")
            report_lines.append(f"RMSE:     {metrics.get('rmse', 'N/A'):.4f}")
            report_lines.append(f"MAE:      {metrics.get('mae', 'N/A'):.4f}")
            report_lines.append(f"MSE:      {metrics.get('mse', 'N/A'):.4f}")
            
        report_lines.append("")
        report_lines.append("=" * 50)
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
            
        return report
        
    def compare_models(self, model_results: Dict[str, Dict[str, Any]],
                      metric: str = 'accuracy') -> pd.DataFrame:
        """Compare multiple models based on a specific metric."""
        comparison_data = []
        
        for model_name, results in model_results.items():
            if metric in results:
                comparison_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': results[metric]
                })
                
        if not comparison_data:
            raise ValueError(f"Metric '{metric}' not found in any model results")
            
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Value', ascending=False)
        
        logger.info(f"Model comparison completed for metric: {metric}")
        return df