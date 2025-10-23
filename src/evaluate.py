"""
Evaluation module for AIDM system.
Provides comprehensive metrics, visualization, and model performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AIDMEvaluator:
    """
    Comprehensive evaluator for AIDM system performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def compute_classification_metrics(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     y_scores: np.ndarray = None,
                                     prefix: str = "") -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities (optional)
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics[f'{prefix}specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False Positive Rate
        metrics[f'{prefix}fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # True Positive Rate (same as recall)
        metrics[f'{prefix}tpr'] = metrics[f'{prefix}recall']
        
        # AUC metrics if scores are provided
        if y_scores is not None and len(np.unique(y_true)) > 1:
            metrics[f'{prefix}auc_roc'] = roc_auc_score(y_true, y_scores)
            
            # Precision-Recall AUC
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
            metrics[f'{prefix}auc_pr'] = np.trapz(precision_vals, recall_vals)
        
        return metrics
    
    def evaluate_component_performance(self, 
                                     component_results: Dict[str, Any],
                                     y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate individual component performance.
        
        Args:
            component_results: Results from AIDM pipeline components
            y_true: True anomaly labels
            
        Returns:
            Dictionary of component evaluation results
        """
        evaluation = {}
        
        # Evaluate each component
        for component in ['autoencoder', 'transformations', 'lstm', 'fusion']:
            flag_key = f'{component}_flags'
            score_key = f'{component}_scores'
            
            if flag_key in component_results:
                y_pred = component_results[flag_key].astype(int)
                
                # Get scores if available
                y_scores = None
                if 'component_scores' in component_results and component in component_results['component_scores']:
                    y_scores = component_results['component_scores'][component]
                elif score_key in component_results:
                    y_scores = component_results[score_key]
                
                # Compute metrics
                metrics = self.compute_classification_metrics(
                    y_true, y_pred, y_scores, prefix=f'{component}_'
                )
                
                evaluation[component] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'scores': y_scores,
                    'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
                }
        
        return evaluation
    
    def evaluate_adversarial_robustness(self, 
                                      clean_results: Dict[str, Any],
                                      adversarial_results: Dict[str, Dict[str, Any]],
                                      y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate adversarial robustness of models.
        
        Args:
            clean_results: Results on clean data
            adversarial_results: Results on adversarial data (by attack type)
            y_true: True labels
            
        Returns:
            Robustness evaluation results
        """
        robustness = {}
        
        # Clean performance baseline
        clean_eval = self.evaluate_component_performance(clean_results, y_true)
        robustness['clean'] = clean_eval
        
        # Adversarial performance
        for attack_type, adv_results in adversarial_results.items():
            adv_eval = self.evaluate_component_performance(adv_results, y_true)
            robustness[attack_type] = adv_eval
            
            # Compute robustness metrics (performance drop)
            for component in adv_eval.keys():
                if component in clean_eval:
                    clean_acc = clean_eval[component]['metrics'].get(f'{component}_accuracy', 0)
                    adv_acc = adv_eval[component]['metrics'].get(f'{component}_accuracy', 0)
                    
                    robustness[attack_type][component]['robustness_drop'] = clean_acc - adv_acc
                    robustness[attack_type][component]['robustness_ratio'] = adv_acc / clean_acc if clean_acc > 0 else 0
        
        return robustness
    
    def plot_roc_curves(self, 
                       evaluation_results: Dict[str, Any],
                       y_true: np.ndarray,
                       save_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for all components.
        
        Args:
            evaluation_results: Component evaluation results
            y_true: True labels
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each component
        for component, results in evaluation_results.items():
            if 'scores' in results and results['scores'] is not None:
                y_scores = results['scores']
                
                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc_score = results['metrics'].get(f'{component}_auc_roc', 0)
                
                # Plot
                ax.plot(fpr, tpr, linewidth=2, 
                       label=f'{component.capitalize()} (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        # Formatting
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - AIDM Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curves(self, 
                                    evaluation_results: Dict[str, Any],
                                    y_true: np.ndarray,
                                    save_path: str = None) -> plt.Figure:
        """
        Plot Precision-Recall curves for all components.
        
        Args:
            evaluation_results: Component evaluation results
            y_true: True labels
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot PR curve for each component
        for component, results in evaluation_results.items():
            if 'scores' in results and results['scores'] is not None:
                y_scores = results['scores']
                
                # Compute PR curve
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                auc_pr = results['metrics'].get(f'{component}_auc_pr', 0)
                
                # Plot
                ax.plot(recall, precision, linewidth=2,
                       label=f'{component.capitalize()} (AUC = {auc_pr:.3f})')
        
        # Formatting
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves - AIDM Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curves saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self, 
                              evaluation_results: Dict[str, Any],
                              save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrices for all components.
        
        Args:
            evaluation_results: Component evaluation results
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        n_components = len(evaluation_results)
        n_cols = min(3, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_components == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (component, results) in enumerate(evaluation_results.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Get confusion matrix
            cm = np.array(results['confusion_matrix'])
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            
            ax.set_title(f'{component.capitalize()} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_components, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        return fig
    
    def plot_performance_comparison(self, 
                                  evaluation_results: Dict[str, Any],
                                  save_path: str = None) -> plt.Figure:
        """
        Plot performance comparison across components.
        
        Args:
            evaluation_results: Component evaluation results
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Extract metrics
        components = list(evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        data = []
        for component in components:
            for metric in metrics:
                metric_key = f'{component}_{metric}'
                if metric_key in evaluation_results[component]['metrics']:
                    value = evaluation_results[component]['metrics'][metric_key]
                    data.append({
                        'Component': component.capitalize(),
                        'Metric': metric.capitalize(),
                        'Value': value
                    })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.barplot(data=df, x='Metric', y='Value', hue='Component', ax=ax)
        
        ax.set_title('Performance Comparison - AIDM Components')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance comparison saved to {save_path}")
        
        return fig
    
    def plot_adversarial_robustness(self, 
                                   robustness_results: Dict[str, Any],
                                   save_path: str = None) -> plt.Figure:
        """
        Plot adversarial robustness analysis.
        
        Args:
            robustness_results: Robustness evaluation results
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Extract robustness data
        data = []
        clean_results = robustness_results.get('clean', {})
        
        for attack_type, attack_results in robustness_results.items():
            if attack_type == 'clean':
                continue
                
            for component, results in attack_results.items():
                if component in clean_results:
                    clean_acc = clean_results[component]['metrics'].get(f'{component}_accuracy', 0)
                    adv_acc = results['metrics'].get(f'{component}_accuracy', 0)
                    
                    data.append({
                        'Attack': attack_type,
                        'Component': component.capitalize(),
                        'Clean Accuracy': clean_acc,
                        'Adversarial Accuracy': adv_acc,
                        'Robustness Drop': clean_acc - adv_acc
                    })
        
        if not data:
            logger.warning("No adversarial robustness data available for plotting")
            return None
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Accuracy comparison
        df_melted = df.melt(
            id_vars=['Attack', 'Component'],
            value_vars=['Clean Accuracy', 'Adversarial Accuracy'],
            var_name='Condition',
            value_name='Accuracy'
        )
        
        sns.barplot(data=df_melted, x='Component', y='Accuracy', 
                   hue='Condition', ax=ax1)
        ax1.set_title('Clean vs Adversarial Accuracy')
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # Plot 2: Robustness drop
        sns.barplot(data=df, x='Component', y='Robustness Drop', 
                   hue='Attack', ax=ax2)
        ax2.set_title('Adversarial Robustness Drop')
        ax2.set_ylabel('Accuracy Drop')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Adversarial robustness plot saved to {save_path}")
        
        return fig
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, Any],
                                 y_true: np.ndarray,
                                 output_dir: str,
                                 experiment_name: str = "aidm_evaluation") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report with plots and metrics.
        
        Args:
            evaluation_results: Component evaluation results
            y_true: True labels
            output_dir: Output directory for reports
            experiment_name: Name of the experiment
            
        Returns:
            Summary report dictionary
        """
        output_path = Path(output_dir) / "reports"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        plots = {}
        
        # ROC curves
        roc_fig = self.plot_roc_curves(
            evaluation_results, y_true, 
            save_path=output_path / f"{experiment_name}_roc_curves.png"
        )
        plots['roc_curves'] = str(output_path / f"{experiment_name}_roc_curves.png")
        plt.close(roc_fig)
        
        # Precision-Recall curves
        pr_fig = self.plot_precision_recall_curves(
            evaluation_results, y_true,
            save_path=output_path / f"{experiment_name}_pr_curves.png"
        )
        plots['pr_curves'] = str(output_path / f"{experiment_name}_pr_curves.png")
        plt.close(pr_fig)
        
        # Confusion matrices
        cm_fig = self.plot_confusion_matrices(
            evaluation_results,
            save_path=output_path / f"{experiment_name}_confusion_matrices.png"
        )
        plots['confusion_matrices'] = str(output_path / f"{experiment_name}_confusion_matrices.png")
        plt.close(cm_fig)
        
        # Performance comparison
        perf_fig = self.plot_performance_comparison(
            evaluation_results,
            save_path=output_path / f"{experiment_name}_performance_comparison.png"
        )
        plots['performance_comparison'] = str(output_path / f"{experiment_name}_performance_comparison.png")
        plt.close(perf_fig)
        
        # Create summary report
        summary = {
            'experiment_name': experiment_name,
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true)),
            'components_evaluated': list(evaluation_results.keys()),
            'plots_generated': plots,
            'detailed_metrics': {}
        }
        
        # Add detailed metrics for each component
        for component, results in evaluation_results.items():
            summary['detailed_metrics'][component] = results['metrics']
        
        # Save summary report
        with open(output_path / f"{experiment_name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed results
        joblib.dump(evaluation_results, output_path / f"{experiment_name}_detailed_results.pkl")
        
        logger.info(f"Evaluation report generated: {output_path}")
        return summary
    
    def compare_models(self, 
                      model_results: Dict[str, Dict[str, Any]],
                      y_true: np.ndarray,
                      save_path: str = None) -> plt.Figure:
        """
        Compare performance across different models/configurations.
        
        Args:
            model_results: Results from different models
            y_true: True labels
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure
        """
        # Extract comparison data
        data = []
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        for model_name, results in model_results.items():
            for component, component_results in results.items():
                if 'metrics' in component_results:
                    for metric in metrics:
                        metric_key = f'{component}_{metric}'
                        if metric_key in component_results['metrics']:
                            value = component_results['metrics'][metric_key]
                            data.append({
                                'Model': model_name,
                                'Component': component.capitalize(),
                                'Metric': metric.upper(),
                                'Value': value
                            })
        
        df = pd.DataFrame(data)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        sns.barplot(data=df, x='Metric', y='Value', hue='Model', ax=ax)
        
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison saved to {save_path}")
        
        return fig


def create_evaluator(config: Dict[str, Any]) -> AIDMEvaluator:
    """
    Factory function to create an AIDM evaluator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AIDMEvaluator instance
    """
    return AIDMEvaluator(config)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc']
        }
    }
    
    # Create synthetic evaluation data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Simulate component results
    component_results = {
        'autoencoder_flags': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'transformations_flags': np.random.choice([0, 1], n_samples, p=[0.82, 0.18]),
        'lstm_flags': np.random.choice([0, 1], n_samples, p=[0.87, 0.13]),
        'fusion_flags': np.random.choice([0, 1], n_samples, p=[0.88, 0.12]),
        'component_scores': {
            'autoencoder': np.random.random(n_samples),
            'transformations': np.random.random(n_samples),
            'lstm': np.random.random(n_samples)
        }
    }
    
    # Create evaluator and run evaluation
    evaluator = create_evaluator(config)
    evaluation_results = evaluator.evaluate_component_performance(component_results, y_true)
    
    # Generate report
    summary = evaluator.generate_evaluation_report(
        evaluation_results, y_true, "./outputs", "example_evaluation"
    )
    
    print("Evaluation completed:")
    print(f"  Components evaluated: {summary['components_evaluated']}")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Plots generated: {len(summary['plots_generated'])}")
