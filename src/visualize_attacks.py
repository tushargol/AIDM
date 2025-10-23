"""
Advanced visualization tools for adversarial attack data in AIDM.
Provides comprehensive plotting and analysis capabilities for different attack types.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AttackVisualizer:
    """
    Comprehensive visualization toolkit for adversarial attack analysis.
    """
    
    def __init__(self, output_dir: str = "./outputs/reports"):
        """
        Initialize the attack visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotting parameters
        self.figsize = (12, 8)
        self.dpi = 300
        self.colors = {
            'clean': '#2E8B57',      # Sea Green
            'fdia': '#DC143C',       # Crimson
            'temporal_stealth': '#FF8C00',  # Dark Orange
            'replay': '#4B0082',     # Indigo
            'fgsm': '#FF1493',       # Deep Pink
            'pgd': '#8B0000'         # Dark Red
        }
    
    def load_attack_data(self, experiment_path: str) -> Dict[str, Any]:
        """
        Load attack dataset from experiment files.
        
        Args:
            experiment_path: Path to experiment files (without extension)
            
        Returns:
            Dictionary containing attack data and metadata
        """
        try:
            # Load attack data
            data_file = f"{experiment_path}_attacks.npz"
            data = np.load(data_file, allow_pickle=True)
            
            # Load metadata
            metadata_file = f"{experiment_path}_metadata.json"
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Convert to more convenient format
            attack_data = {
                'measurements': data['measurements'],
                'labels': data['labels'],
                'attack_types': data['attack_types'],
                'timestamps': pd.to_datetime(data['timestamps']),
                'metadata': metadata
            }
            
            logger.info(f"Loaded attack data: {attack_data['measurements'].shape}")
            return attack_data
            
        except Exception as e:
            logger.error(f"Failed to load attack data: {e}")
            raise
    
    def plot_attack_overview(self, attack_data: Dict[str, Any], save_path: str = None) -> go.Figure:
        """
        Create an overview plot showing attack distribution and timeline.
        
        Args:
            attack_data: Attack dataset
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        measurements = attack_data['measurements']
        labels = attack_data['labels']
        attack_types = attack_data['attack_types']
        timestamps = attack_data['timestamps']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Attack Distribution', 'Attack Timeline',
                'Measurement Statistics', 'Attack Type Distribution',
                'Temporal Attack Patterns', 'Attack Magnitude Analysis'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "box"}, {"type": "pie"}],
                [{"type": "heatmap"}, {"type": "histogram"}]
            ]
        )
        
        # 1. Attack distribution over time
        attack_counts = pd.Series(labels).rolling(window=100).sum()
        fig.add_trace(
            go.Scatter(x=timestamps, y=attack_counts, name='Attack Density',
                      line=dict(color=self.colors['fdia'])),
            row=1, col=2
        )
        
        # 2. Attack type distribution
        attack_type_counts = pd.Series(attack_types[labels == 1]).value_counts()
        fig.add_trace(
            go.Bar(x=attack_type_counts.index, y=attack_type_counts.values,
                  name='Attack Types', marker_color=self.colors['temporal_stealth']),
            row=1, col=1
        )
        
        # 3. Measurement statistics (clean vs attacked)
        clean_data = measurements[labels == 0]
        attack_data_vals = measurements[labels == 1]
        
        fig.add_trace(
            go.Box(y=clean_data.flatten(), name='Clean', marker_color=self.colors['clean']),
            row=2, col=1
        )
        fig.add_trace(
            go.Box(y=attack_data_vals.flatten(), name='Attacked', marker_color=self.colors['fdia']),
            row=2, col=1
        )
        
        # 4. Attack type pie chart
        fig.add_trace(
            go.Pie(labels=attack_type_counts.index, values=attack_type_counts.values,
                  name="Attack Types"),
            row=2, col=2
        )
        
        # 5. Temporal patterns (heatmap of attacks by hour and day)
        df_attacks = pd.DataFrame({
            'timestamp': timestamps[labels == 1],
            'attack_type': attack_types[labels == 1]
        })
        
        if len(df_attacks) > 0:
            df_attacks['hour'] = df_attacks['timestamp'].dt.hour
            df_attacks['day'] = df_attacks['timestamp'].dt.day
            heatmap_data = df_attacks.groupby(['day', 'hour']).size().unstack(fill_value=0)
            
            fig.add_trace(
                go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
                          colorscale='Reds', name='Attack Heatmap'),
                row=3, col=1
            )
        
        # 6. Attack magnitude distribution
        attack_magnitudes = np.linalg.norm(attack_data_vals - clean_data[:len(attack_data_vals)], axis=1)
        fig.add_trace(
            go.Histogram(x=attack_magnitudes, name='Attack Magnitudes',
                        marker_color=self.colors['replay']),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Adversarial Attack Analysis Overview",
            height=1200,
            showlegend=True
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved attack overview to {save_path}")
        
        return fig
    
    def plot_attack_timeseries(self, attack_data: Dict[str, Any], 
                              measurement_indices: List[int] = None,
                              time_window: Tuple[int, int] = None,
                              save_path: str = None) -> go.Figure:
        """
        Plot time series showing clean vs attacked measurements.
        
        Args:
            attack_data: Attack dataset
            measurement_indices: Which measurements to plot (default: first 3)
            time_window: Time window to focus on (start_idx, end_idx)
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        measurements = attack_data['measurements']
        labels = attack_data['labels']
        attack_types = attack_data['attack_types']
        timestamps = attack_data['timestamps']
        
        # Default parameters
        if measurement_indices is None:
            measurement_indices = list(range(min(3, measurements.shape[1])))
        
        if time_window is None:
            time_window = (0, min(1000, len(measurements)))
        
        start_idx, end_idx = time_window
        
        # Create subplots for each measurement
        fig = make_subplots(
            rows=len(measurement_indices), cols=1,
            subplot_titles=[f'Measurement {i}' for i in measurement_indices],
            shared_xaxes=True
        )
        
        for i, meas_idx in enumerate(measurement_indices):
            row = i + 1
            
            # Extract data for this measurement
            y_data = measurements[start_idx:end_idx, meas_idx]
            x_data = timestamps[start_idx:end_idx]
            labels_window = labels[start_idx:end_idx]
            attack_types_window = attack_types[start_idx:end_idx]
            
            # Plot clean data
            clean_mask = labels_window == 0
            fig.add_trace(
                go.Scatter(
                    x=x_data[clean_mask], 
                    y=y_data[clean_mask],
                    mode='lines+markers',
                    name=f'Clean {meas_idx}' if i == 0 else None,
                    line=dict(color=self.colors['clean'], width=2),
                    marker=dict(size=3),
                    showlegend=(i == 0)
                ),
                row=row, col=1
            )
            
            # Plot attacked data by type
            attack_mask = labels_window == 1
            if np.any(attack_mask):
                unique_attacks = np.unique(attack_types_window[attack_mask])
                
                for attack_type in unique_attacks:
                    if attack_type == 'clean':
                        continue
                    
                    type_mask = (labels_window == 1) & (attack_types_window == attack_type)
                    if np.any(type_mask):
                        fig.add_trace(
                            go.Scatter(
                                x=x_data[type_mask],
                                y=y_data[type_mask],
                                mode='markers',
                                name=f'{attack_type.title()}' if i == 0 else None,
                                marker=dict(
                                    color=self.colors.get(attack_type, '#FF69B4'),
                                    size=6,
                                    symbol='diamond'
                                ),
                                showlegend=(i == 0)
                            ),
                            row=row, col=1
                        )
        
        # Update layout
        fig.update_layout(
            title_text="Time Series Analysis: Clean vs Attacked Measurements",
            height=300 * len(measurement_indices),
            xaxis_title="Time",
            showlegend=True
        )
        
        # Update y-axis labels
        for i in range(len(measurement_indices)):
            fig.update_yaxes(title_text="Value", row=i+1, col=1)
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved timeseries plot to {save_path}")
        
        return fig
    
    def plot_attack_detection_analysis(self, attack_data: Dict[str, Any],
                                     detector_scores: Dict[str, np.ndarray] = None,
                                     save_path: str = None) -> go.Figure:
        """
        Analyze attack detection performance across different methods.
        
        Args:
            attack_data: Attack dataset
            detector_scores: Dictionary of detector names to anomaly scores
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        measurements = attack_data['measurements']
        labels = attack_data['labels']
        attack_types = attack_data['attack_types']
        
        # If no detector scores provided, compute simple statistical measures
        if detector_scores is None:
            detector_scores = self._compute_simple_detectors(measurements)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'ROC Curves', 'Detection Score Distributions',
                'Attack Type Detection Rates', 'Confusion Matrix'
            ]
        )
        
        # 1. ROC Curves
        from sklearn.metrics import roc_curve, auc
        
        for detector_name, scores in detector_scores.items():
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{detector_name} (AUC={roc_auc:.3f})',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                      line=dict(dash='dash', color='gray'),
                      name='Random', showlegend=False),
            row=1, col=1
        )
        
        # 2. Score distributions
        for detector_name, scores in detector_scores.items():
            fig.add_trace(
                go.Histogram(
                    x=scores[labels == 0], 
                    name=f'{detector_name} Clean',
                    opacity=0.7,
                    histnorm='probability'
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Histogram(
                    x=scores[labels == 1], 
                    name=f'{detector_name} Attack',
                    opacity=0.7,
                    histnorm='probability'
                ),
                row=1, col=2
            )
        
        # 3. Attack type detection rates
        attack_detection_rates = {}
        for attack_type in np.unique(attack_types[labels == 1]):
            if attack_type == 'clean':
                continue
            
            type_mask = (labels == 1) & (attack_types == attack_type)
            if np.any(type_mask):
                rates = {}
                for detector_name, scores in detector_scores.items():
                    # Use median score as threshold
                    threshold = np.median(scores[labels == 0])
                    detection_rate = np.mean(scores[type_mask] > threshold)
                    rates[detector_name] = detection_rate
                attack_detection_rates[attack_type] = rates
        
        # Plot detection rates
        detector_names = list(detector_scores.keys())
        attack_type_names = list(attack_detection_rates.keys())
        
        for i, detector_name in enumerate(detector_names):
            rates = [attack_detection_rates[at][detector_name] for at in attack_type_names]
            fig.add_trace(
                go.Bar(
                    x=attack_type_names,
                    y=rates,
                    name=detector_name,
                    offsetgroup=i
                ),
                row=2, col=1
            )
        
        # 4. Confusion matrix for best detector
        if detector_scores:
            best_detector = max(detector_scores.keys(), 
                              key=lambda x: auc(*roc_curve(labels, detector_scores[x])[:2]))
            best_scores = detector_scores[best_detector]
            threshold = np.median(best_scores[labels == 0])
            predictions = (best_scores > threshold).astype(int)
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(labels, predictions)
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['Predicted Clean', 'Predicted Attack'],
                    y=['True Clean', 'True Attack'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16}
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Attack Detection Analysis",
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Score", row=1, col=2)
        fig.update_yaxes(title_text="Probability", row=1, col=2)
        fig.update_xaxes(title_text="Attack Type", row=2, col=1)
        fig.update_yaxes(title_text="Detection Rate", row=2, col=1)
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved detection analysis to {save_path}")
        
        return fig
    
    def plot_attack_dimensionality_analysis(self, attack_data: Dict[str, Any],
                                          save_path: str = None) -> go.Figure:
        """
        Visualize attacks in reduced dimensional space using PCA and t-SNE.
        
        Args:
            attack_data: Attack dataset
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        measurements = attack_data['measurements']
        labels = attack_data['labels']
        attack_types = attack_data['attack_types']
        
        # Subsample for computational efficiency
        n_samples = min(2000, len(measurements))
        indices = np.random.choice(len(measurements), n_samples, replace=False)
        
        X = measurements[indices]
        y = labels[indices]
        attack_types_sub = attack_types[indices]
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})',
                't-SNE',
                'PCA by Attack Type',
                't-SNE by Attack Type'
            ]
        )
        
        # Color mapping
        colors_clean_attack = {0: self.colors['clean'], 1: self.colors['fdia']}
        
        # 1. PCA - Clean vs Attack
        for label in [0, 1]:
            mask = y == label
            fig.add_trace(
                go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name='Clean' if label == 0 else 'Attack',
                    marker=dict(
                        color=colors_clean_attack[label],
                        size=5,
                        opacity=0.7
                    )
                ),
                row=1, col=1
            )
        
        # 2. t-SNE - Clean vs Attack
        for label in [0, 1]:
            mask = y == label
            fig.add_trace(
                go.Scatter(
                    x=X_tsne[mask, 0],
                    y=X_tsne[mask, 1],
                    mode='markers',
                    name='Clean' if label == 0 else 'Attack',
                    marker=dict(
                        color=colors_clean_attack[label],
                        size=5,
                        opacity=0.7
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. PCA by attack type
        unique_types = np.unique(attack_types_sub)
        for attack_type in unique_types:
            mask = attack_types_sub == attack_type
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(
                        x=X_pca[mask, 0],
                        y=X_pca[mask, 1],
                        mode='markers',
                        name=attack_type.title(),
                        marker=dict(
                            color=self.colors.get(attack_type, '#FF69B4'),
                            size=5,
                            opacity=0.7
                        )
                    ),
                    row=2, col=1
                )
        
        # 4. t-SNE by attack type
        for attack_type in unique_types:
            mask = attack_types_sub == attack_type
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(
                        x=X_tsne[mask, 0],
                        y=X_tsne[mask, 1],
                        mode='markers',
                        name=attack_type.title(),
                        marker=dict(
                            color=self.colors.get(attack_type, '#FF69B4'),
                            size=5,
                            opacity=0.7
                        ),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Dimensionality Reduction Analysis of Attacks",
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)
        fig.update_xaxes(title_text="PC1", row=2, col=1)
        fig.update_yaxes(title_text="PC2", row=2, col=1)
        fig.update_xaxes(title_text="t-SNE 1", row=2, col=2)
        fig.update_yaxes(title_text="t-SNE 2", row=2, col=2)
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved dimensionality analysis to {save_path}")
        
        return fig
    
    def _compute_simple_detectors(self, measurements: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute simple statistical detectors for demonstration."""
        detectors = {}
        
        # Statistical detectors
        detectors['L2_Norm'] = np.linalg.norm(measurements, axis=1)
        detectors['Max_Deviation'] = np.max(np.abs(measurements - np.mean(measurements, axis=0)), axis=1)
        detectors['Variance'] = np.var(measurements, axis=1)
        
        # Temporal detectors (if enough samples)
        if len(measurements) > 1:
            diff = np.diff(measurements, axis=0)
            temporal_scores = np.zeros(len(measurements))
            temporal_scores[1:] = np.linalg.norm(diff, axis=1)
            detectors['Temporal_Change'] = temporal_scores
        
        return detectors
    
    def create_interactive_dashboard(self, attack_data: Dict[str, Any],
                                   detector_scores: Dict[str, np.ndarray] = None,
                                   save_path: str = None) -> str:
        """
        Create a comprehensive interactive dashboard.
        
        Args:
            attack_data: Attack dataset
            detector_scores: Dictionary of detector scores
            save_path: Path to save the dashboard HTML
            
        Returns:
            Path to the saved dashboard
        """
        if save_path is None:
            save_path = self.output_dir / "attack_analysis_dashboard.html"
        
        # Generate all plots
        overview_fig = self.plot_attack_overview(attack_data)
        timeseries_fig = self.plot_attack_timeseries(attack_data)
        detection_fig = self.plot_attack_detection_analysis(attack_data, detector_scores)
        dimensionality_fig = self.plot_attack_dimensionality_analysis(attack_data)
        
        # Create HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AIDM Attack Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .plot-container {{ margin-bottom: 40px; }}
                .metadata {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AIDM Adversarial Attack Analysis Dashboard</h1>
                <p>Comprehensive visualization of attack patterns and detection performance</p>
            </div>
            
            <div class="metadata">
                <h3>Dataset Information</h3>
                <p><strong>Total Samples:</strong> {attack_data['metadata']['total_samples']}</p>
                <p><strong>Attack Samples:</strong> {attack_data['metadata']['attack_samples']}</p>
                <p><strong>Attack Ratio:</strong> {attack_data['metadata']['attack_ratio']:.2%}</p>
                <p><strong>Attack Types:</strong> {', '.join(attack_data['metadata']['attack_types_used'])}</p>
            </div>
            
            <div class="plot-container">
                <h2>Attack Overview</h2>
                <div id="overview-plot"></div>
            </div>
            
            <div class="plot-container">
                <h2>Time Series Analysis</h2>
                <div id="timeseries-plot"></div>
            </div>
            
            <div class="plot-container">
                <h2>Detection Analysis</h2>
                <div id="detection-plot"></div>
            </div>
            
            <div class="plot-container">
                <h2>Dimensionality Analysis</h2>
                <div id="dimensionality-plot"></div>
            </div>
            
            <script>
                // Plot the figures
                var overviewConfig = {{responsive: true}};
                var timeseriesConfig = {{responsive: true}};
                var detectionConfig = {{responsive: true}};
                var dimensionalityConfig = {{responsive: true}};
                
                Plotly.newPlot('overview-plot', {overview_fig.to_json()}, overviewConfig);
                Plotly.newPlot('timeseries-plot', {timeseries_fig.to_json()}, timeseriesConfig);
                Plotly.newPlot('detection-plot', {detection_fig.to_json()}, detectionConfig);
                Plotly.newPlot('dimensionality-plot', {dimensionality_fig.to_json()}, dimensionalityConfig);
            </script>
        </body>
        </html>
        """
        
        # Save dashboard
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created interactive dashboard: {save_path}")
        return str(save_path)


def create_attack_visualizer(output_dir: str = "./outputs/reports") -> AttackVisualizer:
    """
    Factory function to create an AttackVisualizer instance.
    
    Args:
        output_dir: Directory for saving outputs
        
    Returns:
        AttackVisualizer instance
    """
    return AttackVisualizer(output_dir)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create visualizer
    visualizer = create_attack_visualizer()
    
    # Example: Load and visualize attack data
    try:
        # This would load actual attack data
        experiment_path = "./outputs/experiments/comprehensive_attacks"
        attack_data = visualizer.load_attack_data(experiment_path)
        
        # Create visualizations
        overview_fig = visualizer.plot_attack_overview(attack_data)
        timeseries_fig = visualizer.plot_attack_timeseries(attack_data)
        detection_fig = visualizer.plot_attack_detection_analysis(attack_data)
        dimensionality_fig = visualizer.plot_attack_dimensionality_analysis(attack_data)
        
        # Create dashboard
        dashboard_path = visualizer.create_interactive_dashboard(attack_data)
        print(f"Dashboard created: {dashboard_path}")
        
    except Exception as e:
        print(f"Example failed (expected if no attack data exists): {e}")
        print("Generate attack data first using: python src/attacks.py --type all")
