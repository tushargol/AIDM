"""
Data preprocessing pipeline for the AIDM system.
Handles data alignment, resampling, feature extraction, and sequence creation.
"""

import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import click
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_sample_data

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for power system measurements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scalers = {}
        self.feature_names = []
        
    def align_and_resample(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Align multiple data modalities to a common time base and resample.
        
        Args:
            data: Dictionary of DataFrames with different modalities
            
        Returns:
            Aligned and resampled DataFrame
        """
        base_rate = self.config['preprocessing']['base_sampling_rate']
        
        # Find common time range
        start_times = [df.index.min() for df in data.values() if not df.empty]
        end_times = [df.index.max() for df in data.values() if not df.empty]
        
        if not start_times or not end_times:
            raise ValueError("No valid data found for alignment")
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        logger.info(f"Aligning data from {common_start} to {common_end}")
        
        # Create common time index
        freq_str = f'{base_rate}S'
        common_index = pd.date_range(start=common_start, end=common_end, freq=freq_str)
        
        aligned_data = []
        
        for modality, df in data.items():
            if df.empty:
                continue
                
            logger.info(f"Processing {modality} data: {df.shape}")
            
            # Resample to common frequency
            if modality == 'topology':
                # Forward fill for discrete topology data
                resampled = df.reindex(common_index, method='ffill')
            else:
                # Interpolate for continuous measurements
                resampled = df.reindex(common_index).interpolate(method='linear')
            
            # Add modality prefix to column names
            resampled.columns = [f'{modality}_{col}' for col in resampled.columns]
            aligned_data.append(resampled)
        
        # Combine all modalities
        if aligned_data:
            combined_df = pd.concat(aligned_data, axis=1)
            logger.info(f"Combined data shape: {combined_df.shape}")
            return combined_df
        else:
            raise ValueError("No data available after alignment")
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data with forward fill and gap detection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing data handled
        """
        max_gap = self.config['preprocessing']['max_gap_fill']
        
        # Forward fill small gaps
        df_filled = df.fillna(method='ffill', limit=max_gap)
        
        # Log remaining missing data
        missing_counts = df_filled.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Remaining missing values: {missing_counts.sum()}")
            # Drop rows with any remaining missing values
            df_filled = df_filled.dropna()
        
        return df_filled
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract engineered features from raw measurements.
        
        Args:
            df: Input DataFrame with aligned measurements
            
        Returns:
            DataFrame with engineered features
        """
        feature_df = df.copy()
        
        # Extract voltage magnitudes (assuming they exist)
        voltage_cols = [col for col in df.columns if 'voltage_mag' in col]
        
        # Extract phasor angles and convert to sin/cos components
        angle_cols = [col for col in df.columns if 'voltage_angle' in col]
        for col in angle_cols:
            base_name = col.replace('_angle', '')
            feature_df[f'{base_name}_sin'] = np.sin(df[col])
            feature_df[f'{base_name}_cos'] = np.cos(df[col])
        
        # Rolling statistics if enabled
        if self.config['features']['enable_rolling_stats']:
            window = self.config['preprocessing']['rolling_window']
            
            for col in voltage_cols:
                base_name = col.replace('_voltage_mag', '')
                feature_df[f'{base_name}_voltage_mean'] = df[col].rolling(window=window).mean()
                feature_df[f'{base_name}_voltage_std'] = df[col].rolling(window=window).std()
                feature_df[f'{base_name}_voltage_delta'] = df[col].diff()
        
        # Topology features if enabled
        if self.config['features']['enable_topology_features']:
            topology_cols = [col for col in df.columns if 'topology_' in col]
            # One-hot encode if categorical, otherwise use as-is
            for col in topology_cols:
                if df[col].dtype == 'object':
                    # One-hot encode categorical topology features
                    dummies = pd.get_dummies(df[col], prefix=col)
                    feature_df = pd.concat([feature_df, dummies], axis=1)
                    feature_df.drop(col, axis=1, inplace=True)
        
        # Frequency features if enabled
        if self.config['features'].get('enable_frequency_features', False):
            frequency_cols = [col for col in df.columns if 'frequency_' in col or col in ['system_frequency', 'rocof']]
            for col in frequency_cols:
                if col in df.columns:
                    # Add frequency deviation features
                    if 'frequency' in col and 'rocof' not in col:
                        nominal_freq = 60.0  # Hz (could be configurable)
                        feature_df[f'{col}_deviation'] = df[col] - nominal_freq
                        feature_df[f'{col}_deviation_abs'] = np.abs(df[col] - nominal_freq)
                    
                    # Add ROCOF magnitude for stability analysis
                    if 'rocof' in col:
                        feature_df[f'{col}_magnitude'] = np.abs(df[col])
        
        # Waveform features (placeholder - would extract FFT components in real implementation)
        waveform_cols = [col for col in df.columns if 'waveform_' in col]
        if waveform_cols and self.config['features']['waveform_fft_components'] > 0:
            n_fft = self.config['features']['waveform_fft_components']
            for col in waveform_cols:
                # Placeholder: in real implementation, would compute FFT of waveform segments
                for i in range(n_fft):
                    feature_df[f'{col}_fft_{i}'] = np.random.randn(len(df)) * 0.1
        
        # Drop original angle columns (replaced by sin/cos)
        feature_df.drop(angle_cols, axis=1, inplace=True, errors='ignore')
        
        # Remove any remaining NaN values from feature engineering
        feature_df = feature_df.dropna()
        
        logger.info(f"Extracted features shape: {feature_df.shape}")
        self.feature_names = list(feature_df.columns)
        
        return feature_df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create LSTM sequences from the feature DataFrame.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Tuple of (X_sequences, y_next, X_tabular)
        """
        window = self.config['preprocessing']['sequence_window']
        
        if len(df) < window + 1:
            raise ValueError(f"Not enough data for sequence creation. Need at least {window + 1} samples, got {len(df)}")
        
        # Convert to numpy for efficient slicing
        data = df.values
        
        # Create sequences
        X_sequences = []
        y_next = []
        X_tabular = []
        
        for i in range(window, len(data)):
            # Sequence of past window timesteps
            X_sequences.append(data[i-window:i])
            # Next timestep target
            y_next.append(data[i])
            # Current timestep features (for tabular models)
            X_tabular.append(data[i])
        
        X_sequences = np.array(X_sequences)
        y_next = np.array(y_next)
        X_tabular = np.array(X_tabular)
        
        logger.info(f"Created sequences: X_seq {X_sequences.shape}, y_next {y_next.shape}, X_tab {X_tabular.shape}")
        
        return X_sequences, y_next, X_tabular
    
    def fit_scalers(self, X_train: np.ndarray, X_seq_train: np.ndarray) -> None:
        """
        Fit scalers on training data.
        
        Args:
            X_train: Training tabular features
            X_seq_train: Training sequence features
        """
        # Scaler for tabular features
        self.scalers['tabular'] = StandardScaler()
        self.scalers['tabular'].fit(X_train)
        
        # Scaler for sequence features (fit on flattened sequences)
        self.scalers['sequence'] = StandardScaler()
        X_seq_flat = X_seq_train.reshape(-1, X_seq_train.shape[-1])
        self.scalers['sequence'].fit(X_seq_flat)
        
        logger.info("Fitted scalers on training data")
    
    def transform_data(self, X_tabular: np.ndarray, X_sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted scalers.
        
        Args:
            X_tabular: Tabular features
            X_sequences: Sequence features
            
        Returns:
            Tuple of scaled (X_tabular, X_sequences)
        """
        # Scale tabular features
        X_tab_scaled = self.scalers['tabular'].transform(X_tabular)
        
        # Scale sequence features
        original_shape = X_sequences.shape
        X_seq_flat = X_sequences.reshape(-1, X_sequences.shape[-1])
        X_seq_scaled_flat = self.scalers['sequence'].transform(X_seq_flat)
        X_seq_scaled = X_seq_scaled_flat.reshape(original_shape)
        
        return X_tab_scaled, X_seq_scaled
    
    def split_data(self, X_tabular: np.ndarray, X_sequences: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split data into train/validation/test sets with temporal ordering.
        
        Args:
            X_tabular: Tabular features
            X_sequences: Sequence features  
            y: Target values
            
        Returns:
            Dictionary containing split datasets
        """
        train_split = self.config['preprocessing']['train_split']
        val_split = self.config['preprocessing']['val_split']
        
        n_samples = len(X_tabular)
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))
        
        # Temporal split (no shuffling to preserve time order)
        splits = {
            'X_train': X_tabular[:train_end],
            'X_val': X_tabular[train_end:val_end],
            'X_test': X_tabular[val_end:],
            'X_seq_train': X_sequences[:train_end],
            'X_seq_val': X_sequences[train_end:val_end],
            'X_seq_test': X_sequences[val_end:],
            'y_train': y[:train_end],
            'y_val': y[train_end:val_end],
            'y_test': y[val_end:]
        }
        
        logger.info(f"Data splits - Train: {len(splits['X_train'])}, Val: {len(splits['X_val'])}, Test: {len(splits['X_test'])}")
        
        return splits
    
    def save_processed_data(self, splits: Dict[str, np.ndarray], output_path: str) -> None:
        """
        Save processed data and scalers to disk.
        
        Args:
            splits: Dictionary containing split datasets
            output_path: Output directory path
        """
        output_dir = Path(output_path) / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data splits
        np.savez_compressed(
            output_dir / "processed_data.npz",
            **splits,
            feature_names=self.feature_names
        )
        
        # Save scalers
        joblib.dump(self.scalers, output_dir / "scalers.pkl")
        
        # Save preprocessing config
        with open(output_dir / "preprocessing_config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Saved processed data to {output_dir}")
    
    def run_full_pipeline(self, data: Dict[str, pd.DataFrame], output_path: str) -> Dict[str, np.ndarray]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            data: Raw data dictionary
            output_path: Output directory path
            
        Returns:
            Dictionary containing processed data splits
        """
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Align and resample data
        aligned_df = self.align_and_resample(data)
        
        # Step 2: Handle missing data
        clean_df = self.handle_missing_data(aligned_df)
        
        # Step 3: Feature engineering
        feature_df = self.extract_features(clean_df)
        
        # Step 4: Create sequences
        X_sequences, y_next, X_tabular = self.create_sequences(feature_df)
        
        # Step 5: Split data
        splits = self.split_data(X_tabular, X_sequences, y_next)
        
        # Step 6: Fit scalers on training data
        self.fit_scalers(splits['X_train'], splits['X_seq_train'])
        
        # Step 7: Transform all data
        for split_name in ['train', 'val', 'test']:
            X_tab_key = f'X_{split_name}'
            X_seq_key = f'X_seq_{split_name}'
            
            splits[X_tab_key], splits[X_seq_key] = self.transform_data(
                splits[X_tab_key], splits[X_seq_key]
            )
        
        # Step 8: Save processed data
        self.save_processed_data(splits, output_path)
        
        logger.info("Preprocessing pipeline completed successfully")
        return splits


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@click.command()
@click.option('--config', default='config.yaml', help='Path to configuration file')
@click.option('--mode', default='small_data', type=click.Choice(['small_data', 'full_data']), 
              help='Processing mode')
@click.option('--output', default=None, help='Output directory (overrides config)')
def main(config: str, mode: str, output: str):
    """
    Run the preprocessing pipeline.
    
    Example usage:
        python src/preprocess.py --mode small_data
        python src/preprocess.py --config custom_config.yaml --output ./custom_output
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_dict = load_config(config)
    
    # Override output path if specified
    if output:
        config_dict['data']['output_path'] = output
    
    # Set small data mode
    config_dict['compute']['small_data_mode'] = (mode == 'small_data')
    
    try:
        # Load data
        dataset_path = config_dict['data']['dataset_path']
        logger.info(f"Loading data from {dataset_path}")
        
        data = load_sample_data(
            dataset_path, 
            small_data_mode=config_dict['compute']['small_data_mode']
        )
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config_dict)
        
        # Run preprocessing pipeline
        splits = preprocessor.run_full_pipeline(data, config_dict['data']['output_path'])
        
        # Print summary
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Output directory: {config_dict['data']['output_path']}/processed/")
        print(f"Feature count: {len(preprocessor.feature_names)}")
        print(f"Training samples: {len(splits['X_train'])}")
        print(f"Validation samples: {len(splits['X_val'])}")
        print(f"Test samples: {len(splits['X_test'])}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
