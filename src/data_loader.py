"""
Data loading utilities for the digital twin dataset.
Supports loading magnitudes, phasors, waveforms, and topology data.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import glob
import json

# Try to import the API client if available
try:
    import sys
    sys.path.append('./digital-twin-dataset/digital-twin-dataset')
    from dataset_api_client import DatasetApiClient
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False
    logging.warning("DatasetApiClient not available. Only local sample data will be accessible.")

logger = logging.getLogger(__name__)


class DigitalTwinDataLoader:
    """
    Data loader for the digital twin dataset supporting both local sample data
    and API-based access to the full dataset.
    """
    
    def __init__(self, dataset_path: str, use_api: bool = False):
        """
        Initialize the data loader.
        
        Args:
            dataset_path: Path to the local dataset directory
            use_api: Whether to use API client for larger dataset access
        """
        self.dataset_path = Path(dataset_path)
        self.sample_path = self.dataset_path / "sample_dataset"
        self.use_api = use_api and API_CLIENT_AVAILABLE
        
        if self.use_api:
            self.api_client = DatasetApiClient()
        else:
            self.api_client = None
            
        # Validate paths
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        if not self.sample_path.exists():
            raise FileNotFoundError(f"Sample dataset path not found: {self.sample_path}")
    
    def list_available_files(self, modality: str = "all") -> Dict[str, List[str]]:
        """
        List available data files in the sample dataset.
        
        Args:
            modality: Type of data to list ("magnitude", "phasor", "waveform", "topology", "all")
            
        Returns:
            Dictionary mapping modality to list of available files
        """
        available_files = {}
        
        # Note: Based on the repo structure, the sample dataset mainly contains topology data
        # For a complete implementation, we would need the actual magnitude/phasor/waveform files
        
        if modality in ["topology", "all"]:
            topology_path = self.sample_path / "topology"
            if topology_path.exists():
                # List parameter timeseries files
                param_path = topology_path / "parameter_timeseries"
                if param_path.exists():
                    topology_files = list(param_path.glob("*.csv"))
                    available_files["topology"] = [str(f.relative_to(param_path)) for f in topology_files]
                
                # List network files
                network_path = topology_path / "network_files"
                if network_path.exists():
                    network_files = list(network_path.glob("*.json"))
                    available_files["network"] = [str(f.relative_to(network_path)) for f in network_files]
        
        # Placeholder for other modalities (would be implemented based on actual file structure)
        for mod in ["magnitude", "phasor", "waveform"]:
            if modality in [mod, "all"]:
                # In a real implementation, these would point to actual data directories
                available_files[mod] = []
                logger.warning(f"No {mod} files found in sample dataset. Use API client for full dataset access.")
        
        return available_files
    
    def load_timeseries(self, 
                       file_path: str, 
                       modality: str,
                       start_time: Optional[pd.Timestamp] = None,
                       end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Load timeseries data from a file.
        
        Args:
            file_path: Path to the data file (relative to sample dataset)
            modality: Type of data ("magnitude", "phasor", "waveform", "topology")
            start_time: Start time for filtering (optional)
            end_time: End time for filtering (optional)
            
        Returns:
            DataFrame with timestamp index and measurement columns
        """
        if modality == "topology":
            return self._load_topology_timeseries(file_path, start_time, end_time)
        elif modality in ["magnitude", "phasor", "waveform"]:
            # Placeholder implementation - would be customized based on actual file formats
            return self._load_measurement_timeseries(file_path, modality, start_time, end_time)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def _load_topology_timeseries(self, 
                                 file_path: str,
                                 start_time: Optional[pd.Timestamp] = None,
                                 end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Load topology parameter timeseries data."""
        full_path = self.sample_path / "topology" / "parameter_timeseries" / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Topology file not found: {full_path}")
        
        # Load CSV file
        df = pd.read_csv(full_path)
        
        # Convert timestamp column to datetime (assuming first column is timestamp)
        if len(df.columns) > 0:
            timestamp_col = df.columns[0]
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df.set_index(timestamp_col, inplace=True)
        
        # Filter by time range if specified
        if start_time is not None:
            df = df[df.index >= start_time]
        if end_time is not None:
            df = df[df.index <= end_time]
        
        return df
    
    def _load_measurement_timeseries(self, 
                                   file_path: str,
                                   modality: str,
                                   start_time: Optional[pd.Timestamp] = None,
                                   end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Load measurement timeseries data (magnitude, phasor, waveform).
        This is a placeholder implementation.
        """
        # In a real implementation, this would handle the specific file formats
        # for magnitude, phasor, and waveform data
        logger.warning(f"Loading {modality} data from {file_path} - placeholder implementation")
        
        # Return empty DataFrame with proper structure
        timestamps = pd.date_range(start='2024-01-01', periods=100, freq='1S')
        data = np.random.randn(100, 5)  # Placeholder data
        df = pd.DataFrame(data, index=timestamps, columns=[f'{modality}_{i}' for i in range(5)])
        
        # Filter by time range if specified
        if start_time is not None:
            df = df[df.index >= start_time]
        if end_time is not None:
            df = df[df.index <= end_time]
        
        return df
    
    def load_network_topology(self, network_file: str = None) -> Dict:
        """
        Load network topology information.
        
        Args:
            network_file: Specific network file to load (optional)
            
        Returns:
            Dictionary containing network topology data
        """
        network_path = self.sample_path / "topology" / "network_files"
        
        if network_file is None:
            # Find the first available network file
            network_files = list(network_path.glob("*.json"))
            if not network_files:
                raise FileNotFoundError("No network files found")
            network_file = network_files[0].name
        
        full_path = network_path / network_file
        
        if not full_path.exists():
            raise FileNotFoundError(f"Network file not found: {full_path}")
        
        with open(full_path, 'r') as f:
            network_data = json.load(f)
        
        return network_data
    
    def generate_synthetic_data(self, 
                              duration_hours: float = 1.0,
                              sampling_rate: float = 1.0,
                              n_buses: int = 12,
                              noise_level: float = 0.01) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic data for testing when real data is not available.
        
        Args:
            duration_hours: Duration of synthetic data in hours
            sampling_rate: Sampling rate in Hz
            n_buses: Number of buses in the synthetic system
            noise_level: Noise level for synthetic measurements
            
        Returns:
            Dictionary containing synthetic timeseries data
        """
        n_samples = int(duration_hours * 3600 * sampling_rate)
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq=f'{1/sampling_rate}S')
        
        # Generate synthetic voltage magnitudes (around 1.0 pu with variations)
        base_voltages = 1.0 + 0.05 * np.sin(np.linspace(0, 4*np.pi, n_samples))[:, np.newaxis]
        voltage_noise = noise_level * np.random.randn(n_samples, n_buses)
        voltages = base_voltages + voltage_noise
        
        # Generate synthetic phasor angles
        base_angles = np.linspace(0, 2*np.pi/3, n_buses)  # Phase differences
        angle_variations = 0.1 * np.sin(np.linspace(0, 2*np.pi, n_samples))[:, np.newaxis]
        angles = base_angles + angle_variations + noise_level * np.random.randn(n_samples, n_buses)
        
        # Create DataFrames
        magnitude_df = pd.DataFrame(
            voltages, 
            index=timestamps, 
            columns=[f'bus_{i}_voltage_mag' for i in range(n_buses)]
        )
        
        phasor_df = pd.DataFrame(
            angles, 
            index=timestamps, 
            columns=[f'bus_{i}_voltage_angle' for i in range(n_buses)]
        )
        
        # Generate synthetic topology data (breaker status)
        breaker_status = np.random.choice([0, 1], size=(n_samples, n_buses//2), p=[0.1, 0.9])
        topology_df = pd.DataFrame(
            breaker_status,
            index=timestamps,
            columns=[f'breaker_{i}_status' for i in range(n_buses//2)]
        )
        
        return {
            'magnitude': magnitude_df,
            'phasor': phasor_df,
            'topology': topology_df
        }
    
    def load_api_data(self, 
                     magnitudes_for: List[str] = None,
                     phasors_for: List[str] = None,
                     waveforms_for: List[str] = None,
                     time_range: Tuple = None,
                     resolution: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load data using the API client (requires authentication).
        
        Args:
            magnitudes_for: List of element names for magnitude data
            phasors_for: List of element names for phasor data
            waveforms_for: List of element names for waveform data
            time_range: Tuple of (start_time, end_time)
            resolution: Time resolution string
            
        Returns:
            Dictionary containing loaded data
        """
        if not self.use_api or self.api_client is None:
            raise RuntimeError("API client not available or not enabled")
        
        # Use API client to download data
        data = self.api_client.download_data(
            magnitudes_for=magnitudes_for or [],
            phasors_for=phasors_for or [],
            waveforms_for=waveforms_for or [],
            time_range=time_range,
            resolution=resolution
        )
        
        return data


def load_sample_data(dataset_path: str, 
                    small_data_mode: bool = True,
                    synthetic_fallback: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load sample data for development and testing.
    
    Args:
        dataset_path: Path to the dataset
        small_data_mode: Whether to use only a small subset of data
        synthetic_fallback: Whether to generate synthetic data if real data unavailable
        
    Returns:
        Dictionary containing loaded timeseries data
    """
    loader = DigitalTwinDataLoader(dataset_path)
    
    try:
        # Try to load real topology data
        available_files = loader.list_available_files()
        data = {}
        
        if available_files.get('topology'):
            # Load first available topology file
            topology_file = available_files['topology'][0]
            data['topology'] = loader.load_timeseries(topology_file, 'topology')
            logger.info(f"Loaded topology data from {topology_file}")
        
        # If no real measurement data available, use synthetic data
        if synthetic_fallback and not any(available_files.get(mod) for mod in ['magnitude', 'phasor', 'waveform']):
            logger.info("Generating synthetic data for development")
            duration = 0.5 if small_data_mode else 2.0  # hours
            synthetic_data = loader.generate_synthetic_data(duration_hours=duration)
            data.update(synthetic_data)
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        if synthetic_fallback:
            logger.info("Falling back to synthetic data generation")
            duration = 0.5 if small_data_mode else 2.0
            return loader.generate_synthetic_data(duration_hours=duration)
        else:
            raise


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    dataset_path = "./digital-twin-dataset/digital-twin-dataset"
    data = load_sample_data(dataset_path, small_data_mode=True)
    
    print("Loaded data modalities:")
    for modality, df in data.items():
        print(f"  {modality}: {df.shape} - {df.index[0]} to {df.index[-1]}")
