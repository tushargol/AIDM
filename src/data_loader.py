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
                              noise_level: float = 0.01,
                              physics_based: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic PMU data with realistic power system characteristics.
        
        Args:
            duration_hours: Duration of synthetic data in hours
            sampling_rate: Sampling rate in Hz
            n_buses: Number of buses in the synthetic system
            noise_level: Noise level for synthetic measurements
            physics_based: Whether to use physics-based generation
            
        Returns:
            Dictionary containing synthetic timeseries data
        """
        n_samples = int(duration_hours * 3600 * sampling_rate)
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq=f'{1/sampling_rate}S')
        
        if physics_based:
            return self._generate_physics_based_data(timestamps, n_samples, n_buses, sampling_rate, noise_level)
        else:
            return self._generate_simple_synthetic_data(timestamps, n_samples, n_buses, noise_level)
    
    def _generate_physics_based_data(self, timestamps, n_samples, n_buses, sampling_rate, noise_level):
        """Generate physics-based synthetic PMU data."""
        
        # Generate realistic load profile
        load_profile = self._generate_load_profile(timestamps, n_buses)
        
        # Initialize power system model
        try:
            from pandapower_utils import PowerSystemModel
            power_model = PowerSystemModel({'power_system': {'n_buses': n_buses}})
            use_power_model = True
        except:
            logger.warning("PowerSystemModel not available, using simplified physics")
            use_power_model = False
        
        # Generate base measurements with power flow constraints
        voltages, angles = self._solve_power_flow_sequence(load_profile, n_buses, use_power_model)
        
        # Add spatial correlations
        voltages, angles = self._add_spatial_correlations(voltages, angles, n_buses)
        
        # Add PMU-specific characteristics
        frequency_data = self._generate_frequency_data(n_samples, sampling_rate)
        
        # Add realistic measurement noise
        voltage_noise = self._generate_correlated_noise(n_samples, n_buses, noise_level, 'voltage')
        angle_noise = self._generate_correlated_noise(n_samples, n_buses, noise_level * 0.1, 'angle')
        
        voltages += voltage_noise
        angles += angle_noise
        
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
        
        # Generate realistic topology data with temporal correlation
        topology_df = self._generate_realistic_topology(timestamps, n_buses)
        
        # Add frequency measurements
        frequency_df = pd.DataFrame(
            frequency_data, 
            index=timestamps, 
            columns=['system_frequency', 'rocof']
        )
        
        return {
            'magnitude': magnitude_df,
            'phasor': phasor_df,
            'topology': topology_df,
            'frequency': frequency_df
        }
    
    def _generate_load_profile(self, timestamps, n_buses):
        """Generate realistic daily load profile."""
        n_samples = len(timestamps)
        
        # Convert timestamps to hours of day
        hours_of_day = timestamps.hour + timestamps.minute/60.0 + timestamps.second/3600.0
        
        # Base daily load pattern (normalized 0.6 to 1.0)
        daily_pattern = 0.7 + 0.25 * (
            np.sin(2*np.pi * (hours_of_day - 6)/24) +  # Peak around 6 PM
            0.3 * np.sin(4*np.pi * hours_of_day/24 + np.pi/4) +  # Morning/evening peaks
            0.1 * np.sin(8*np.pi * hours_of_day/24)  # Higher frequency variations
        )
        
        # Add random load variations (±5%)
        load_variations = 1.0 + 0.05 * np.random.randn(n_samples)
        daily_pattern *= load_variations
        
        # Create load profile for each bus with diversity
        load_profile = np.zeros((n_samples, n_buses))
        for bus in range(n_buses):
            # Each bus has slightly different load pattern
            bus_factor = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2 scaling
            phase_shift = np.random.uniform(-2, 2)  # ±2 hour phase shift
            
            shifted_hours = (hours_of_day + phase_shift) % 24
            bus_pattern = 0.7 + 0.25 * (
                np.sin(2*np.pi * (shifted_hours - 6)/24) +
                0.3 * np.sin(4*np.pi * shifted_hours/24 + np.pi/4) +
                0.1 * np.sin(8*np.pi * shifted_hours/24)
            )
            
            load_profile[:, bus] = bus_pattern * bus_factor * load_variations
        
        return load_profile
    
    def _solve_power_flow_sequence(self, load_profile, n_buses, use_power_model):
        """Solve power flow for each time step."""
        n_samples = load_profile.shape[0]
        voltages = np.zeros((n_samples, n_buses))
        angles = np.zeros((n_samples, n_buses))
        
        # Base case voltages and angles
        base_voltages = np.ones(n_buses)  # 1.0 pu
        base_angles = np.linspace(0, -np.pi/6, n_buses)  # Realistic angle spread
        
        for t in range(n_samples):
            if use_power_model:
                # Use actual power flow (simplified for demo)
                v_mag, v_angle = self._simplified_power_flow(load_profile[t], base_voltages, base_angles)
            else:
                # Simplified physics-based relationships
                v_mag, v_angle = self._approximate_power_flow(load_profile[t], base_voltages, base_angles)
            
            voltages[t] = v_mag
            angles[t] = v_angle
        
        return voltages, angles
    
    def _simplified_power_flow(self, loads, base_v, base_angles):
        """Simplified power flow solution."""
        # Voltage drops proportional to load
        voltage_drops = 0.02 * (loads - 0.8)  # 2% drop per 0.1 pu load increase
        v_mag = base_v - voltage_drops
        v_mag = np.clip(v_mag, 0.95, 1.05)  # Voltage limits
        
        # Angle changes based on power flow
        angle_changes = 0.1 * (loads - 0.8)  # Angle increases with load
        v_angle = base_angles - np.cumsum(angle_changes) * 0.1
        
        return v_mag, v_angle
    
    def _approximate_power_flow(self, loads, base_v, base_angles):
        """Approximate power flow relationships."""
        # Simple DC power flow approximation
        # Voltage magnitude affected by reactive power (approximated)
        q_loads = loads * 0.3  # Assume 0.3 power factor
        v_mag = base_v - 0.01 * q_loads
        v_mag = np.clip(v_mag, 0.95, 1.05)
        
        # Angle differences from DC power flow
        # Simplified: angle difference proportional to power flow
        p_flow = np.diff(loads, prepend=loads[0])
        angle_deltas = p_flow * 0.05  # 0.05 rad per pu power
        v_angle = base_angles + np.cumsum(angle_deltas)
        
        return v_mag, v_angle
    
    def _add_spatial_correlations(self, voltages, angles, n_buses):
        """Add realistic spatial correlations."""
        # Create distance matrix (simplified ring topology)
        distance_matrix = np.zeros((n_buses, n_buses))
        for i in range(n_buses):
            for j in range(n_buses):
                distance_matrix[i, j] = min(abs(i-j), n_buses - abs(i-j))
        
        # Correlation decreases with distance
        correlation_length = 3.0  # buses
        correlation_matrix = np.exp(-distance_matrix / correlation_length)
        
        # Apply spatial smoothing
        n_samples = voltages.shape[0]
        for t in range(n_samples):
            # Smooth voltages
            v_smooth = correlation_matrix @ voltages[t] / np.sum(correlation_matrix, axis=1)
            voltages[t] = 0.7 * voltages[t] + 0.3 * v_smooth
            
            # Smooth angles
            a_smooth = correlation_matrix @ angles[t] / np.sum(correlation_matrix, axis=1)
            angles[t] = 0.7 * angles[t] + 0.3 * a_smooth
        
        return voltages, angles
    
    def _generate_frequency_data(self, n_samples, sampling_rate):
        """Generate realistic system frequency data."""
        # Base frequency (60 Hz in North America, 50 Hz in Europe)
        base_freq = 60.0
        
        # Frequency variations (±0.1 Hz typical)
        freq_variations = 0.02 * np.sin(2*np.pi * np.arange(n_samples) / (sampling_rate * 300))  # 5-minute cycle
        freq_variations += 0.01 * np.sin(2*np.pi * np.arange(n_samples) / (sampling_rate * 60))   # 1-minute cycle
        freq_variations += 0.005 * np.random.randn(n_samples)  # Random variations
        
        frequency = base_freq + freq_variations
        
        # Rate of Change of Frequency (ROCOF)
        rocof = np.gradient(frequency) * sampling_rate  # Hz/s
        
        return np.column_stack([frequency, rocof])
    
    def _generate_correlated_noise(self, n_samples, n_buses, noise_level, measurement_type):
        """Generate correlated measurement noise."""
        if measurement_type == 'voltage':
            # Voltage magnitude noise (PMU TVE < 1%)
            base_noise = noise_level * np.random.randn(n_samples, n_buses)
        else:  # angle
            # Angle noise (PMU angle accuracy)
            base_noise = noise_level * np.random.randn(n_samples, n_buses)
        
        # Add temporal correlation (measurement system effects)
        alpha = 0.1  # Correlation coefficient
        correlated_noise = np.zeros_like(base_noise)
        correlated_noise[0] = base_noise[0]
        
        for t in range(1, n_samples):
            correlated_noise[t] = alpha * correlated_noise[t-1] + np.sqrt(1-alpha**2) * base_noise[t]
        
        return correlated_noise
    
    def _generate_realistic_topology(self, timestamps, n_buses):
        """Generate realistic topology changes."""
        n_samples = len(timestamps)
        n_breakers = n_buses // 2
        
        # Initialize all breakers closed
        breaker_status = np.ones((n_samples, n_breakers))
        
        # Add occasional switching operations (maintenance, contingencies)
        for breaker in range(n_breakers):
            # Random switching events (very rare)
            if np.random.random() < 0.01:  # 1% chance of switching event
                switch_time = np.random.randint(n_samples // 4, 3 * n_samples // 4)
                switch_duration = np.random.randint(10, 100)  # 10-100 samples
                
                # Open breaker for maintenance
                end_time = min(switch_time + switch_duration, n_samples)
                breaker_status[switch_time:end_time, breaker] = 0
        
        topology_df = pd.DataFrame(
            breaker_status,
            index=timestamps,
            columns=[f'breaker_{i}_status' for i in range(n_breakers)]
        )
        
        return topology_df
    
    def _generate_simple_synthetic_data(self, timestamps, n_samples, n_buses, noise_level):
        """Original simple synthetic data generation (fallback)."""
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
    
    def get_example_element_names(self) -> Dict[str, List[str]]:
        """
        Get example element names that can be used with the API.
        Based on the digital-twin-dataset documentation.
        
        Returns:
            Dictionary with example element names for different data types
        """
        return {
            'magnitude': [
                'egauge_1-CT1', 'egauge_1-CT2', 'egauge_1-CT3',
                'egauge_2-CT1', 'egauge_2-CT2', 'egauge_2-CT3',
                'egauge_3-CT1', 'egauge_3-CT2', 'egauge_3-CT3'
            ],
            'phasor': [
                'egauge_1-L1', 'egauge_1-L2', 'egauge_1-L3',
                'egauge_2-L1', 'egauge_2-L2', 'egauge_2-L3',
                'egauge_3-L1', 'egauge_3-L2', 'egauge_3-L3'
            ],
            'waveform': [
                'egauge_1-CT1', 'egauge_1-CT2',
                'egauge_2-CT1', 'egauge_2-CT2'
            ]
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
            time_range: Tuple of (start_time, end_time) - can be datetime objects or ISO strings
            resolution: Time resolution (e.g., '1min', '10s') or timedelta object
            
        Returns:
            Dictionary containing loaded data
            
        Example:
            from datetime import datetime, timedelta
            
            # Load magnitude data for June 2024
            data = loader.load_api_data(
                magnitudes_for=["egauge_1-CT1"],
                time_range=(datetime(2024, 6, 1), datetime(2024, 7, 1)),
                resolution=timedelta(minutes=1)
            )
        """
        if not self.use_api or self.api_client is None:
            raise RuntimeError("API client not available or not enabled. "
                             "Set use_api=True and ensure DatasetApiClient is available.")
        
        # Use default example elements if none provided
        if not any([magnitudes_for, phasors_for, waveforms_for]):
            examples = self.get_example_element_names()
            magnitudes_for = examples['magnitude'][:2]  # Use first 2 elements
            logger.info(f"Using default magnitude elements: {magnitudes_for}")
        
        try:
            # Use API client to download data
            data = self.api_client.download_data(
                magnitudes_for=magnitudes_for or [],
                phasors_for=phasors_for or [],
                waveforms_for=waveforms_for or [],
                time_range=time_range,
                resolution=resolution
            )
            
            logger.info(f"Successfully loaded API data with {len(data)} modalities")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load API data: {e}")
            logger.info("Note: API access requires GitHub authentication. "
                       "Submit a ticket at https://forms.office.com/r/Ds6rKEtyTV to get access.")
            raise


def load_sample_data(dataset_path: str, 
                    small_data_mode: bool = True,
                    synthetic_fallback: bool = True,
                    use_api: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load sample data for development and testing.
    
    Args:
        dataset_path: Path to the dataset
        small_data_mode: Whether to use only a small subset of data
        synthetic_fallback: Whether to generate synthetic data if real data unavailable
        use_api: Whether to attempt API access for real data
        
    Returns:
        Dictionary containing loaded timeseries data
    """
    loader = DigitalTwinDataLoader(dataset_path, use_api=use_api)
    
    try:
        # Try to load real topology data
        available_files = loader.list_available_files()
        data = {}
        
        if available_files.get('topology'):
            # Load first available topology file
            topology_file = available_files['topology'][0]
            data['topology'] = loader.load_timeseries(topology_file, 'topology')
            logger.info(f"Loaded topology data from {topology_file}")
        
        # Try to load API data if enabled
        if use_api and loader.api_client is not None:
            try:
                from datetime import datetime, timedelta
                
                # Load a small sample of real data
                if small_data_mode:
                    # Load 1 hour of data at 1-minute resolution
                    api_data = loader.load_api_data(
                        magnitudes_for=["egauge_1-CT1"],
                        time_range=(datetime(2024, 6, 1), datetime(2024, 6, 1, 1)),
                        resolution=timedelta(minutes=1)
                    )
                else:
                    # Load 1 day of data at 10-second resolution
                    api_data = loader.load_api_data(
                        magnitudes_for=["egauge_1-CT1", "egauge_1-CT2"],
                        phasors_for=["egauge_1-L1", "egauge_1-L2"],
                        time_range=(datetime(2024, 6, 1), datetime(2024, 6, 2)),
                        resolution=timedelta(seconds=10)
                    )
                
                if api_data:
                    data.update(api_data)
                    logger.info("Successfully loaded real data via API")
                    return data
                    
            except Exception as e:
                logger.warning(f"API data loading failed: {e}")
                logger.info("Falling back to local/synthetic data")
        
        # If no real measurement data available, use synthetic data
        if synthetic_fallback and not any(available_files.get(mod) for mod in ['magnitude', 'phasor', 'waveform']):
            logger.info("Generating physics-based synthetic PMU data for development")
            duration = 0.5 if small_data_mode else 2.0  # hours
            synthetic_data = loader.generate_synthetic_data(
                duration_hours=duration,
                physics_based=True  # Use improved physics-based generation
            )
            data.update(synthetic_data)
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        if synthetic_fallback:
            logger.info("Falling back to physics-based synthetic data generation")
            duration = 0.5 if small_data_mode else 2.0
            return loader.generate_synthetic_data(
                duration_hours=duration,
                physics_based=True
            )
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
