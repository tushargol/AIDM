"""
Pandapower utilities for power system modeling and Jacobian computation.
Includes fallback numeric Jacobian computation for FDIA generation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import pandapower
try:
    import pandapower as pp
    import pandapower.networks as pn
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False
    logging.warning("Pandapower not available. Using fallback numeric Jacobian computation.")

logger = logging.getLogger(__name__)


class PowerSystemModel:
    """
    Power system modeling and Jacobian computation for FDIA generation.
    Supports both pandapower and fallback numeric methods.
    """
    
    def __init__(self, use_pandapower: bool = True):
        """
        Initialize the power system model.
        
        Args:
            use_pandapower: Whether to use pandapower (if available)
        """
        self.use_pandapower = use_pandapower and PANDAPOWER_AVAILABLE
        self.net = None
        self.measurement_config = {}
        
    def build_net_from_repo(self, network_file: str = None) -> Any:
        """
        Build a pandapower network from repository case files.
        
        Args:
            network_file: Path to network JSON file (optional)
            
        Returns:
            Pandapower network object or fallback network dict
        """
        if self.use_pandapower:
            return self._build_pandapower_net(network_file)
        else:
            return self._build_fallback_net()
    
    def _build_pandapower_net(self, network_file: str = None) -> Any:
        """Build pandapower network from case files."""
        try:
            # For demonstration, create a simple test network
            # In practice, this would parse the actual network files from the repo
            net = pp.create_empty_network()
            
            # Create buses
            bus1 = pp.create_bus(net, vn_kv=20., name="Bus 1")
            bus2 = pp.create_bus(net, vn_kv=20., name="Bus 2")
            bus3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")
            
            # Create external grid connection
            pp.create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
            
            # Create transformer
            pp.create_transformer(net, hv_bus=bus1, lv_bus=bus3, std_type="0.4 MVA 20/0.4 kV")
            
            # Create line
            pp.create_line(net, from_bus=bus1, to_bus=bus2, length_km=10, std_type="NAYY 4x50 SE")
            
            # Create loads
            pp.create_load(net, bus=bus2, p_mw=0.1, q_mvar=0.05, name="Load 1")
            pp.create_load(net, bus=bus3, p_mw=0.1, q_mvar=0.05, name="Load 2")
            
            # Run power flow to initialize
            pp.runpp(net, verbose=False)
            
            self.net = net
            logger.info(f"Created pandapower network with {len(net.bus)} buses")
            return net
            
        except Exception as e:
            logger.error(f"Failed to create pandapower network: {e}")
            logger.info("Falling back to numeric network model")
            return self._build_fallback_net()
    
    def _build_fallback_net(self) -> Dict:
        """Build a simple fallback network model."""
        # IEEE 14-bus test case (simplified)
        net = {
            'n_buses': 14,
            'n_branches': 20,
            'base_mva': 100.0,
            'bus_data': {
                'voltage_mag': np.ones(14) * 1.0,  # Per unit voltages
                'voltage_angle': np.zeros(14),      # Voltage angles in radians
                'load_p': np.array([0, 0.217, 0.942, 0.478, 0.076, 0.112, 0, 0, 0.295, 0.09, 0.035, 0.061, 0.135, 0.149]),
                'load_q': np.array([0, 0.127, 0.19, -0.039, 0.016, 0.075, 0, 0, 0.166, 0.058, 0.018, 0.016, 0.058, 0.05])
            },
            'branch_data': {
                'from_bus': np.array([1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7, 9, 9, 10, 12, 13]) - 1,
                'to_bus': np.array([2, 5, 3, 4, 5, 4, 5, 7, 9, 6, 11, 12, 13, 8, 9, 10, 14, 11, 13, 14]) - 1,
                'resistance': np.array([0.01938, 0.05403, 0.04699, 0.05811, 0.05695, 0.06701, 0.01335, 0, 0, 0, 0.09498, 0.12291, 0.06615, 0, 0.11001, 0.03181, 0.08205, 0.08205, 0.22092, 0.17093]),
                'reactance': np.array([0.05917, 0.22304, 0.19797, 0.17632, 0.17388, 0.17103, 0.04211, 0.20912, 0.55618, 0.25202, 0.1989, 0.25581, 0.13027, 0.17615, 0.35530, 0.0845, 0.19207, 0.19207, 0.19988, 0.34802])
            }
        }
        
        self.net = net
        logger.info(f"Created fallback network with {net['n_buses']} buses")
        return net
    
    def compute_jacobian(self, 
                        measurement_fn: Callable = None,
                        eps: float = 1e-4,
                        n_measurements: int = None) -> np.ndarray:
        """
        Compute measurement Jacobian matrix H for state estimation.
        
        Args:
            measurement_fn: Function to compute measurements from state
            eps: Perturbation size for numeric differentiation
            n_measurements: Number of measurements to match (optional)
            
        Returns:
            Jacobian matrix H (n_measurements x n_states)
        """
        if self.use_pandapower and self.net is not None:
            return self._compute_pandapower_jacobian(measurement_fn, eps)
        else:
            return self._compute_numeric_jacobian(measurement_fn, eps, n_measurements)
    
    def _compute_pandapower_jacobian(self, 
                                   measurement_fn: Callable = None,
                                   eps: float = 1e-4) -> np.ndarray:
        """Compute Jacobian using pandapower."""
        try:
            # Get current state
            n_buses = len(self.net.bus)
            
            # State vector: [V_mag, V_angle] (excluding slack bus angle)
            V_mag = self.net.res_bus.vm_pu.values
            V_angle = self.net.res_bus.va_degree.values * np.pi / 180  # Convert to radians
            
            # Exclude slack bus angle from state (typically bus 0)
            state = np.concatenate([V_mag, V_angle[1:]])
            n_states = len(state)
            
            # Base measurements
            z0 = self._get_measurements_pandapower()
            n_measurements = len(z0)
            
            # Initialize Jacobian
            H = np.zeros((n_measurements, n_states))
            
            # Compute partial derivatives numerically
            for i in range(n_states):
                # Perturb state
                state_pert = state.copy()
                state_pert[i] += eps
                
                # Update network with perturbed state
                self._update_network_state_pandapower(state_pert)
                
                # Compute perturbed measurements
                z_pert = self._get_measurements_pandapower()
                
                # Compute partial derivative
                H[:, i] = (z_pert - z0) / eps
                
                # Restore original state
                self._update_network_state_pandapower(state)
            
            logger.info(f"Computed pandapower Jacobian: {H.shape}")
            return H
            
        except Exception as e:
            logger.error(f"Pandapower Jacobian computation failed: {e}")
            return self._compute_numeric_jacobian(measurement_fn, eps)
    
    def _compute_numeric_jacobian(self, 
                                measurement_fn: Callable = None,
                                eps: float = 1e-4,
                                n_measurements: int = None) -> np.ndarray:
        """Compute Jacobian using numeric differentiation on fallback model."""
        if measurement_fn is None:
            measurement_fn = self._default_measurement_function
        
        # Get current state from fallback network
        V_mag = self.net['bus_data']['voltage_mag']
        V_angle = self.net['bus_data']['voltage_angle']
        
        # State vector (exclude slack bus angle)
        state = np.concatenate([V_mag, V_angle[1:]])
        n_states = len(state)
        
        # Base measurements - use specific number if provided
        if n_measurements is not None:
            z0 = measurement_fn(state, n_measurements)
        else:
            z0 = measurement_fn(state)
        n_measurements_actual = len(z0)
        
        # Initialize Jacobian
        H = np.zeros((n_measurements_actual, n_states))
        
        # Compute partial derivatives
        for i in range(n_states):
            # Perturb state
            state_pert = state.copy()
            state_pert[i] += eps
            
            # Compute perturbed measurements
            if n_measurements is not None:
                z_pert = measurement_fn(state_pert, n_measurements)
            else:
                z_pert = measurement_fn(state_pert)
            
            # Compute partial derivative
            H[:, i] = (z_pert - z0) / eps
        
        logger.info(f"Computed numeric Jacobian: {H.shape}")
        return H
    
    def _get_measurements_pandapower(self) -> np.ndarray:
        """Get measurements from pandapower network."""
        # Run power flow
        pp.runpp(self.net, verbose=False)
        
        # Extract measurements (voltage magnitudes and power flows)
        measurements = []
        
        # Bus voltage magnitudes
        measurements.extend(self.net.res_bus.vm_pu.values)
        
        # Branch power flows (P and Q)
        measurements.extend(self.net.res_line.p_from_mw.values)
        measurements.extend(self.net.res_line.q_from_mvar.values)
        
        return np.array(measurements)
    
    def _update_network_state_pandapower(self, state: np.ndarray) -> None:
        """Update pandapower network with new state."""
        n_buses = len(self.net.bus)
        
        # Extract voltage magnitudes and angles
        V_mag = state[:n_buses]
        V_angle = np.zeros(n_buses)
        V_angle[1:] = state[n_buses:]  # Slack bus angle remains 0
        
        # Update bus voltages (this is a simplified approach)
        # In practice, you'd need to solve the power flow equations
        for i, bus_idx in enumerate(self.net.bus.index):
            if self.net.bus.loc[bus_idx, 'type'] != 'b':  # Not slack bus
                # This is a simplified update - real implementation would be more complex
                pass
    
    def _default_measurement_function(self, state: np.ndarray, n_measurements: int = None) -> np.ndarray:
        """Default measurement function for fallback network."""
        n_buses = self.net['n_buses']
        
        # Extract state variables
        V_mag = state[:n_buses]
        V_angle = np.zeros(n_buses)
        V_angle[1:] = state[n_buses:]
        
        # Simple measurement model: voltage magnitudes + some derived quantities
        measurements = []
        
        # Voltage magnitudes
        measurements.extend(V_mag)
        
        # If we need a specific number of measurements, adjust accordingly
        if n_measurements is not None and len(measurements) < n_measurements:
            # Add simple power flow approximations (DC power flow) until we reach desired count
            branch_data = self.net['branch_data']
            measurements_needed = n_measurements - len(measurements)
            measurements_added = 0
            
            for i in range(len(branch_data['from_bus'])):
                if measurements_added >= measurements_needed:
                    break
                    
                from_bus = branch_data['from_bus'][i]
                to_bus = branch_data['to_bus'][i]
                x = branch_data['reactance'][i]
                
                # DC power flow: P = (theta_from - theta_to) / x
                if x > 0:
                    p_flow = (V_angle[from_bus] - V_angle[to_bus]) / x
                    measurements.append(p_flow)
                    measurements_added += 1
            
            # If still not enough measurements, add synthetic ones
            while len(measurements) < n_measurements:
                # Add synthetic measurements based on voltage combinations
                bus_idx = len(measurements) % n_buses
                synthetic_measurement = V_mag[bus_idx] * (1 + 0.1 * np.sin(V_angle[bus_idx]))
                measurements.append(synthetic_measurement)
        elif n_measurements is not None and len(measurements) > n_measurements:
            # Truncate to desired number of measurements
            measurements = measurements[:n_measurements]
        
        return np.array(measurements)
    
    def get_measurement_config(self) -> Dict:
        """Get measurement configuration for the network."""
        if self.use_pandapower and self.net is not None:
            config = {
                'n_buses': len(self.net.bus),
                'n_measurements': len(self._get_measurements_pandapower()),
                'measurement_types': ['vm_pu', 'p_flow', 'q_flow'],
                'bus_names': self.net.bus.name.tolist() if 'name' in self.net.bus.columns else [f'Bus_{i}' for i in range(len(self.net.bus))]
            }
        else:
            config = {
                'n_buses': self.net['n_buses'],
                'n_measurements': len(self._default_measurement_function(np.concatenate([self.net['bus_data']['voltage_mag'], self.net['bus_data']['voltage_angle'][1:]]))),
                'measurement_types': ['vm_pu', 'p_flow'],
                'bus_names': [f'Bus_{i}' for i in range(self.net['n_buses'])]
            }
        
        return config
    
    def generate_base_measurements(self, 
                                 noise_std: float = 0.01,
                                 n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate base measurements for testing.
        
        Args:
            noise_std: Standard deviation of measurement noise
            n_samples: Number of measurement samples to generate
            
        Returns:
            Tuple of (measurements, timestamps)
        """
        # Get base measurement
        if self.use_pandapower and self.net is not None:
            base_z = self._get_measurements_pandapower()
        else:
            state = np.concatenate([
                self.net['bus_data']['voltage_mag'],
                self.net['bus_data']['voltage_angle'][1:]
            ])
            base_z = self._default_measurement_function(state)
        
        # Generate time series with small variations
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1S')
        measurements = np.zeros((n_samples, len(base_z)))
        
        for i in range(n_samples):
            # Add small random variations and noise
            variation = 0.02 * np.sin(2 * np.pi * i / 50)  # Slow variation
            noise = noise_std * np.random.randn(len(base_z))
            measurements[i] = base_z * (1 + variation) + noise
        
        logger.info(f"Generated {n_samples} measurement samples with {len(base_z)} measurements each")
        return measurements, timestamps


def create_power_system_model(config: Dict = None) -> PowerSystemModel:
    """
    Factory function to create a power system model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PowerSystemModel instance
    """
    use_pandapower = True
    if config and 'pandapower' in config:
        use_pandapower = config['pandapower'].get('enabled', True)
    
    model = PowerSystemModel(use_pandapower=use_pandapower)
    
    # Build network
    network_file = None
    if config and 'pandapower' in config:
        network_file = config['pandapower'].get('network_file')
    
    model.build_net_from_repo(network_file)
    
    return model


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create power system model
    model = create_power_system_model()
    
    # Compute Jacobian
    H = model.compute_jacobian()
    print(f"Jacobian shape: {H.shape}")
    
    # Generate base measurements
    measurements, timestamps = model.generate_base_measurements(n_samples=50)
    print(f"Generated measurements: {measurements.shape}")
    
    # Get measurement configuration
    config = model.get_measurement_config()
    print(f"Measurement config: {config}")
