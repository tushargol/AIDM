"""
Attack generation module for AIDM evaluation.
Implements FDIA, temporal stealth, replay attacks, and ART-based adversarial examples.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import yaml
import click
import warnings
warnings.filterwarnings('ignore')

from pandapower_utils import create_power_system_model

# Try to import ART (Adversarial Robustness Toolbox)
try:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification import TensorFlowV2Classifier, SklearnClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    logging.warning("Adversarial Robustness Toolbox (ART) not available. ML adversarial attacks will be disabled.")

logger = logging.getLogger(__name__)


class AttackGenerator:
    """
    Comprehensive attack generator for power system security evaluation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the attack generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.power_model = None
        self.jacobian = None
        
    def initialize_power_model(self) -> None:
        """Initialize the power system model for FDIA generation."""
        try:
            self.power_model = create_power_system_model(self.config)
            self.jacobian = self.power_model.compute_jacobian()
            logger.info(f"Initialized power model with Jacobian shape: {self.jacobian.shape}")
        except Exception as e:
            logger.error(f"Failed to initialize power model: {e}")
            self.power_model = None
            self.jacobian = None
    
    def make_fdia(self, 
                  z: np.ndarray, 
                  H: Optional[np.ndarray] = None,
                  c: Optional[np.ndarray] = None,
                  attack_magnitude: float = None) -> np.ndarray:
        """
        Generate False Data Injection Attack (FDIA).
        
        Args:
            z: Original measurements (n_measurements,)
            H: Measurement Jacobian matrix (n_measurements x n_states)
            c: Attack vector on state variables (n_states,)
            attack_magnitude: Maximum attack magnitude relative to measurement std
            
        Returns:
            Attacked measurements z_adv
        """
        if H is None:
            if self.jacobian is not None:
                H = self.jacobian
            else:
                logger.warning("No Jacobian available, using identity matrix")
                H = np.eye(len(z))
        
        if c is None:
            # Generate random attack vector
            n_states = H.shape[1]
            c = np.random.randn(n_states)
            c = c / np.linalg.norm(c)  # Normalize
        
        # Set attack magnitude
        if attack_magnitude is None:
            attack_magnitude = self.config['attacks']['fdia']['max_attack_magnitude']
        
        # Scale attack vector
        z_std = np.std(z)
        c_scaled = c * attack_magnitude * z_std
        
        # Compute attack on measurements: a = H * c
        a = H @ c_scaled
        
        # Apply rate limiting
        rate_limit = self.config['attacks']['fdia']['rate_limit']
        a = np.clip(a, -rate_limit * z_std, rate_limit * z_std)
        
        # Generate attacked measurements
        z_adv = z + a
        
        # Ensure physical constraints (e.g., positive voltage magnitudes)
        z_adv = self._apply_physical_constraints(z_adv, z)
        
        logger.info(f"Generated FDIA with attack magnitude: {np.linalg.norm(a):.4f}")
        return z_adv
    
    def make_temporal_stealth(self, 
                            Z: np.ndarray,
                            indices: List[int],
                            duration: int = None,
                            max_step: float = None) -> np.ndarray:
        """
        Generate temporal stealth attack with gradual ramp.
        
        Args:
            Z: Time series measurements (n_timesteps x n_measurements)
            indices: Measurement indices to attack
            duration: Attack duration in timesteps
            max_step: Maximum change per timestep
            
        Returns:
            Attacked time series Z_adv
        """
        if duration is None:
            duration = self.config['attacks']['temporal_stealth']['max_duration']
        if max_step is None:
            max_step = self.config['attacks']['temporal_stealth']['max_step_size']
        
        Z_adv = Z.copy()
        n_timesteps, n_measurements = Z.shape
        
        # Select random start time
        start_time = np.random.randint(0, max(1, n_timesteps - duration))
        end_time = min(start_time + duration, n_timesteps)
        
        # Generate attack trajectory for each selected measurement
        for idx in indices:
            if idx >= n_measurements:
                continue
                
            # Target attack magnitude
            z_std = np.std(Z[:, idx])
            target_magnitude = np.random.uniform(0.1, 0.3) * z_std
            target_sign = np.random.choice([-1, 1])
            target_attack = target_sign * target_magnitude
            
            # Generate gradual ramp
            attack_trajectory = np.zeros(end_time - start_time)
            current_attack = 0.0
            
            for t in range(len(attack_trajectory)):
                # Gradual increase/decrease
                step_size = np.random.uniform(-max_step, max_step) * z_std
                if abs(current_attack) < abs(target_attack):
                    step_size += np.sign(target_attack) * max_step * z_std * 0.5
                
                current_attack += step_size
                current_attack = np.clip(current_attack, -abs(target_attack), abs(target_attack))
                attack_trajectory[t] = current_attack
            
            # Apply attack
            Z_adv[start_time:end_time, idx] += attack_trajectory
        
        logger.info(f"Generated temporal stealth attack on {len(indices)} measurements for {end_time - start_time} timesteps")
        return Z_adv
    
    def make_replay(self, 
                   Z: np.ndarray,
                   source_window: Tuple[int, int] = None,
                   target_time: int = None) -> np.ndarray:
        """
        Generate replay attack by copying previous window.
        
        Args:
            Z: Time series measurements (n_timesteps x n_measurements)
            source_window: Tuple of (start, end) for source window
            target_time: Target time to start replay
            
        Returns:
            Attacked time series Z_adv
        """
        Z_adv = Z.copy()
        n_timesteps, n_measurements = Z.shape
        
        # Select source window
        if source_window is None:
            min_window = self.config['attacks']['replay']['min_window_size']
            max_window = self.config['attacks']['replay']['max_window_size']
            window_size = np.random.randint(min_window, max_window + 1)
            
            source_start = np.random.randint(0, n_timesteps - window_size - max_window)
            source_end = source_start + window_size
            source_window = (source_start, source_end)
        
        # Select target time
        if target_time is None:
            window_size = source_window[1] - source_window[0]
            target_time = np.random.randint(source_window[1] + 10, n_timesteps - window_size)
        
        # Copy source window to target location
        source_data = Z[source_window[0]:source_window[1]]
        window_size = source_data.shape[0]
        target_end = min(target_time + window_size, n_timesteps)
        actual_window_size = target_end - target_time
        
        Z_adv[target_time:target_end] = source_data[:actual_window_size]
        
        logger.info(f"Generated replay attack: copied window {source_window} to timestep {target_time}")
        return Z_adv
    
    def make_art_attacks(self, 
                        model: Any,
                        X: np.ndarray,
                        attack_params: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """
        Generate adversarial examples using ART library.
        
        Args:
            model: Trained classifier model
            X: Input data (n_samples x n_features)
            attack_params: Attack parameters
            
        Returns:
            Dictionary of attacked data for different attack types
        """
        if not ART_AVAILABLE:
            logger.error("ART library not available for adversarial attacks")
            return {}
        
        if attack_params is None:
            attack_params = self.config['attacks']['art_attacks']
        
        # Wrap model for ART
        try:
            if hasattr(model, 'predict_proba'):
                # Sklearn-style model
                art_classifier = SklearnClassifier(model=model)
            else:
                # Assume TensorFlow/Keras model
                art_classifier = TensorFlowV2Classifier(
                    model=model,
                    nb_classes=2,  # Assuming binary classification
                    input_shape=X.shape[1:],
                    loss_object=None  # Will be inferred
                )
        except Exception as e:
            logger.error(f"Failed to wrap model for ART: {e}")
            return {}
        
        attacked_data = {}
        
        # FGSM attacks
        try:
            for eps in attack_params['fgsm_eps']:
                fgsm = FastGradientMethod(estimator=art_classifier, eps=eps)
                X_adv_fgsm = fgsm.generate(x=X)
                attacked_data[f'fgsm_eps_{eps}'] = X_adv_fgsm
                logger.info(f"Generated FGSM attack with eps={eps}")
        except Exception as e:
            logger.error(f"FGSM attack generation failed: {e}")
        
        # PGD attacks
        try:
            for eps in attack_params['pgd_eps']:
                pgd = ProjectedGradientDescent(
                    estimator=art_classifier,
                    eps=eps,
                    eps_step=attack_params['pgd_step_size'],
                    max_iter=attack_params['pgd_steps']
                )
                X_adv_pgd = pgd.generate(x=X)
                attacked_data[f'pgd_eps_{eps}'] = X_adv_pgd
                logger.info(f"Generated PGD attack with eps={eps}")
        except Exception as e:
            logger.error(f"PGD attack generation failed: {e}")
        
        return attacked_data
    
    def _apply_physical_constraints(self, z_adv: np.ndarray, z_orig: np.ndarray) -> np.ndarray:
        """Apply physical constraints to attacked measurements."""
        # Ensure voltage magnitudes remain positive and within reasonable bounds
        # This is a simplified constraint - real implementation would be more sophisticated
        
        # Assume first n measurements are voltage magnitudes
        n_voltage_measurements = min(len(z_adv) // 2, len(z_adv))
        
        for i in range(n_voltage_measurements):
            # Keep voltage magnitudes between 0.8 and 1.2 pu
            if z_adv[i] < 0.8:
                z_adv[i] = max(0.8, z_orig[i] * 0.9)
            elif z_adv[i] > 1.2:
                z_adv[i] = min(1.2, z_orig[i] * 1.1)
        
        return z_adv
    
    def generate_attack_dataset(self, 
                              measurements: np.ndarray,
                              timestamps: pd.DatetimeIndex,
                              attack_types: List[str] = None,
                              attack_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Generate a comprehensive attack dataset.
        
        Args:
            measurements: Clean measurements (n_timesteps x n_measurements)
            timestamps: Timestamp index
            attack_types: List of attack types to generate
            attack_ratio: Fraction of data to attack
            
        Returns:
            Dictionary containing attack dataset and metadata
        """
        if attack_types is None:
            attack_types = ['fdia', 'temporal_stealth', 'replay']
        
        n_timesteps, n_measurements = measurements.shape
        n_attack_samples = int(n_timesteps * attack_ratio)
        
        # Initialize attack dataset
        attack_data = {
            'clean': measurements.copy(),
            'labels': np.zeros(n_timesteps),  # 0 = clean, 1 = attack
            'attack_types': np.full(n_timesteps, 'clean', dtype=object),
            'timestamps': timestamps
        }
        
        # Generate attacks
        attack_indices = np.random.choice(n_timesteps, n_attack_samples, replace=False)
        
        for i, attack_idx in enumerate(attack_indices):
            attack_type = np.random.choice(attack_types)
            
            if attack_type == 'fdia':
                # Single-timestep FDIA
                z_clean = measurements[attack_idx]
                z_attacked = self.make_fdia(z_clean)
                attack_data['clean'][attack_idx] = z_attacked
                
            elif attack_type == 'temporal_stealth':
                # Multi-timestep stealth attack
                duration = np.random.randint(5, 20)
                end_idx = min(attack_idx + duration, n_timesteps)
                
                # Select random measurements to attack
                n_measurements_to_attack = np.random.randint(1, min(5, n_measurements))
                measurement_indices = np.random.choice(n_measurements, n_measurements_to_attack, replace=False)
                
                # Apply stealth attack to window
                window_data = measurements[attack_idx:end_idx].copy()
                attacked_window = self.make_temporal_stealth(
                    window_data, 
                    measurement_indices.tolist(),
                    duration=end_idx - attack_idx
                )
                attack_data['clean'][attack_idx:end_idx] = attacked_window
                
                # Mark all timesteps in window as attacked
                for j in range(attack_idx, end_idx):
                    if j < n_timesteps:
                        attack_data['labels'][j] = 1
                        attack_data['attack_types'][j] = attack_type
                continue  # Skip the single-timestep labeling below
                
            elif attack_type == 'replay':
                # Replay attack
                window_size = np.random.randint(5, 15)
                if attack_idx + window_size < n_timesteps:
                    # Find source window
                    source_start = max(0, attack_idx - 50)
                    source_end = max(source_start + window_size, attack_idx - 10)
                    
                    if source_end <= attack_idx:
                        source_window = (source_start, source_end)
                        window_data = measurements[attack_idx:attack_idx + window_size].copy()
                        attacked_window = self.make_replay(
                            np.vstack([measurements[source_start:source_end], window_data]),
                            source_window=(0, source_end - source_start),
                            target_time=source_end - source_start
                        )
                        attack_data['clean'][attack_idx:attack_idx + window_size] = attacked_window[source_end - source_start:]
                        
                        # Mark all timesteps in window as attacked
                        for j in range(attack_idx, attack_idx + window_size):
                            if j < n_timesteps:
                                attack_data['labels'][j] = 1
                                attack_data['attack_types'][j] = attack_type
                        continue
            
            # Single-timestep labeling (for FDIA and failed multi-timestep attacks)
            attack_data['labels'][attack_idx] = 1
            attack_data['attack_types'][attack_idx] = attack_type
        
        # Add metadata
        attack_data['metadata'] = {
            'total_samples': n_timesteps,
            'attack_samples': np.sum(attack_data['labels']),
            'attack_ratio': np.sum(attack_data['labels']) / n_timesteps,
            'attack_types_used': attack_types,
            'generation_config': self.config['attacks']
        }
        
        logger.info(f"Generated attack dataset: {attack_data['metadata']['attack_samples']}/{n_timesteps} samples attacked")
        return attack_data
    
    def save_attack_dataset(self, attack_data: Dict[str, Any], output_path: str, experiment_name: str) -> None:
        """Save comprehensive attack dataset to disk for IDS training."""
        output_dir = Path(output_path) / "experiments"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete dataset with both clean and attacked measurements
        np.savez_compressed(
            output_dir / f"{experiment_name}_attacks.npz",
            measurements=attack_data['clean'],  # Contains both clean and attacked data
            labels=attack_data['labels'],       # 0 = clean, 1 = attack
            attack_types=attack_data['attack_types'],  # Type of attack or 'clean'
            timestamps=attack_data['timestamps'].values
        )
        
        # Also save clean and attack data separately for analysis
        clean_mask = attack_data['labels'] == 0
        attack_mask = attack_data['labels'] == 1
        
        np.savez_compressed(
            output_dir / f"{experiment_name}_clean_data.npz",
            measurements=attack_data['clean'][clean_mask],
            timestamps=attack_data['timestamps'].values[clean_mask]
        )
        
        if np.any(attack_mask):
            np.savez_compressed(
                output_dir / f"{experiment_name}_attack_data.npz",
                measurements=attack_data['clean'][attack_mask],
                attack_types=attack_data['attack_types'][attack_mask],
                timestamps=attack_data['timestamps'].values[attack_mask]
            )
        
        # Save metadata
        with open(output_dir / f"{experiment_name}_metadata.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            metadata = attack_data['metadata'].copy()
            for key, value in metadata.items():
                if isinstance(value, np.integer):
                    metadata[key] = int(value)
                elif isinstance(value, np.floating):
                    metadata[key] = float(value)
            
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved comprehensive attack dataset to {output_dir / experiment_name}")
        logger.info(f"  - Combined dataset: {experiment_name}_attacks.npz")
        logger.info(f"  - Clean data only: {experiment_name}_clean_data.npz")
        logger.info(f"  - Attack data only: {experiment_name}_attack_data.npz")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@click.command()
@click.option('--config', default='config.yaml', help='Path to configuration file')
@click.option('--type', 'attack_type', default='fdia', 
              type=click.Choice(['fdia', 'temporal_stealth', 'replay', 'all']),
              help='Type of attack to generate')
@click.option('--output', default=None, help='Output directory (overrides config)')
@click.option('--experiment', default='default_attacks', help='Experiment name')
@click.option('--samples', default=1000, help='Number of samples to generate')
def main(config: str, attack_type: str, output: str, experiment: str, samples: int):
    """
    Generate attack datasets for AIDM evaluation.
    
    Example usage:
        python src/attacks.py --type fdia --samples 500
        python src/attacks.py --type all --experiment comprehensive_attacks
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
    
    try:
        # Initialize attack generator
        generator = AttackGenerator(config_dict)
        generator.initialize_power_model()
        
        # Load sample data from digital-twin-dataset as foundation
        try:
            from data_loader import load_sample_data
            
            dataset_path = config_dict.get('data', {}).get('dataset_path', './digital-twin-dataset/digital-twin-dataset')
            
            logger.info(f"Loading sample data from: {dataset_path}")
            data = load_sample_data(
                dataset_path=dataset_path,
                small_data_mode=config_dict['compute']['small_data_mode'],
                synthetic_fallback=True,
                use_api=False  # Use sample data only
            )
            
            # Extract measurements from loaded data
            if 'magnitude' in data and not data['magnitude'].empty:
                base_measurements = data['magnitude'].values
                base_timestamps = data['magnitude'].index
                logger.info(f"Using magnitude data: {base_measurements.shape}")
            elif 'phasor' in data and not data['phasor'].empty:
                base_measurements = data['phasor'].values
                base_timestamps = data['phasor'].index
                logger.info(f"Using phasor data: {base_measurements.shape}")
            else:
                # Use any available data
                available_keys = [k for k, v in data.items() if not v.empty]
                if available_keys:
                    key = available_keys[0]
                    base_measurements = data[key].values
                    base_timestamps = data[key].index
                    logger.info(f"Using {key} data: {base_measurements.shape}")
                else:
                    raise ValueError("No valid data found in sample dataset")
            
            # Expand dataset if we need more samples for training
            if len(base_measurements) < samples:
                logger.info(f"Expanding dataset from {len(base_measurements)} to {samples} samples")
                repeat_factor = (samples // len(base_measurements)) + 1
                expanded_measurements = []
                expanded_timestamps = []
                
                for i in range(repeat_factor):
                    # Add small noise variations for each repetition
                    noise_factor = 0.005 * (i + 1)  # Small noise to maintain realism
                    noise = np.random.normal(0, noise_factor, base_measurements.shape)
                    varied_measurements = base_measurements + noise
                    
                    # Create new timestamps
                    time_offset = pd.Timedelta(hours=i * len(base_measurements) / 3600)
                    varied_timestamps = base_timestamps + time_offset
                    
                    expanded_measurements.append(varied_measurements)
                    expanded_timestamps.append(varied_timestamps)
                
                measurements = np.vstack(expanded_measurements)[:samples]
                timestamps = pd.concat([pd.Series(ts) for ts in expanded_timestamps])[:samples]
                timestamps = pd.DatetimeIndex(timestamps)
                
                logger.info(f"Expanded dataset to {measurements.shape} samples")
            else:
                measurements = base_measurements[:samples]
                timestamps = base_timestamps[:samples]
            
        except Exception as e:
            logger.warning(f"Failed to load real dataset: {e}")
            logger.info("Falling back to synthetic measurements")
            
            # Fallback: generate synthetic measurements based on power system model
            if generator.power_model:
                try:
                    measurements, timestamps = generator.power_model.generate_base_measurements(
                        n_samples=samples,
                        noise_std=0.01
                    )
                    logger.info("Using power system model synthetic data")
                except Exception as model_e:
                    logger.warning(f"Power model failed: {model_e}")
                    timestamps = pd.date_range(start='2024-01-01', periods=samples, freq='1S')
                    measurements = np.random.randn(samples, 20) * 0.1 + 1.0  # Voltage-like measurements
                    logger.info("Using basic synthetic measurements")
            else:
                timestamps = pd.date_range(start='2024-01-01', periods=samples, freq='1S')
                measurements = np.random.randn(samples, 20) * 0.1 + 1.0  # Voltage-like measurements
                logger.info("Using basic synthetic measurements")
        
        # Determine attack types to generate
        if attack_type == 'all':
            attack_types = ['fdia', 'temporal_stealth', 'replay']
        else:
            attack_types = [attack_type]
        
        # Generate comprehensive attack dataset for IDS training
        # Use a balanced approach: 70% clean, 30% attacks for proper training
        attack_data = generator.generate_attack_dataset(
            measurements=measurements,
            timestamps=timestamps,
            attack_types=attack_types,
            attack_ratio=0.3  # 30% attacks, 70% clean data
        )
        
        # Ensure we have enough clean data for training
        clean_samples = np.sum(attack_data['labels'] == 0)
        attack_samples = np.sum(attack_data['labels'] == 1)
        
        logger.info(f"Dataset composition:")
        logger.info(f"  Clean samples: {clean_samples} ({clean_samples/len(attack_data['labels'])*100:.1f}%)")
        logger.info(f"  Attack samples: {attack_samples} ({attack_samples/len(attack_data['labels'])*100:.1f}%)")
        
        # Add clean data metadata for IDS training
        attack_data['metadata']['clean_samples'] = int(clean_samples)
        attack_data['metadata']['training_ready'] = True
        attack_data['metadata']['data_source'] = 'digital_twin_sample_data'
        
        # Save attack dataset
        generator.save_attack_dataset(
            attack_data, 
            config_dict['data']['output_path'], 
            experiment
        )
        
        # Print comprehensive summary
        print("\n" + "="*60)
        print("ATTACK DATASET GENERATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Experiment: {experiment}")
        print(f"Data source: {attack_data['metadata']['data_source']}")
        print(f"Attack types: {attack_types}")
        print()
        print("DATASET COMPOSITION:")
        print(f"  Total samples: {attack_data['metadata']['total_samples']}")
        print(f"  Clean samples: {attack_data['metadata']['clean_samples']} ({(1-attack_data['metadata']['attack_ratio'])*100:.1f}%)")
        print(f"  Attack samples: {attack_data['metadata']['attack_samples']} ({attack_data['metadata']['attack_ratio']*100:.1f}%)")
        print()
        print("SAVED FILES:")
        print(f"  📁 {config_dict['data']['output_path']}/experiments/")
        print(f"    📊 {experiment}_attacks.npz (complete dataset for IDS training)")
        print(f"    ✅ {experiment}_clean_data.npz (clean data only)")
        print(f"    ⚠️  {experiment}_attack_data.npz (attack data only)")
        print(f"    📋 {experiment}_metadata.json (dataset information)")
        print()
        print("READY FOR IDS TRAINING: ✅")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Attack generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
