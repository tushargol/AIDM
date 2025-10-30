"""
AIDM (Anomaly and Intrusion Detection Model) Pipeline.
Integrates autoencoder, randomized transformations, LSTM forecaster, and fusion classifier.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# Import our models
import sys
sys.path.append('..')
from models.autoencoder import DenseAutoencoder
from models.forecaster import LSTMForecaster

logger = logging.getLogger(__name__)


class RandomizedTransformations:
    """
    Randomized transformations for consistency-based anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize randomized transformations.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['transformations']
        self.n_transforms = self.config['n_transforms']
        self.noise_std = self.config['noise_std']
        self.dropout_rate = self.config['dropout_rate']
    
    def apply_transformations(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Apply multiple randomized transformations to input data.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            List of transformed data arrays
        """
        transformations = []
        
        for i in range(self.n_transforms):
            # Set random seed for reproducible transformations
            np.random.seed(42 + i)
            
            X_transformed = X.copy()
            
            # Apply noise
            noise = np.random.normal(0, self.noise_std, X.shape)
            X_transformed += noise
            
            # Apply dropout (randomly set some features to zero)
            dropout_mask = np.random.random(X.shape) > self.dropout_rate
            X_transformed *= dropout_mask
            
            # Apply small rotations (for numerical stability)
            if X.shape[1] > 1:
                # Create small rotation matrix
                angle = np.random.uniform(-0.1, 0.1)  # Small angle in radians
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                # Apply rotation to pairs of features
                for j in range(0, X.shape[1] - 1, 2):
                    if j + 1 < X.shape[1]:
                        x1, x2 = X_transformed[:, j], X_transformed[:, j + 1]
                        X_transformed[:, j] = cos_a * x1 - sin_a * x2
                        X_transformed[:, j + 1] = sin_a * x1 + cos_a * x2
            
            transformations.append(X_transformed)
        
        return transformations
    
    def compute_consistency_score(self, 
                                 model_predict_fn: Callable,
                                 X: np.ndarray,
                                 method: str = 'std') -> np.ndarray:
        """
        Compute transformation consistency scores.
        
        Args:
            model_predict_fn: Function that takes data and returns predictions
            X: Input data
            method: Consistency metric ('std', 'entropy', 'variance')
            
        Returns:
            Consistency scores (higher = more inconsistent = more anomalous)
        """
        # Apply transformations
        transformations = self.apply_transformations(X)
        
        # Get predictions for each transformation
        predictions = []
        for X_transformed in transformations:
            pred = model_predict_fn(X_transformed)
            predictions.append(pred)
        
        predictions = np.array(predictions)  # Shape: (n_transforms, n_samples, ...)
        
        if method == 'std':
            # Standard deviation across transformations
            consistency_scores = np.std(predictions, axis=0)
            if len(consistency_scores.shape) > 1:
                consistency_scores = np.mean(consistency_scores, axis=1)
        
        elif method == 'variance':
            # Variance across transformations
            consistency_scores = np.var(predictions, axis=0)
            if len(consistency_scores.shape) > 1:
                consistency_scores = np.mean(consistency_scores, axis=1)
        
        elif method == 'entropy':
            # Entropy-based consistency (for classification outputs)
            if predictions.shape[-1] > 1:  # Multi-class predictions
                # Convert to probabilities if needed
                if np.any(predictions < 0) or np.any(predictions > 1):
                    from scipy.special import softmax
                    predictions = softmax(predictions, axis=-1)
                
                # Compute entropy for each sample across transformations
                consistency_scores = []
                for i in range(predictions.shape[1]):
                    sample_preds = predictions[:, i, :]  # (n_transforms, n_classes)
                    mean_pred = np.mean(sample_preds, axis=0)
                    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8))
                    consistency_scores.append(entropy)
                consistency_scores = np.array(consistency_scores)
            else:
                # Fall back to std for single output
                consistency_scores = np.std(predictions, axis=0).flatten()
        
        else:
            raise ValueError(f"Unknown consistency method: {method}")
        
        return consistency_scores


class SurrogatePredictor:
    """
    Lightweight surrogate predictor to mimic IDS outputs for transformation consistency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize surrogate predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train surrogate predictor on IDS outputs.
        
        Args:
            X: Input features
            y: IDS predictions/probabilities
        """
        from sklearn.neural_network import MLPClassifier
        
        # Simple MLP surrogate
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        
        self.model.fit(X, y)
        logger.info("Trained surrogate predictor")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using surrogate model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Surrogate model must be trained first")
        
        return self.model.predict_proba(X)


class FusionClassifier:
    """
    Fusion classifier for combining multiple anomaly detection scores.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fusion classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['fusion']
        self.method = self.config['method']
        self.weights = np.array(self.config['weights'])
        self.meta_classifier = None
        self.thresholds = {}
    
    def set_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Set detection thresholds for each component.
        
        Args:
            thresholds: Dictionary of component thresholds
        """
        self.thresholds = thresholds
    
    def weighted_fusion(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Weighted sum fusion of anomaly scores.
        
        Args:
            scores: Dictionary of anomaly scores
            
        Returns:
            Fused anomaly scores
        """
        # Normalize scores by their thresholds
        normalized_scores = []
        component_names = ['autoencoder', 'transformations', 'lstm']
        
        for i, component in enumerate(component_names):
            if component in scores and component in self.thresholds:
                normalized = scores[component] / self.thresholds[component]
                normalized_scores.append(normalized * self.weights[i])
        
        if not normalized_scores:
            raise ValueError("No valid scores provided for fusion")
        
        # Weighted sum
        fused_scores = np.sum(normalized_scores, axis=0)
        return fused_scores
    
    def voting_fusion(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Majority voting fusion of binary decisions.
        
        Args:
            scores: Dictionary of anomaly scores
            
        Returns:
            Voting-based decisions
        """
        decisions = []
        component_names = ['autoencoder', 'transformations', 'lstm']
        
        for component in component_names:
            if component in scores and component in self.thresholds:
                decision = (scores[component] > self.thresholds[component]).astype(int)
                decisions.append(decision)
        
        if not decisions:
            raise ValueError("No valid scores provided for voting")
        
        # Majority vote
        votes = np.sum(decisions, axis=0)
        return (votes > len(decisions) / 2).astype(int)
    
    def train_meta_classifier(self, 
                            scores: Dict[str, np.ndarray], 
                            labels: np.ndarray) -> None:
        """
        Train meta-classifier for fusion.
        
        Args:
            scores: Dictionary of anomaly scores for training
            labels: True labels
        """
        # Prepare feature matrix
        feature_matrix = []
        component_names = ['autoencoder', 'transformations', 'lstm']
        
        for component in component_names:
            if component in scores:
                feature_matrix.append(scores[component])
        
        if not feature_matrix:
            raise ValueError("No scores provided for meta-classifier training")
        
        X_meta = np.column_stack(feature_matrix)
        
        # Train logistic regression meta-classifier
        self.meta_classifier = LogisticRegression(random_state=42)
        self.meta_classifier.fit(X_meta, labels)
        
        logger.info("Trained meta-classifier for fusion")
    
    def meta_classifier_fusion(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Meta-classifier fusion of anomaly scores.
        
        Args:
            scores: Dictionary of anomaly scores
            
        Returns:
            Meta-classifier predictions
        """
        if self.meta_classifier is None:
            raise ValueError("Meta-classifier must be trained first")
        
        # Prepare feature matrix
        feature_matrix = []
        component_names = ['autoencoder', 'transformations', 'lstm']
        
        for component in component_names:
            if component in scores:
                feature_matrix.append(scores[component])
        
        if not feature_matrix:
            raise ValueError("No scores provided for meta-classifier")
        
        X_meta = np.column_stack(feature_matrix)
        return self.meta_classifier.predict_proba(X_meta)[:, 1]
    
    def fuse_decisions(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse anomaly detection decisions based on configured method.
        
        Args:
            scores: Dictionary of anomaly scores
            
        Returns:
            Fused decisions or scores
        """
        if self.method == 'weighted_sum':
            return self.weighted_fusion(scores)
        elif self.method == 'voting':
            return self.voting_fusion(scores)
        elif self.method == 'meta_classifier':
            return self.meta_classifier_fusion(scores)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")


class AIDMPipeline:
    """
    Complete AIDM pipeline integrating all components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AIDM pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.autoencoder = None
        self.lstm_forecaster = None
        self.transformations = RandomizedTransformations(config)
        self.surrogate_predictor = SurrogatePredictor(config)
        self.fusion_classifier = FusionClassifier(config)
        self.thresholds = {}
        
    def load_models(self, model_path: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            model_path: Path to saved models
        """
        model_path = Path(model_path)
        
        # Load autoencoder
        try:
            self.autoencoder = DenseAutoencoder(self.config)
            self.autoencoder.load_model(model_path / "autoencoder")
            logger.info("Loaded autoencoder model")
        except Exception as e:
            logger.warning(f"Failed to load autoencoder: {e}")
        
        # Load LSTM forecaster
        try:
            self.lstm_forecaster = LSTMForecaster(self.config)
            self.lstm_forecaster.load_model(model_path / "lstm_forecaster")
            logger.info("Loaded LSTM forecaster model")
        except Exception as e:
            logger.warning(f"Failed to load LSTM forecaster: {e}")
        
        # Load thresholds
        try:
            threshold_file = model_path / "thresholds.pkl"
            if threshold_file.exists():
                self.thresholds = joblib.load(threshold_file)
                self.fusion_classifier.set_thresholds(self.thresholds)
                logger.info("Loaded detection thresholds")
        except Exception as e:
            logger.warning(f"Failed to load thresholds: {e}")
    
    def run_pipeline_on_set(self, 
                           X_tabular: np.ndarray,
                           X_sequences: np.ndarray,
                           y_sequences: np.ndarray = None,
                           return_scores: bool = True) -> Dict[str, Any]:
        """
        Run complete AIDM pipeline on a dataset.
        
        Args:
            X_tabular: Tabular features (n_samples, n_features)
            X_sequences: Sequence features (n_samples, seq_len, n_features)
            y_sequences: Target sequences for LSTM (optional)
            return_scores: Whether to return individual component scores
            
        Returns:
            Dictionary containing detection results and scores
        """
        start_time = time.time()
        results = {}
        scores = {}
        
        # 1. Autoencoder anomaly detection
        if self.autoencoder is not None:
            ae_start = time.time()
            ae_flags, ae_errors = self.autoencoder.detect_anomalies(X_tabular)
            ae_time = time.time() - ae_start
            
            scores['autoencoder'] = ae_errors
            results['autoencoder_flags'] = ae_flags
            results['autoencoder_time'] = ae_time
            logger.info(f"Autoencoder detection: {np.sum(ae_flags)}/{len(ae_flags)} anomalies")
        
        # 2. Randomized transformations consistency
        if self.autoencoder is not None:
            trans_start = time.time()
            
            # Use autoencoder reconstruction as surrogate predictor
            def ae_predict_fn(X):
                return self.autoencoder.compute_reconstruction_errors(X)
            
            consistency_scores = self.transformations.compute_consistency_score(
                ae_predict_fn, X_tabular
            )
            
            # Apply threshold if available
            if 'transformations' in self.thresholds:
                trans_flags = consistency_scores > self.thresholds['transformations']
            else:
                # Use adaptive threshold
                trans_threshold = np.percentile(consistency_scores, 95)
                trans_flags = consistency_scores > trans_threshold
            
            trans_time = time.time() - trans_start
            
            scores['transformations'] = consistency_scores
            results['transformations_flags'] = trans_flags
            results['transformations_time'] = trans_time
            logger.info(f"Transformation consistency: {np.sum(trans_flags)}/{len(trans_flags)} anomalies")
        
        # 3. LSTM forecaster anomaly detection
        if self.lstm_forecaster is not None and y_sequences is not None:
            lstm_start = time.time()
            lstm_flags, lstm_residuals = self.lstm_forecaster.detect_anomalies(
                X_sequences, y_sequences
            )
            lstm_time = time.time() - lstm_start
            
            scores['lstm'] = lstm_residuals
            results['lstm_flags'] = lstm_flags
            results['lstm_time'] = lstm_time
            logger.info(f"LSTM forecaster: {np.sum(lstm_flags)}/{len(lstm_flags)} anomalies")
        
        # 4. Fusion decision
        if len(scores) > 1:
            fusion_start = time.time()
            
            try:
                fused_scores = self.fusion_classifier.fuse_decisions(scores)
                
                # Apply fusion threshold
                if self.fusion_classifier.method == 'weighted_sum':
                    fusion_threshold = 1.0  # Normalized threshold
                    fused_flags = fused_scores > fusion_threshold
                elif self.fusion_classifier.method == 'meta_classifier':
                    fusion_threshold = 0.5
                    fused_flags = fused_scores > fusion_threshold
                else:  # voting
                    fused_flags = fused_scores.astype(bool)
                
                fusion_time = time.time() - fusion_start
                
                results['fusion_flags'] = fused_flags
                results['fusion_scores'] = fused_scores
                results['fusion_time'] = fusion_time
                logger.info(f"Fusion decision: {np.sum(fused_flags)}/{len(fused_flags)} anomalies")
                
            except Exception as e:
                logger.error(f"Fusion failed: {e}")
                # Fall back to majority vote of available components
                available_flags = [results[key] for key in results.keys() if key.endswith('_flags')]
                if available_flags:
                    votes = np.sum(available_flags, axis=0)
                    results['fusion_flags'] = votes > len(available_flags) / 2
        
        # Total pipeline time
        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['per_sample_time'] = total_time / len(X_tabular)
        
        if return_scores:
            results['component_scores'] = scores
        
        logger.info(f"AIDM pipeline completed in {total_time:.3f}s ({results['per_sample_time']*1000:.2f}ms per sample)")
        return results
    
    def evaluate_pipeline(self, 
                         X_tabular: np.ndarray,
                         X_sequences: np.ndarray,
                         y_sequences: np.ndarray,
                         y_labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate AIDM pipeline performance.
        
        Args:
            X_tabular: Tabular features
            X_sequences: Sequence features
            y_sequences: Target sequences
            y_labels: True anomaly labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Run pipeline
        results = self.run_pipeline_on_set(X_tabular, X_sequences, y_sequences)
        
        # Compute metrics for each component
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {}
        
        for component in ['autoencoder', 'transformations', 'lstm', 'fusion']:
            flag_key = f'{component}_flags'
            if flag_key in results:
                flags = results[flag_key]
                
                metrics[f'{component}_accuracy'] = accuracy_score(y_labels, flags)
                metrics[f'{component}_precision'] = precision_score(y_labels, flags, zero_division=0)
                metrics[f'{component}_recall'] = recall_score(y_labels, flags, zero_division=0)
                metrics[f'{component}_f1'] = f1_score(y_labels, flags, zero_division=0)
                
                # AUC using scores if available
                if 'component_scores' in results and component in results['component_scores']:
                    scores = results['component_scores'][component]
                    if len(np.unique(y_labels)) > 1:
                        metrics[f'{component}_auc'] = roc_auc_score(y_labels, scores)
        
        # Performance metrics
        metrics['total_time'] = results['total_time']
        metrics['per_sample_time_ms'] = results['per_sample_time'] * 1000
        
        return metrics


def create_aidm_pipeline(config: Dict[str, Any]) -> AIDMPipeline:
    """
    Factory function to create AIDM pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AIDMPipeline instance
    """
    return AIDMPipeline(config)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        'models': {
            'autoencoder': {
                'latent_dim': 16,
                'hidden_layers': [64, 32],
                'dropout_rate': 0.1,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50
            },
            'lstm_forecaster': {
                'units': 64,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 30
            }
        },
        'preprocessing': {
            'sequence_window': 10
        },
        'transformations': {
            'n_transforms': 5,
            'noise_std': 0.001,
            'dropout_rate': 0.02
        },
        'fusion': {
            'method': 'weighted_sum',
            'weights': [0.4, 0.3, 0.3]
        }
    }
    
    # Create realistic synthetic PMU data
    np.random.seed(42)
    
    # Generate physics-based synthetic data using data loader
    from data_loader import DigitalTwinDataLoader
    
    # Initialize data loader
    dataset_path = "./digital-twin-dataset/digital-twin-dataset"  # Placeholder path
    loader = DigitalTwinDataLoader(dataset_path, use_api=False)
    
    # Generate realistic PMU data for training (benign only)
    print("Generating physics-based synthetic PMU data...")
    train_duration = 0.5  # hours
    test_duration = 0.2   # hours
    sampling_rate = 1.0   # Hz
    n_buses = 12
    
    # Generate training data (benign only)
    train_data = loader.generate_synthetic_data(
        duration_hours=train_duration,
        sampling_rate=sampling_rate,
        n_buses=n_buses,
        physics_based=True
    )
    
    # Generate test data (benign only, will add attacks later)
    test_data = loader.generate_synthetic_data(
        duration_hours=test_duration,
        sampling_rate=sampling_rate,
        n_buses=n_buses,
        physics_based=True
    )
    
    # Process data through preprocessing pipeline
    from preprocess import DataPreprocessor
    preprocessor = DataPreprocessor(config)
    
    # Combine and preprocess training data
    combined_train_data = {
        'magnitude': train_data['magnitude'],
        'phasor': train_data['phasor'],
        'topology': train_data['topology']
    }
    
    # Align and extract features for training data
    train_aligned = preprocessor.align_and_resample(combined_train_data)
    train_clean = preprocessor.handle_missing_data(train_aligned)
    train_features = preprocessor.extract_features(train_clean)
    
    # Create sequences for training
    X_seq_train, y_seq_train, X_tab_train = preprocessor.create_sequences(train_features)
    
    # Process test data similarly
    combined_test_data = {
        'magnitude': test_data['magnitude'],
        'phasor': test_data['phasor'],
        'topology': test_data['topology']
    }
    
    test_aligned = preprocessor.align_and_resample(combined_test_data)
    test_clean = preprocessor.handle_missing_data(test_aligned)
    test_features = preprocessor.extract_features(test_clean)
    
    # Create sequences for test data
    X_seq_test, y_seq_test, X_tab_test = preprocessor.create_sequences(test_features)
    
    # Add anomalies to test data only (50% of test samples)
    n_test = len(X_tab_test)
    n_anomalies = n_test // 2
    anomaly_indices = np.random.choice(n_test, n_anomalies, replace=False)
    
    # Add realistic anomalies (not just random noise)
    attack_magnitude = 0.1  # 10% of standard deviation
    for idx in anomaly_indices:
        # Add structured anomalies that could represent real attacks
        std_vals = np.std(X_tab_test, axis=0)
        attack_vector = attack_magnitude * std_vals * np.random.randn(X_tab_test.shape[1])
        X_tab_test[idx] += attack_vector
        y_seq_test[idx] += attack_magnitude * np.std(y_seq_test, axis=0) * np.random.randn(y_seq_test.shape[1])
    
    # Combine training and test data for pipeline demonstration
    X_tabular = np.vstack([X_tab_train, X_tab_test])
    X_sequences = np.vstack([X_seq_train, X_seq_test])
    y_sequences = np.vstack([y_seq_train, y_seq_test])
    
    # Create labels (0 for all training data, 0/1 for test data)
    n_train = len(X_tab_train)
    n_samples = len(X_tabular)
    y_labels = np.zeros(n_samples)
    y_labels[n_train + anomaly_indices] = 1
    
    # Update dimensions
    n_features = X_tabular.shape[1]
    seq_length = X_sequences.shape[1]
    
    print(f"Generated realistic PMU data:")
    print(f"  Training samples: {n_train} (all benign)")
    print(f"  Test samples: {n_test} ({n_test - n_anomalies} benign + {n_anomalies} anomalous)")
    print(f"  Features: {n_features}")
    print(f"  Sequence length: {seq_length}")
    
    # Split training and test data for proper training
    X_tabular_benign = X_tab_train
    X_sequences_benign = X_seq_train
    y_sequences_benign = y_seq_train
    
    # Create and initialize pipeline
    pipeline = create_aidm_pipeline(config)
    
    # For demo, create simple models with thresholds
    pipeline.autoencoder = DenseAutoencoder(config)
    pipeline.autoencoder.build_model(n_features)
    pipeline.autoencoder.threshold = 1.0
    
    pipeline.lstm_forecaster = LSTMForecaster(config)
    pipeline.lstm_forecaster.build_model((seq_length, n_features))
    pipeline.lstm_forecaster.threshold = 1.0
    
    pipeline.thresholds = {
        'autoencoder': 1.0,
        'transformations': 0.5,
        'lstm': 1.0
    }
    pipeline.fusion_classifier.set_thresholds(pipeline.thresholds)
    
    # Train models on benign data only (first n_train samples)
    print(f"Training models on {n_train} benign samples...")
    if hasattr(pipeline.autoencoder, 'train'):
        pipeline.autoencoder.train(X_tabular_benign)
    if hasattr(pipeline.lstm_forecaster, 'train'):
        pipeline.lstm_forecaster.train(X_sequences_benign, y_sequences_benign)
    
    # Run pipeline on all data (including test data with anomalies)
    print(f"Testing pipeline on {n_samples} samples ({n_train} benign + {n_test - n_anomalies} benign + {n_anomalies} anomalous)...")
    results = pipeline.run_pipeline_on_set(X_tabular, X_sequences, y_sequences)
    
    print("Pipeline results:")
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}, anomalies: {np.sum(value) if value.dtype == bool else 'N/A'}")
        else:
            print(f"  {key}: {value}")
    
    # Evaluate performance on test data only
    if 'fusion_flags' in results:
        test_predictions = results['fusion_flags'][n_train:]  # Only test portion
        test_labels = y_labels[n_train:]  # Only test portion
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(test_labels, test_predictions)
        precision = precision_score(test_labels, test_predictions, zero_division=0)
        recall = recall_score(test_labels, test_predictions, zero_division=0)
        f1 = f1_score(test_labels, test_predictions, zero_division=0)
        
        print(f"\nTest Performance (on {n_test} test samples):")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  True anomalies: {np.sum(test_labels)}")
        print(f"  Detected anomalies: {np.sum(test_predictions)}")
