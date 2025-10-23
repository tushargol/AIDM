"""
Autoencoder model for anomaly detection in power system measurements.
Implements dense autoencoder with reconstruction error-based detection.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Autoencoder functionality will be limited.")

logger = logging.getLogger(__name__)


class DenseAutoencoder:
    """
    Dense autoencoder for tabular power system data anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the autoencoder.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config['models']['autoencoder']
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None
        self.history = None
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for autoencoder functionality")
    
    def build_model(self, input_dim: int) -> Model:
        """
        Build the autoencoder architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        input_layer = layers.Input(shape=(input_dim,), name='input')
        
        # Encoder
        x = input_layer
        hidden_layers = self.config['hidden_layers']
        
        for i, units in enumerate(hidden_layers):
            x = layers.Dense(
                units, 
                activation='relu', 
                name=f'encoder_dense_{i}'
            )(x)
            x = layers.Dropout(
                self.config['dropout_rate'], 
                name=f'encoder_dropout_{i}'
            )(x)
        
        # Latent layer
        latent = layers.Dense(
            self.config['latent_dim'], 
            activation='relu', 
            name='latent'
        )(x)
        
        # Decoder
        x = latent
        for i, units in enumerate(reversed(hidden_layers)):
            x = layers.Dense(
                units, 
                activation='relu', 
                name=f'decoder_dense_{i}'
            )(x)
            x = layers.Dropout(
                self.config['dropout_rate'], 
                name=f'decoder_dropout_{i}'
            )(x)
        
        # Output layer
        output = layers.Dense(
            input_dim, 
            activation='linear', 
            name='output'
        )(x)
        
        # Create models
        self.model = Model(inputs=input_layer, outputs=output, name='autoencoder')
        self.encoder = Model(inputs=input_layer, outputs=latent, name='encoder')
        
        # Decoder model (for generating from latent space)
        decoder_input = layers.Input(shape=(self.config['latent_dim'],))
        decoder_layers = self.model.layers[-(len(hidden_layers) * 2 + 1):]
        x = decoder_input
        for layer in decoder_layers:
            if 'decoder' in layer.name or layer.name == 'output':
                x = layer(x)
        self.decoder = Model(inputs=decoder_input, outputs=x, name='decoder')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built autoencoder: {input_dim} -> {self.config['latent_dim']} -> {input_dim}")
        return self.model
    
    def train(self, 
              X_train: np.ndarray, 
              X_val: np.ndarray = None,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the autoencoder on benign data only.
        
        Args:
            X_train: Training data (benign only)
            X_val: Validation data (optional)
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Prepare callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model (autoencoder learns to reconstruct input)
        validation_data = (X_val, X_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, X_train,  # Input and target are the same
            validation_data=validation_data,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callback_list,
            verbose=verbose
        )
        
        logger.info("Autoencoder training completed")
        return self.history.history
    
    def compute_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction errors for input data.
        
        Args:
            X: Input data
            
        Returns:
            Reconstruction errors (MSE per sample)
        """
        if self.model is None:
            raise ValueError("Model must be trained before computing errors")
        
        # Get reconstructions
        X_reconstructed = self.model.predict(X, verbose=0)
        
        # Compute MSE per sample
        reconstruction_errors = np.mean((X - X_reconstructed) ** 2, axis=1)
        
        return reconstruction_errors
    
    def fit_threshold(self, 
                     X_val: np.ndarray, 
                     method: str = 'percentile',
                     percentile: float = 95.0,
                     multiplier: float = None) -> float:
        """
        Fit detection threshold on validation data.
        
        Args:
            X_val: Validation data (benign)
            method: Threshold fitting method ('percentile' or 'std')
            percentile: Percentile for threshold (if method='percentile')
            multiplier: Standard deviation multiplier (if method='std')
            
        Returns:
            Fitted threshold value
        """
        if multiplier is None:
            multiplier = self.config.get('threshold_multiplier', 2.0)
        
        # Compute reconstruction errors on validation set
        errors = self.compute_reconstruction_errors(X_val)
        
        if method == 'percentile':
            self.threshold = np.percentile(errors, percentile)
        elif method == 'std':
            self.threshold = np.mean(errors) + multiplier * np.std(errors)
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        logger.info(f"Fitted threshold: {self.threshold:.6f} (method: {method})")
        return self.threshold
    
    def detect_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using reconstruction error threshold.
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (anomaly_flags, reconstruction_errors)
        """
        if self.threshold is None:
            raise ValueError("Threshold must be fitted before anomaly detection")
        
        # Compute reconstruction errors
        errors = self.compute_reconstruction_errors(X)
        
        # Apply threshold
        anomaly_flags = errors > self.threshold
        
        return anomaly_flags, errors
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent space representation of input data.
        
        Args:
            X: Input data
            
        Returns:
            Latent representations
        """
        if self.encoder is None:
            raise ValueError("Model must be built before getting latent representation")
        
        return self.encoder.predict(X, verbose=0)
    
    def generate_from_latent(self, z: np.ndarray) -> np.ndarray:
        """
        Generate data from latent space representation.
        
        Args:
            z: Latent space vectors
            
        Returns:
            Generated data
        """
        if self.decoder is None:
            raise ValueError("Model must be built before generation")
        
        return self.decoder.predict(z, verbose=0)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and threshold.
        
        Args:
            filepath: Path to save the model (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(f"{filepath}_autoencoder.h5")
        
        # Save threshold and config
        model_info = {
            'threshold': self.threshold,
            'config': self.config,
            'input_shape': self.model.input_shape,
            'latent_dim': self.config['latent_dim']
        }
        
        joblib.dump(model_info, f"{filepath}_autoencoder_info.pkl")
        
        logger.info(f"Saved autoencoder model to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and threshold.
        
        Args:
            filepath: Path to the saved model (without extension)
        """
        filepath = Path(filepath)
        
        # Load model
        self.model = keras.models.load_model(f"{filepath}_autoencoder.h5")
        
        # Load threshold and config
        model_info = joblib.load(f"{filepath}_autoencoder_info.pkl")
        self.threshold = model_info['threshold']
        self.config.update(model_info['config'])
        
        # Rebuild encoder and decoder
        input_layer = self.model.input
        latent_layer = None
        
        # Find latent layer
        for layer in self.model.layers:
            if layer.name == 'latent':
                latent_layer = layer.output
                break
        
        if latent_layer is not None:
            self.encoder = Model(inputs=input_layer, outputs=latent_layer)
            
            # Rebuild decoder
            decoder_input = layers.Input(shape=(model_info['latent_dim'],))
            decoder_layers = []
            found_latent = False
            
            for layer in self.model.layers:
                if layer.name == 'latent':
                    found_latent = True
                    continue
                if found_latent and ('decoder' in layer.name or layer.name == 'output'):
                    decoder_layers.append(layer)
            
            x = decoder_input
            for layer in decoder_layers:
                x = layer(x)
            
            self.decoder = Model(inputs=decoder_input, outputs=x)
        
        logger.info(f"Loaded autoencoder model from {filepath}")
    
    def evaluate_model(self, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate the autoencoder performance.
        
        Args:
            X_test: Test data
            y_test: Test labels (0=normal, 1=anomaly) - optional
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Compute reconstruction errors
        errors = self.compute_reconstruction_errors(X_test)
        
        metrics = {
            'mean_reconstruction_error': np.mean(errors),
            'std_reconstruction_error': np.std(errors),
            'threshold': self.threshold
        }
        
        if y_test is not None and self.threshold is not None:
            # Compute detection metrics
            predictions = errors > self.threshold
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics.update({
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, zero_division=0),
                'recall': recall_score(y_test, predictions, zero_division=0),
                'f1_score': f1_score(y_test, predictions, zero_division=0),
                'auc_score': roc_auc_score(y_test, errors) if len(np.unique(y_test)) > 1 else 0.0
            })
        
        return metrics


def create_autoencoder(config: Dict[str, Any]) -> DenseAutoencoder:
    """
    Factory function to create an autoencoder.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DenseAutoencoder instance
    """
    return DenseAutoencoder(config)


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
            }
        }
    }
    
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    X_val = np.random.randn(200, 20)
    X_test = np.random.randn(300, 20)
    
    # Add some anomalies to test set
    X_test[250:] += np.random.randn(50, 20) * 2
    y_test = np.concatenate([np.zeros(250), np.ones(50)])
    
    # Create and train autoencoder
    ae = create_autoencoder(config)
    ae.train(X_train, X_val)
    
    # Fit threshold
    ae.fit_threshold(X_val)
    
    # Evaluate
    metrics = ae.evaluate_model(X_test, y_test)
    print("Evaluation metrics:", metrics)
