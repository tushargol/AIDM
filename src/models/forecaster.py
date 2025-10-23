"""
LSTM forecaster for time series prediction and anomaly detection.
Implements sequence-to-one prediction with residual-based detection.
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
    logging.warning("TensorFlow not available. LSTM forecaster functionality will be limited.")

logger = logging.getLogger(__name__)


class LSTMForecaster:
    """
    LSTM-based time series forecaster for power system anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LSTM forecaster.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config['models']['lstm_forecaster']
        self.model = None
        self.threshold = None
        self.history = None
        self.sequence_length = config['preprocessing']['sequence_window']
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM forecaster functionality")
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build the LSTM forecaster architecture.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        input_layer = layers.Input(shape=input_shape, name='sequence_input')
        
        # LSTM layer
        x = layers.LSTM(
            self.config['units'],
            return_sequences=False,  # Only return last output
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate'],
            name='lstm'
        )(input_layer)
        
        # Dense layers for prediction
        x = layers.Dense(
            self.config['units'] // 2,
            activation='relu',
            name='dense_1'
        )(x)
        
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Output layer (predict next timestep features)
        output = layers.Dense(
            input_shape[1],  # Same number of features as input
            activation='linear',
            name='output'
        )(x)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output, name='lstm_forecaster')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built LSTM forecaster: {input_shape} -> {input_shape[1]}")
        return self.model
    
    def train(self, 
              X_seq_train: np.ndarray,
              y_train: np.ndarray,
              X_seq_val: np.ndarray = None,
              y_val: np.ndarray = None,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the LSTM forecaster on benign data only.
        
        Args:
            X_seq_train: Training sequences (n_samples, sequence_length, n_features)
            y_train: Training targets (n_samples, n_features)
            X_seq_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(X_seq_train.shape[1:])
        
        # Prepare callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_seq_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_seq_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Prepare validation data
        validation_data = (X_seq_val, y_val) if X_seq_val is not None else None
        
        # Train model
        self.history = self.model.fit(
            X_seq_train, y_train,
            validation_data=validation_data,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callback_list,
            verbose=verbose
        )
        
        logger.info("LSTM forecaster training completed")
        return self.history.history
    
    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X_seq: Input sequences (n_samples, sequence_length, n_features)
            
        Returns:
            Predictions (n_samples, n_features)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X_seq, verbose=0)
    
    def compute_prediction_residuals(self, 
                                   X_seq: np.ndarray, 
                                   y_true: np.ndarray) -> np.ndarray:
        """
        Compute prediction residuals for anomaly detection.
        
        Args:
            X_seq: Input sequences
            y_true: True next timestep values
            
        Returns:
            Prediction residuals (MSE per sample)
        """
        # Get predictions
        y_pred = self.predict(X_seq)
        
        # Compute MSE per sample
        residuals = np.mean((y_true - y_pred) ** 2, axis=1)
        
        return residuals
    
    def fit_threshold(self, 
                     X_seq_val: np.ndarray,
                     y_val: np.ndarray,
                     method: str = 'percentile',
                     percentile: float = 95.0,
                     multiplier: float = None) -> float:
        """
        Fit detection threshold on validation data.
        
        Args:
            X_seq_val: Validation sequences
            y_val: Validation targets
            method: Threshold fitting method ('percentile' or 'std')
            percentile: Percentile for threshold (if method='percentile')
            multiplier: Standard deviation multiplier (if method='std')
            
        Returns:
            Fitted threshold value
        """
        if multiplier is None:
            multiplier = self.config.get('threshold_multiplier', 2.5)
        
        # Compute prediction residuals on validation set
        residuals = self.compute_prediction_residuals(X_seq_val, y_val)
        
        if method == 'percentile':
            self.threshold = np.percentile(residuals, percentile)
        elif method == 'std':
            self.threshold = np.mean(residuals) + multiplier * np.std(residuals)
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        logger.info(f"Fitted LSTM threshold: {self.threshold:.6f} (method: {method})")
        return self.threshold
    
    def detect_anomalies(self, 
                        X_seq: np.ndarray, 
                        y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using prediction residual threshold.
        
        Args:
            X_seq: Input sequences
            y_true: True next timestep values
            
        Returns:
            Tuple of (anomaly_flags, residuals)
        """
        if self.threshold is None:
            raise ValueError("Threshold must be fitted before anomaly detection")
        
        # Compute prediction residuals
        residuals = self.compute_prediction_residuals(X_seq, y_true)
        
        # Apply threshold
        anomaly_flags = residuals > self.threshold
        
        return anomaly_flags, residuals
    
    def predict_sequence(self, 
                        X_seq: np.ndarray, 
                        n_steps: int = 1) -> np.ndarray:
        """
        Predict multiple steps into the future.
        
        Args:
            X_seq: Input sequence (1, sequence_length, n_features)
            n_steps: Number of steps to predict
            
        Returns:
            Multi-step predictions (n_steps, n_features)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_seq = X_seq.copy()
        
        for _ in range(n_steps):
            # Predict next step
            next_pred = self.model.predict(current_seq, verbose=0)
            predictions.append(next_pred[0])
            
            # Update sequence for next prediction
            # Shift sequence and append prediction
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, :] = next_pred[0]
        
        return np.array(predictions)
    
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
        self.model.save(f"{filepath}_lstm_forecaster.h5")
        
        # Save threshold and config
        model_info = {
            'threshold': self.threshold,
            'config': self.config,
            'input_shape': self.model.input_shape,
            'sequence_length': self.sequence_length
        }
        
        joblib.dump(model_info, f"{filepath}_lstm_forecaster_info.pkl")
        
        logger.info(f"Saved LSTM forecaster model to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and threshold.
        
        Args:
            filepath: Path to the saved model (without extension)
        """
        filepath = Path(filepath)
        
        # Load model
        self.model = keras.models.load_model(f"{filepath}_lstm_forecaster.h5")
        
        # Load threshold and config
        model_info = joblib.load(f"{filepath}_lstm_forecaster_info.pkl")
        self.threshold = model_info['threshold']
        self.config.update(model_info['config'])
        self.sequence_length = model_info['sequence_length']
        
        logger.info(f"Loaded LSTM forecaster model from {filepath}")
    
    def evaluate_model(self, 
                      X_seq_test: np.ndarray,
                      y_test: np.ndarray,
                      y_labels: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate the LSTM forecaster performance.
        
        Args:
            X_seq_test: Test sequences
            y_test: Test targets
            y_labels: Test labels (0=normal, 1=anomaly) - optional
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Compute prediction residuals
        residuals = self.compute_prediction_residuals(X_seq_test, y_test)
        
        # Get predictions for additional metrics
        y_pred = self.predict(X_seq_test)
        
        metrics = {
            'mean_prediction_residual': np.mean(residuals),
            'std_prediction_residual': np.std(residuals),
            'mean_absolute_error': np.mean(np.abs(y_test - y_pred)),
            'root_mean_squared_error': np.sqrt(np.mean((y_test - y_pred) ** 2)),
            'threshold': self.threshold
        }
        
        if y_labels is not None and self.threshold is not None:
            # Compute detection metrics
            predictions = residuals > self.threshold
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics.update({
                'accuracy': accuracy_score(y_labels, predictions),
                'precision': precision_score(y_labels, predictions, zero_division=0),
                'recall': recall_score(y_labels, predictions, zero_division=0),
                'f1_score': f1_score(y_labels, predictions, zero_division=0),
                'auc_score': roc_auc_score(y_labels, residuals) if len(np.unique(y_labels)) > 1 else 0.0
            })
        
        return metrics
    
    def get_attention_weights(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Get attention weights for interpretability (if model has attention).
        This is a placeholder for future attention-based models.
        
        Args:
            X_seq: Input sequences
            
        Returns:
            Attention weights (placeholder: returns uniform weights)
        """
        # Placeholder implementation
        batch_size, seq_len, n_features = X_seq.shape
        # Return uniform attention weights
        return np.ones((batch_size, seq_len)) / seq_len


class BidirectionalLSTMForecaster(LSTMForecaster):
    """
    Bidirectional LSTM forecaster for improved sequence modeling.
    """
    
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build the bidirectional LSTM forecaster architecture.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        input_layer = layers.Input(shape=input_shape, name='sequence_input')
        
        # Bidirectional LSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.config['units'],
                return_sequences=False,
                dropout=self.config['dropout_rate'],
                recurrent_dropout=self.config['dropout_rate']
            ),
            name='bidirectional_lstm'
        )(input_layer)
        
        # Dense layers for prediction
        x = layers.Dense(
            self.config['units'],
            activation='relu',
            name='dense_1'
        )(x)
        
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        x = layers.Dense(
            self.config['units'] // 2,
            activation='relu',
            name='dense_2'
        )(x)
        
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        # Output layer
        output = layers.Dense(
            input_shape[1],
            activation='linear',
            name='output'
        )(x)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output, name='bidirectional_lstm_forecaster')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built bidirectional LSTM forecaster: {input_shape} -> {input_shape[1]}")
        return self.model


def create_lstm_forecaster(config: Dict[str, Any], bidirectional: bool = False) -> LSTMForecaster:
    """
    Factory function to create an LSTM forecaster.
    
    Args:
        config: Configuration dictionary
        bidirectional: Whether to use bidirectional LSTM
        
    Returns:
        LSTMForecaster instance
    """
    if bidirectional:
        return BidirectionalLSTMForecaster(config)
    else:
        return LSTMForecaster(config)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        'preprocessing': {
            'sequence_window': 10
        },
        'models': {
            'lstm_forecaster': {
                'units': 64,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 30
            }
        }
    }
    
    # Create synthetic sequence data
    np.random.seed(42)
    n_samples = 1000
    seq_length = 10
    n_features = 5
    
    # Generate sequences with some temporal patterns
    X_seq_train = np.random.randn(n_samples, seq_length, n_features)
    y_train = np.mean(X_seq_train, axis=1) + 0.1 * np.random.randn(n_samples, n_features)
    
    X_seq_val = np.random.randn(200, seq_length, n_features)
    y_val = np.mean(X_seq_val, axis=1) + 0.1 * np.random.randn(200, n_features)
    
    X_seq_test = np.random.randn(300, seq_length, n_features)
    y_test = np.mean(X_seq_test, axis=1) + 0.1 * np.random.randn(300, n_features)
    
    # Add some anomalies to test set
    y_test[250:] += np.random.randn(50, n_features) * 2
    y_labels = np.concatenate([np.zeros(250), np.ones(50)])
    
    # Create and train LSTM forecaster
    forecaster = create_lstm_forecaster(config)
    forecaster.train(X_seq_train, y_train, X_seq_val, y_val)
    
    # Fit threshold
    forecaster.fit_threshold(X_seq_val, y_val)
    
    # Evaluate
    metrics = forecaster.evaluate_model(X_seq_test, y_test, y_labels)
    print("Evaluation metrics:", metrics)
