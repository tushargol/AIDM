"""
IDS training module with baseline classifiers and adversarial training support.
Implements RandomForest, DNN, and LSTM classifiers with ART integration.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
import yaml
import click
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Try to import TensorFlow and ART
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. DNN and LSTM classifiers will be disabled.")

try:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification import TensorFlowV2Classifier, SklearnClassifier
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    logging.warning("ART not available. Adversarial training will be disabled.")

logger = logging.getLogger(__name__)


class BaselineIDS:
    """
    Baseline IDS classifiers for anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize baseline IDS.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['models']['ids_classifier']
        self.rf_model = None
        self.dnn_model = None
        self.lstm_model = None
        self.models = {}
        
    def train_random_forest(self, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          X_val: np.ndarray = None,
                          y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train Random Forest classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training results
        """
        self.rf_model = RandomForestClassifier(
            n_estimators=self.config['rf_n_estimators'],
            max_depth=self.config['rf_max_depth'],
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.rf_model.score(X_train, y_train)
        results = {'train_accuracy': train_score}
        
        if X_val is not None and y_val is not None:
            val_score = self.rf_model.score(X_val, y_val)
            results['val_accuracy'] = val_score
        
        # Cross-validation
        cv_scores = cross_val_score(self.rf_model, X_train, y_train, cv=5)
        results['cv_mean'] = np.mean(cv_scores)
        results['cv_std'] = np.std(cv_scores)
        
        self.models['random_forest'] = self.rf_model
        logger.info(f"Random Forest trained - Train: {train_score:.3f}, CV: {results['cv_mean']:.3f}±{results['cv_std']:.3f}")
        
        return results
    
    def build_dnn_model(self, input_dim: int) -> Model:
        """
        Build DNN classifier architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for DNN classifier")
        
        # Input layer
        input_layer = layers.Input(shape=(input_dim,), name='input')
        
        # Hidden layers
        x = input_layer
        for i, units in enumerate(self.config['dnn_hidden_layers']):
            x = layers.Dense(
                units, 
                activation='relu', 
                name=f'dense_{i}'
            )(x)
            x = layers.Dropout(
                self.config['dnn_dropout_rate'], 
                name=f'dropout_{i}'
            )(x)
        
        # Output layer
        output = layers.Dense(
            1, 
            activation='sigmoid', 
            name='output'
        )(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output, name='dnn_classifier')
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.config['dnn_learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_dnn(self, 
                  X_train: np.ndarray, 
                  y_train: np.ndarray,
                  X_val: np.ndarray = None,
                  y_val: np.ndarray = None,
                  verbose: int = 1) -> Dict[str, Any]:
        """
        Train DNN classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available for DNN training")
            return {}
        
        # Build model
        self.dnn_model = self.build_dnn_model(X_train.shape[1])
        
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
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train model
        history = self.dnn_model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.config['dnn_epochs'],
            batch_size=self.config['dnn_batch_size'],
            callbacks=callback_list,
            verbose=verbose
        )
        
        self.models['dnn'] = self.dnn_model
        logger.info("DNN classifier training completed")
        
        return history.history
    
    def build_lstm_classifier(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build LSTM classifier architecture.
        
        Args:
            input_shape: Shape of input sequences (seq_len, n_features)
            
        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM classifier")
        
        # Input layer
        input_layer = layers.Input(shape=input_shape, name='sequence_input')
        
        # LSTM layer
        x = layers.LSTM(
            64,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='lstm'
        )(input_layer)
        
        # Dense layers
        x = layers.Dense(32, activation='relu', name='dense_1')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output, name='lstm_classifier')
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train_lstm_classifier(self, 
                            X_seq_train: np.ndarray, 
                            y_train: np.ndarray,
                            X_seq_val: np.ndarray = None,
                            y_val: np.ndarray = None,
                            verbose: int = 1) -> Dict[str, Any]:
        """
        Train LSTM classifier.
        
        Args:
            X_seq_train: Training sequences
            y_train: Training labels
            X_seq_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available for LSTM training")
            return {}
        
        # Build model
        self.lstm_model = self.build_lstm_classifier(X_seq_train.shape[1:])
        
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
        history = self.lstm_model.fit(
            X_seq_train, y_train,
            validation_data=validation_data,
            epochs=30,
            batch_size=32,
            callbacks=callback_list,
            verbose=verbose
        )
        
        self.models['lstm'] = self.lstm_model
        logger.info("LSTM classifier training completed")
        
        return history.history
    
    def evaluate_models(self, 
                       X_test: np.ndarray, 
                       y_test: np.ndarray,
                       X_seq_test: np.ndarray = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features (tabular)
            y_test: Test labels
            X_seq_test: Test sequences (optional)
            
        Returns:
            Dictionary of evaluation results
        """
        results = {}
        
        # Evaluate Random Forest
        if self.rf_model is not None:
            y_pred_rf = self.rf_model.predict(X_test)
            y_prob_rf = self.rf_model.predict_proba(X_test)[:, 1]
            
            results['random_forest'] = {
                'predictions': y_pred_rf,
                'probabilities': y_prob_rf,
                'accuracy': np.mean(y_pred_rf == y_test),
                'classification_report': classification_report(y_test, y_pred_rf, output_dict=True)
            }
        
        # Evaluate DNN
        if self.dnn_model is not None:
            y_prob_dnn = self.dnn_model.predict(X_test, verbose=0).flatten()
            y_pred_dnn = (y_prob_dnn > 0.5).astype(int)
            
            results['dnn'] = {
                'predictions': y_pred_dnn,
                'probabilities': y_prob_dnn,
                'accuracy': np.mean(y_pred_dnn == y_test),
                'classification_report': classification_report(y_test, y_pred_dnn, output_dict=True)
            }
        
        # Evaluate LSTM
        if self.lstm_model is not None and X_seq_test is not None:
            y_prob_lstm = self.lstm_model.predict(X_seq_test, verbose=0).flatten()
            y_pred_lstm = (y_prob_lstm > 0.5).astype(int)
            
            results['lstm'] = {
                'predictions': y_pred_lstm,
                'probabilities': y_prob_lstm,
                'accuracy': np.mean(y_pred_lstm == y_test),
                'classification_report': classification_report(y_test, y_pred_lstm, output_dict=True)
            }
        
        return results
    
    def save_models(self, filepath: str) -> None:
        """
        Save all trained models.
        
        Args:
            filepath: Base filepath for saving models
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Random Forest
        if self.rf_model is not None:
            joblib.dump(self.rf_model, f"{filepath}_random_forest.pkl")
        
        # Save DNN
        if self.dnn_model is not None:
            self.dnn_model.save(f"{filepath}_dnn.h5")
        
        # Save LSTM
        if self.lstm_model is not None:
            self.lstm_model.save(f"{filepath}_lstm.h5")
        
        # Save config
        joblib.dump(self.config, f"{filepath}_config.pkl")
        
        logger.info(f"Saved IDS models to {filepath}")
    
    def load_models(self, filepath: str) -> None:
        """
        Load trained models.
        
        Args:
            filepath: Base filepath for loading models
        """
        filepath = Path(filepath)
        
        # Load Random Forest
        rf_path = f"{filepath}_random_forest.pkl"
        if Path(rf_path).exists():
            self.rf_model = joblib.load(rf_path)
            self.models['random_forest'] = self.rf_model
        
        # Load DNN
        dnn_path = f"{filepath}_dnn.h5"
        if Path(dnn_path).exists() and TF_AVAILABLE:
            self.dnn_model = keras.models.load_model(dnn_path)
            self.models['dnn'] = self.dnn_model
        
        # Load LSTM
        lstm_path = f"{filepath}_lstm.h5"
        if Path(lstm_path).exists() and TF_AVAILABLE:
            self.lstm_model = keras.models.load_model(lstm_path)
            self.models['lstm'] = self.lstm_model
        
        logger.info(f"Loaded IDS models from {filepath}")


class AdversarialTrainer:
    """
    Adversarial training for robust IDS classifiers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize adversarial trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.attack_params = config['attacks']['art_attacks']
        
    def generate_adversarial_examples(self, 
                                    model: Any,
                                    X: np.ndarray,
                                    y: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Generate adversarial examples using ART.
        
        Args:
            model: Trained model
            X: Input data
            y: Labels (optional)
            
        Returns:
            Dictionary of adversarial examples
        """
        if not ART_AVAILABLE:
            logger.error("ART not available for adversarial example generation")
            return {}
        
        # Wrap model for ART
        try:
            if hasattr(model, 'predict_proba'):
                # Sklearn model
                art_classifier = SklearnClassifier(model=model)
            else:
                # TensorFlow model
                art_classifier = TensorFlowV2Classifier(
                    model=model,
                    nb_classes=2,
                    input_shape=X.shape[1:],
                    loss_object=tf.keras.losses.BinaryCrossentropy()
                )
        except Exception as e:
            logger.error(f"Failed to wrap model for ART: {e}")
            return {}
        
        adversarial_examples = {}
        
        # FGSM attacks
        try:
            for eps in self.attack_params['fgsm_eps']:
                fgsm = FastGradientMethod(estimator=art_classifier, eps=eps)
                X_adv = fgsm.generate(x=X)
                adversarial_examples[f'fgsm_eps_{eps}'] = X_adv
                logger.info(f"Generated FGSM adversarial examples with eps={eps}")
        except Exception as e:
            logger.error(f"FGSM generation failed: {e}")
        
        # PGD attacks
        try:
            for eps in self.attack_params['pgd_eps']:
                pgd = ProjectedGradientDescent(
                    estimator=art_classifier,
                    eps=eps,
                    eps_step=self.attack_params['pgd_step_size'],
                    max_iter=self.attack_params['pgd_steps']
                )
                X_adv = pgd.generate(x=X)
                adversarial_examples[f'pgd_eps_{eps}'] = X_adv
                logger.info(f"Generated PGD adversarial examples with eps={eps}")
        except Exception as e:
            logger.error(f"PGD generation failed: {e}")
        
        return adversarial_examples
    
    def adversarial_training(self, 
                           model: Any,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray = None,
                           y_val: np.ndarray = None,
                           adv_ratio: float = 0.3) -> Any:
        """
        Perform adversarial training.
        
        Args:
            model: Model to train adversarially
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            adv_ratio: Ratio of adversarial examples to add
            
        Returns:
            Adversarially trained model
        """
        if not ART_AVAILABLE:
            logger.error("ART not available for adversarial training")
            return model
        
        # Generate adversarial examples
        adv_examples = self.generate_adversarial_examples(model, X_train, y_train)
        
        if not adv_examples:
            logger.warning("No adversarial examples generated, returning original model")
            return model
        
        # Combine clean and adversarial data
        X_combined = [X_train]
        y_combined = [y_train]
        
        n_adv_samples = int(len(X_train) * adv_ratio)
        
        for attack_name, X_adv in adv_examples.items():
            # Sample subset of adversarial examples
            indices = np.random.choice(len(X_adv), n_adv_samples, replace=False)
            X_combined.append(X_adv[indices])
            y_combined.append(y_train[indices])
        
        X_augmented = np.vstack(X_combined)
        y_augmented = np.hstack(y_combined)
        
        # Shuffle augmented dataset
        shuffle_indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[shuffle_indices]
        y_augmented = y_augmented[shuffle_indices]
        
        logger.info(f"Adversarial training with {len(X_augmented)} samples ({len(X_train)} clean + {len(X_augmented) - len(X_train)} adversarial)")
        
        # Retrain model
        if hasattr(model, 'fit'):
            # Sklearn model
            model.fit(X_augmented, y_augmented)
        else:
            # TensorFlow model
            model.fit(
                X_augmented, y_augmented,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=20,
                batch_size=32,
                verbose=1
            )
        
        return model


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@click.command()
@click.option('--config', default='config.yaml', help='Path to configuration file')
@click.option('--model', default='all', 
              type=click.Choice(['random_forest', 'dnn', 'lstm', 'all']),
              help='Model type to train')
@click.option('--data-path', default='./outputs/processed/processed_data.npz', 
              help='Path to processed data')
@click.option('--output', default=None, help='Output directory (overrides config)')
@click.option('--adversarial', is_flag=True, help='Enable adversarial training')
def main(config: str, model: str, data_path: str, output: str, adversarial: bool):
    """
    Train IDS classifiers.
    
    Example usage:
        python src/train_ids.py --model random_forest
        python src/train_ids.py --model dnn --adversarial
        python src/train_ids.py --model all --output ./custom_models
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
        # Load processed data
        logger.info(f"Loading processed data from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        X_seq_train = data['X_seq_train']
        X_seq_val = data['X_seq_val']
        X_seq_test = data['X_seq_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        
        # Create synthetic labels for demonstration (in practice, these would come from attack data)
        # Assume last 20% of data contains anomalies
        def create_labels(y_data):
            labels = np.zeros(len(y_data))
            anomaly_start = int(len(y_data) * 0.8)
            labels[anomaly_start:] = 1
            return labels.astype(int)
        
        y_train_labels = create_labels(y_train)
        y_val_labels = create_labels(y_val)
        y_test_labels = create_labels(y_test)
        
        # Initialize IDS
        ids = BaselineIDS(config_dict)
        
        # Train models
        if model in ['random_forest', 'all']:
            logger.info("Training Random Forest classifier...")
            rf_results = ids.train_random_forest(X_train, y_train_labels, X_val, y_val_labels)
            print(f"Random Forest Results: {rf_results}")
        
        if model in ['dnn', 'all'] and TF_AVAILABLE:
            logger.info("Training DNN classifier...")
            dnn_results = ids.train_dnn(X_train, y_train_labels, X_val, y_val_labels)
            print(f"DNN training completed")
        
        if model in ['lstm', 'all'] and TF_AVAILABLE:
            logger.info("Training LSTM classifier...")
            lstm_results = ids.train_lstm_classifier(X_seq_train, y_train_labels, X_seq_val, y_val_labels)
            print(f"LSTM training completed")
        
        # Adversarial training
        if adversarial and ART_AVAILABLE:
            logger.info("Performing adversarial training...")
            adv_trainer = AdversarialTrainer(config_dict)
            
            if ids.dnn_model is not None:
                ids.dnn_model = adv_trainer.adversarial_training(
                    ids.dnn_model, X_train, y_train_labels, X_val, y_val_labels
                )
        
        # Evaluate models
        logger.info("Evaluating models...")
        results = ids.evaluate_models(X_test, y_test_labels, X_seq_test)
        
        # Save models
        model_path = Path(config_dict['data']['output_path']) / "models" / "ids_classifiers"
        ids.save_models(model_path)
        
        # Print results
        print("\n" + "="*60)
        print("IDS TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        
        for model_name, model_results in results.items():
            print(f"\n{model_name.upper()} RESULTS:")
            print(f"  Accuracy: {model_results['accuracy']:.3f}")
            report = model_results['classification_report']
            print(f"  Precision: {report['1']['precision']:.3f}")
            print(f"  Recall: {report['1']['recall']:.3f}")
            print(f"  F1-Score: {report['1']['f1-score']:.3f}")
        
        print(f"\nModels saved to: {model_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"IDS training failed: {e}")
        raise


if __name__ == "__main__":
    main()
