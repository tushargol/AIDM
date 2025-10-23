# AIDM: Anomaly and Intrusion Detection Model for Power Systems

A comprehensive framework for detecting anomalies and intrusions in power system measurements using advanced machine learning techniques including autoencoders, LSTM forecasters, randomized transformations, and adversarial training.

## Overview

This project implements an **AIDM (Anomaly and Intrusion Detection Model)** system that combines multiple detection techniques:

- **Autoencoder**: Reconstruction-based anomaly detection for tabular features
- **LSTM Forecaster**: Time series prediction with residual-based detection
- **Randomized Transformations**: Consistency-based detection using multiple data transformations
- **Fusion Classifier**: Meta-classifier combining all detection components
- **Adversarial Training**: Robust training against adversarial attacks (FDIA, FGSM, PGD)

## Project Structure

```
./
├── requirements.txt                    # Python dependencies
├── config.yaml                        # Configuration parameters
├── README.md                          # This file
├── notebooks/
│   ├── notebook_0_explore.ipynb      # Dataset exploration
│   ├── notebook_1_preprocess.ipynb   # Data preprocessing pipeline
│   ├── notebook_2_attacks.ipynb      # Attack generation demos
│   └── notebook_3_train_evaluate.ipynb # Model training and evaluation
├── src/
│   ├── data_loader.py                 # Data loading utilities
│   ├── preprocess.py                  # Preprocessing pipeline with CLI
│   ├── pandapower_utils.py            # Power system modeling and Jacobian computation
│   ├── attacks.py                     # Attack generation functions with CLI
│   ├── train_ids.py                   # IDS training and adversarial training
│   ├── evaluate.py                    # Evaluation metrics and visualization
│   ├── models/
│   │   ├── autoencoder.py            # Autoencoder implementation
│   │   └── forecaster.py             # LSTM forecaster
│   └── pipeline/
│       └── aidm.py                   # Complete AIDM pipeline
└── outputs/
    ├── processed/                     # Processed data arrays
    ├── models/                        # Saved models and scalers
    ├── experiments/                   # Attack datasets and metadata
    └── reports/                       # Plots and evaluation results
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv aidm_env
source aidm_env/bin/activate  # On Windows: aidm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import pandapower as pp; print('Pandapower installed')"
```

### 2. Configuration

Edit `config.yaml` to adjust hyperparameters:

```yaml
# Key parameters you might want to modify
compute:
  small_data_mode: true      # Use sample dataset only
  
models:
  autoencoder:
    latent_dim: 32            # Autoencoder bottleneck size
    epochs: 50                # Training epochs
  
  lstm_forecaster:
    units: 64                 # LSTM units
    epochs: 30                # Training epochs

attacks:
  art_attacks:
    fgsm_eps: [0.01, 0.05, 0.1]  # FGSM attack strengths
```

### 3. Run Complete Pipeline

```bash
# Step 1: Preprocess data
python src/preprocess.py --mode small_data

# Step 2: Generate attack datasets
python src/attacks.py --type all --experiment comprehensive_attacks

# Step 3: Train IDS classifiers
python src/train_ids.py --model all

# Step 4: Run evaluation (via notebook)
jupyter notebook notebooks/notebook_3_train_evaluate.ipynb
```

## Usage Examples

### Data Preprocessing

```bash
# Basic preprocessing (small data mode)
python src/preprocess.py --mode small_data

# Full preprocessing with custom config
python src/preprocess.py --config custom_config.yaml --mode full_data --output ./custom_output

# Check preprocessing results
python -c "
import numpy as np
data = np.load('./outputs/processed/processed_data.npz')
print('Processed data shapes:')
for key in data.files:
    if hasattr(data[key], 'shape'):
        print(f'  {key}: {data[key].shape}')
"
```

### Attack Generation

```bash
# Generate FDIA attacks
python src/attacks.py --type fdia --samples 1000 --experiment fdia_test

# Generate all attack types
python src/attacks.py --type all --samples 2000 --experiment comprehensive_attacks

# Generate temporal stealth attacks
python src/attacks.py --type temporal_stealth --experiment stealth_attacks

# Check generated attacks
python -c "
import numpy as np
data = np.load('./outputs/experiments/comprehensive_attacks_attacks.npz')
print(f'Attack dataset: {data[\"measurements\"].shape}')
print(f'Attack ratio: {np.mean(data[\"labels\"]):.2%}')
"
```

### Model Training

```bash
# Train Random Forest classifier
python src/train_ids.py --model random_forest --data-path ./outputs/processed/processed_data.npz

# Train deep neural network with adversarial training
python src/train_ids.py --model dnn --adversarial

# Train all models
python src/train_ids.py --model all --output ./outputs/models

# Train LSTM classifier only
python src/train_ids.py --model lstm
```

### Evaluation and Analysis

```bash
# Run evaluation via Python script
python -c "
import sys
sys.path.append('src')
from evaluate import create_evaluator
import yaml
import numpy as np

# Load config and run evaluation
with open('config.yaml') as f:
    config = yaml.safe_load(f)

evaluator = create_evaluator(config)
print('Evaluator ready for analysis')
"

# Or use Jupyter notebooks for interactive analysis
jupyter notebook notebooks/notebook_3_train_evaluate.ipynb
```

## Advanced Usage

### Custom Attack Parameters

```python
# Custom FDIA attack
python -c "
import sys
sys.path.append('src')
from attacks import AttackGenerator
from pandapower_utils import create_power_system_model
import yaml
import numpy as np

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Create attack generator
generator = AttackGenerator(config)
generator.initialize_power_model()

# Generate custom FDIA
measurements = np.random.randn(100, 20) + 1.0  # Synthetic measurements
z_attacked = generator.make_fdia(measurements[0], attack_magnitude=0.2)
print(f'Original: {measurements[0][:5]}')
print(f'Attacked: {z_attacked[:5]}')
"
```

### Custom Model Training

```python
# Train autoencoder with custom parameters
python -c "
import sys
sys.path.append('src')
from models.autoencoder import create_autoencoder
import numpy as np
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Modify config
config['models']['autoencoder']['latent_dim'] = 16
config['models']['autoencoder']['epochs'] = 100

# Create and train
ae = create_autoencoder(config)
X_train = np.random.randn(1000, 50)
ae.build_model(50)
history = ae.train(X_train, verbose=1)
print('Training completed')
"
```

### Pipeline Integration

```python
# Run complete AIDM pipeline
python -c "
import sys
sys.path.append('src')
from pipeline.aidm import create_aidm_pipeline
import numpy as np
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Create synthetic data
X_tabular = np.random.randn(500, 20)
X_sequences = np.random.randn(500, 10, 20)
y_sequences = np.mean(X_sequences, axis=1)

# Create and run pipeline
pipeline = create_aidm_pipeline(config)
# Note: In practice, you would load trained models first
# pipeline.load_models('./outputs/models')

print('AIDM pipeline ready')
"
```

## Performance Monitoring

### Check Training Progress

```bash
# Monitor model training logs
tail -f outputs/aidm.log

# Check model performance
python -c "
import joblib
import numpy as np

# Load saved model info
try:
    ae_info = joblib.load('./outputs/models/autoencoder_autoencoder_info.pkl')
    print(f'Autoencoder threshold: {ae_info[\"threshold\"]:.6f}')
except:
    print('Autoencoder not trained yet')

try:
    lstm_info = joblib.load('./outputs/models/lstm_forecaster_lstm_forecaster_info.pkl')
    print(f'LSTM threshold: {lstm_info[\"threshold\"]:.6f}')
except:
    print('LSTM forecaster not trained yet')
"
```

### Evaluation Reports

```bash
# Generate evaluation report
python -c "
import sys
sys.path.append('src')
from evaluate import create_evaluator
import yaml
import numpy as np
import json

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

evaluator = create_evaluator(config)

# Check for existing reports
import os
reports_dir = './outputs/reports'
if os.path.exists(reports_dir):
    reports = [f for f in os.listdir(reports_dir) if f.endswith('_summary.json')]
    print(f'Available reports: {reports}')
    
    if reports:
        with open(os.path.join(reports_dir, reports[0])) as f:
            summary = json.load(f)
        print(f'Latest report: {summary[\"experiment_name\"]}')
        print(f'Components: {summary[\"components_evaluated\"]}')
else:
    print('No reports generated yet')
"
```

## Troubleshooting

### Common Issues

1. **TensorFlow Installation Issues**
   ```bash
   # For CPU-only installation
   pip install tensorflow-cpu
   
   # For GPU support (requires CUDA)
   pip install tensorflow-gpu
   ```

2. **Pandapower Installation Issues**
   ```bash
   # If pandapower fails to install
   pip install pandapower --no-deps
   pip install pandas numpy scipy matplotlib networkx
   ```

3. **Memory Issues with Large Datasets**
   ```bash
   # Use small data mode
   python src/preprocess.py --mode small_data
   
   # Reduce batch sizes in config.yaml
   # models.autoencoder.batch_size: 32
   # models.lstm_forecaster.batch_size: 16
   ```

4. **ART (Adversarial Robustness Toolbox) Issues**
   ```bash
   # Install specific version
   pip install adversarial-robustness-toolbox==1.12.0
   
   # Skip adversarial training if ART unavailable
   python src/train_ids.py --model dnn  # without --adversarial flag
   ```

### Debug Mode

```bash
# Enable debug logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Run your commands here
"

# Check data shapes and types
python src/preprocess.py --mode small_data 2>&1 | grep -E "(shape|INFO|ERROR)"
```

## References and Documentation

### Key Components

- **Autoencoder**: Dense neural network for reconstruction-based anomaly detection
- **LSTM Forecaster**: Recurrent neural network for time series prediction
- **Randomized Transformations**: Multiple data augmentations for consistency checking
- **Fusion Classifier**: Meta-learning approach combining multiple detectors
- **FDIA (False Data Injection Attack)**: Power system specific attack using measurement Jacobian
- **ART Integration**: Adversarial robustness evaluation using state-of-the-art attacks

### Configuration Parameters

Key parameters in `config.yaml`:

- `compute.small_data_mode`: Use sample dataset only (faster development)
- `preprocessing.sequence_window`: LSTM sequence length (default: 10)
- `models.autoencoder.latent_dim`: Autoencoder bottleneck size (default: 32)
- `transformations.n_transforms`: Number of randomized transformations (default: 5)
- `fusion.method`: Fusion strategy ("weighted_sum", "voting", "meta_classifier")

### Output Files

- `outputs/processed/processed_data.npz`: Preprocessed and split datasets
- `outputs/models/`: Trained model files (.h5, .pkl)
- `outputs/experiments/`: Generated attack datasets
- `outputs/reports/`: Evaluation plots and metrics

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Digital Twin Dataset](https://github.com/caltech-netlab/digital-twin-dataset) from Caltech NetLab
- Adversarial Robustness Toolbox (ART) by IBM
- TensorFlow and Keras teams
- Pandapower development team

---

**Note**: This framework is designed for research and educational purposes. For production deployment in critical infrastructure, additional security measures and validation are required.
