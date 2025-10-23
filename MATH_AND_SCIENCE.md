# Mathematical and Scientific Foundations of AIDM

## Table of Contents
1. [Overview](#overview)
2. [Power System State Estimation](#power-system-state-estimation)
3. [Anomaly Detection Methods](#anomaly-detection-methods)
4. [Attack Models](#attack-models)
5. [Machine Learning Components](#machine-learning-components)
6. [Fusion and Meta-Learning](#fusion-and-meta-learning)
7. [Adversarial Robustness](#adversarial-robustness)
8. [Mathematical Notation](#mathematical-notation)
9. [References](#references)

## Overview

The **Anomaly and Intrusion Detection Model (AIDM)** framework combines multiple mathematical approaches for detecting anomalies and cyber-attacks in power system measurements. This document provides the mathematical foundations and scientific principles underlying each component.

## Power System State Estimation

### Mathematical Model

Power system state estimation is formulated as a weighted least squares problem:

```
minimize J(x) = [z - h(x)]ᵀ R⁻¹ [z - h(x)]
```

Where:
- `x ∈ ℝⁿ` is the state vector (voltage magnitudes and phase angles)
- `z ∈ ℝᵐ` is the measurement vector
- `h(x): ℝⁿ → ℝᵐ` is the nonlinear measurement function
- `R ∈ ℝᵐˣᵐ` is the measurement error covariance matrix

### Measurement Jacobian

The measurement Jacobian matrix is crucial for FDIA generation:

```
H = ∂h(x)/∂x |ₓ₌ₓ̂
```

For power flow measurements:
- **Active power injection**: `Pᵢ = Vᵢ Σⱼ Vⱼ [Gᵢⱼ cos(θᵢ - θⱼ) + Bᵢⱼ sin(θᵢ - θⱼ)]`
- **Reactive power injection**: `Qᵢ = Vᵢ Σⱼ Vⱼ [Gᵢⱼ sin(θᵢ - θⱼ) - Bᵢⱼ cos(θᵢ - θⱼ)]`
- **Active power flow**: `Pᵢⱼ = Vᵢ² gᵢⱼ - VᵢVⱼ [gᵢⱼ cos(θᵢ - θⱼ) + bᵢⱼ sin(θᵢ - θⱼ)]`

Where:
- `Vᵢ, θᵢ` are voltage magnitude and phase angle at bus i
- `Gᵢⱼ + jBᵢⱼ` are elements of the bus admittance matrix
- `gᵢⱼ + jbᵢⱼ` are branch admittance parameters

## Anomaly Detection Methods

### 1. Autoencoder-Based Detection

#### Architecture
The autoencoder consists of an encoder `f: ℝᵈ → ℝᵏ` and decoder `g: ℝᵏ → ℝᵈ`:

```
Encoder: h = f(x) = σ(W₁x + b₁)
Decoder: x̂ = g(h) = σ(W₂h + b₂)
```

#### Loss Function
The reconstruction loss is:

```
L(x, x̂) = ||x - x̂||₂² = Σᵢ (xᵢ - x̂ᵢ)²
```

#### Anomaly Score
For a test sample `x`, the anomaly score is:

```
s(x) = ||x - g(f(x))||₂²
```

Anomaly detection: `s(x) > τ` where `τ` is the threshold determined from training data.

#### Threshold Selection
Using the 95th percentile of reconstruction errors on clean training data:

```
τ = percentile₉₅({s(xᵢ) : xᵢ ∈ X_train})
```

### 2. LSTM Forecaster

#### Model Architecture
The LSTM forecaster predicts the next measurement based on historical sequences:

```
hₜ = LSTM(xₜ, hₜ₋₁)
ŷₜ₊₁ = Wₒhₜ + bₒ
```

#### Prediction Error
The residual-based anomaly score is:

```
r(t) = ||y(t) - ŷ(t)||₂²
```

#### Temporal Modeling
For a sequence window of length `w`:

```
X_seq = [x(t-w+1), x(t-w+2), ..., x(t)]
y_pred = LSTM(X_seq)
```

### 3. Randomized Transformations

#### Transformation Functions
Multiple randomized transformations `Tᵢ: ℝᵈ → ℝᵈ` are applied:

1. **Gaussian Noise**: `T₁(x) = x + ε`, where `ε ~ N(0, σ²I)`
2. **Feature Dropout**: `T₂(x) = x ⊙ m`, where `m` is a binary mask
3. **Scaling**: `T₃(x) = αx`, where `α ~ U(0.9, 1.1)`
4. **Rotation**: `T₄(x) = Rx`, where `R` is a random rotation matrix
5. **Permutation**: `T₅(x) = P(x)`, where `P` is a random permutation

#### Consistency Score
For `n` transformations, the consistency score is:

```
C(x) = (1/n) Σᵢ ||f(x) - f(Tᵢ(x))||₂²
```

Where `f` is a feature extraction function (e.g., autoencoder encoder).

## Attack Models

### 1. False Data Injection Attack (FDIA)

#### Mathematical Formulation
An FDIA constructs malicious measurements that bypass bad data detection:

```
z_a = z + a
```

Where the attack vector `a` satisfies:

```
a = Hc
```

For some attack vector `c` on the state variables.

#### Stealth Condition
The attack is undetectable by traditional χ² bad data detection if:

```
||z_a - h(x̂_a)||²_R⁻¹ ≤ χ²_threshold
```

#### Attack Construction Algorithm
1. Choose target state perturbation `c`
2. Compute attack vector: `a = Hc`
3. Apply physical constraints and rate limiting
4. Generate attacked measurements: `z_a = z + a`

### 2. Temporal Stealth Attack

#### Gradual Ramp Model
The attack evolves gradually over time to avoid detection:

```
a(t) = a(t-1) + Δa(t)
```

Where `Δa(t)` is constrained by:

```
||Δa(t)||∞ ≤ δ_max
```

#### Temporal Consistency
The attack maintains temporal correlation:

```
Corr(z(t), z(t-1)) ≈ Corr(z_a(t), z_a(t-1))
```

### 3. Replay Attack

#### Data Substitution Model
Replace current measurements with historical data:

```
z_a(t:t+w) = z(s:s+w)
```

Where `s < t` is the source time window.

## Machine Learning Components

### 1. Deep Neural Network Classifier

#### Architecture
Multi-layer perceptron with ReLU activations:

```
h₁ = ReLU(W₁x + b₁)
h₂ = ReLU(W₂h₁ + b₂)
...
y = sigmoid(Wₗhₗ₋₁ + bₗ)
```

#### Loss Function
Binary cross-entropy for anomaly classification:

```
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

### 2. Random Forest

#### Ensemble Prediction
For `T` trees, the prediction is:

```
ŷ = (1/T) Σₜ fₜ(x)
```

#### Feature Importance
Gini importance for feature `j`:

```
I(j) = Σₜ Σₙ p(n) * G(n) * I(split_n uses feature j)
```

Where `G(n)` is the Gini impurity at node `n`.

## Fusion and Meta-Learning

### Weighted Fusion

#### Score Combination
Multiple detector scores are combined:

```
s_fusion = Σᵢ wᵢ sᵢ(x)
```

Where `wᵢ` are learned weights and `sᵢ(x)` are individual detector scores.

#### Weight Optimization
Weights are optimized to minimize validation error:

```
w* = argmin_w Σⱼ L(y_j, Σᵢ wᵢ sᵢ(x_j))
```

### Meta-Classifier Approach

#### Feature Vector Construction
Meta-features from individual detectors:

```
φ(x) = [s₁(x), s₂(x), ..., sₖ(x), conf₁(x), ..., confₖ(x)]
```

#### Meta-Learning
Train a classifier on meta-features:

```
ŷ_meta = f_meta(φ(x))
```

## Adversarial Robustness

### 1. Fast Gradient Sign Method (FGSM)

#### Attack Generation
```
x_adv = x + ε * sign(∇ₓ L(θ, x, y))
```

Where:
- `ε` is the perturbation magnitude
- `L(θ, x, y)` is the loss function
- `∇ₓ L` is the gradient with respect to input

### 2. Projected Gradient Descent (PGD)

#### Iterative Attack
```
x_adv^(t+1) = Π_S(x_adv^(t) + α * sign(∇ₓ L(θ, x_adv^(t), y)))
```

Where `Π_S` projects onto the constraint set `S = {x' : ||x' - x||∞ ≤ ε}`.

### 3. Adversarial Training

#### Robust Optimization
```
min_θ E_{(x,y)~D} [max_{δ∈S} L(θ, x + δ, y)]
```

This minimax formulation trains the model to be robust against worst-case perturbations.

## Mathematical Notation

### Sets and Spaces
- `ℝ`: Real numbers
- `ℝⁿ`: n-dimensional real vector space
- `ℝᵐˣⁿ`: m×n real matrices
- `𝒳`: Input space
- `𝒴`: Output space

### Vectors and Matrices
- `x, y, z`: Vectors (lowercase bold)
- `A, B, H`: Matrices (uppercase bold)
- `xᵢ`: i-th element of vector x
- `Aᵢⱼ`: (i,j)-th element of matrix A
- `x̂`: Estimated/predicted value
- `x̃`: Perturbed/noisy value

### Operations
- `||·||₂`: L2 (Euclidean) norm
- `||·||∞`: L∞ (maximum) norm
- `⊙`: Element-wise (Hadamard) product
- `∇`: Gradient operator
- `∂f/∂x`: Partial derivative
- `E[·]`: Expected value
- `Var[·]`: Variance
- `Cov[·]`: Covariance

### Probability and Statistics
- `P(·)`: Probability
- `p(·)`: Probability density function
- `N(μ, σ²)`: Normal distribution
- `U(a, b)`: Uniform distribution
- `χ²`: Chi-squared distribution

## References

### Power System Security
1. Monticelli, A. (1999). *State Estimation in Electric Power Systems: A Generalized Approach*
2. Liu, Y., Ning, P., & Reiter, M. K. (2011). False data injection attacks against state estimation in electric power grids. *ACM Transactions on Information and System Security*

### Anomaly Detection
3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press

### Adversarial Machine Learning
5. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. *arXiv preprint arXiv:1412.6572*
6. Madry, A., et al. (2017). Towards deep learning models resistant to adversarial attacks. *arXiv preprint arXiv:1706.06083*

### Power System Cybersecurity
7. Ten, C. W., Liu, C. C., & Manimaran, G. (2008). Vulnerability assessment of cybersecurity for SCADA systems. *IEEE Transactions on Power Systems*
8. Liang, G., Weller, S. R., Zhao, J., Luo, F., & Dong, Z. Y. (2017). The 2015 Ukraine blackout: Implications for false data injection attacks. *IEEE Transactions on Power Systems*

---

*This document provides the mathematical foundations for understanding and implementing the AIDM framework. For implementation details, refer to the main README.md and source code documentation.*
