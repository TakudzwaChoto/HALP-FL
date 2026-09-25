# HALP-FL: Risk-Responsive Privacy Allocation with Threshold Encryption for Federated Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

HALP-FL is a reference implementation of a hybrid federated learning framework that integrates **client-specific differential privacy** with **threshold homomorphic encryption** and a **closed-loop risk-adaptive privacy controller**. The framework's central architectural contribution is a *dual-use risk signal*: the same anomaly score that triggers conditional cryptographic verification within a round is also fed back to modulate the differential-privacy budget of subsequent rounds. This closes a loop that prior hybrid DP/HE work leaves open.

### Important scope note

The framework protects two disjoint parameter subsets with two different mechanisms:

- The **DP-protected subset** $\mathcal{W}_{\mathrm{DP}}$ receives calibrated Gaussian noise and satisfies $(\varepsilon, \delta)$-differential privacy under the accounting implemented in this repository.
- The **HE-protected subset** $\mathcal{W}_{\mathrm{HE}}$ is protected by $(t,n)$ threshold Paillier confidentiality during aggregation. It does **not** carry DP noise, and the composite release is not $(\varepsilon, \delta)$-DP as a whole. Whenever this repository or any associated write-up reports accuracy "at $\varepsilon = 1$", this refers to the DP budget of $\mathcal{W}_{\mathrm{DP}}$ only.

The verification layer is a **proof-of-concept design**: the Bulletproof range proof and the Paillier ciphertext are linked by a cross-group linkage proof whose concrete sound and efficient construction is left to future work.

---

## Key Features

- **Dual-use risk signal** (primary contribution): a single anomaly score drives both (a) selective cryptographic verification within a round and (b) risk-weighted privacy budget hardening in subsequent rounds.
- **Client-specific adaptive DP**: per-client, per-round privacy budgets $\varepsilon_{i,t}$ that depend on data sensitivity $s_i$, non-IID degree $h_i$, training progress $p_t$, and a running risk estimate $r_i$.
- **Threshold Paillier aggregation**: $(t, n)$ threshold encryption, so decryption requires at least $t$ clients to collaborate. Standard Paillier fails with a single colluding client; HALP-FL resists up to $t - 1$.
- **Risk-activated verification**: Bulletproof range proofs generated only when the anomaly score exceeds a threshold (activated in fewer than 5% of rounds in our experiments).
- **Efficiency optimizations**: mean-$k$ clipping, top-$k$ sparsification, and random DP/HE assignment reduce encrypted parameter count by 70–80% relative to encrypting all parameters.

---

## Quick Start

### Prerequisites

- Python 3.8 or higher (tested on 3.10 and 3.11)
- GPU with CUDA support (recommended; CPU fallback supported)
- 32 GB+ RAM for full experiments

### Installation

```bash
git clone https://github.com/TakudzwaChoto/HALP-FL.git
cd HALP-FL

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running the code

The repository supports two modes.

**Fast preview mode (smoke test, does not reproduce the reported results).** This mode runs the full experimental pipeline with a small number of rounds and clients to verify that the code is functional. It prints illustrative numbers only and is not intended to reproduce the reported results.

```bash
python halp_fl.py --preview
```

**Full training mode (reproduces the reported results).** This mode runs the complete end-to-end federated training pipeline with the reference configuration (30 rounds, 5 clients, 10 local epochs for MNIST and 20 for FashionMNIST / CIFAR-10). On a single NVIDIA V100, it takes approximately 3–4 hours per dataset, and approximately 10 hours for all three datasets.

```bash
python halp_fl.py --full
```

We strongly recommend the full mode when evaluating the reported results. The preview mode exists only for pipeline verification.

---

## Dependencies

All required packages are listed in `requirements.txt`:

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
pandas>=1.3.0
tqdm>=4.64.0
pillow>=9.0.0
```

Threshold Paillier operations are implemented in pure Python on top of the `python-paillier` library (version 1.4.0). Bulletproof verification uses a Rust implementation with Python bindings.

---

## Repository Structure

```
HALP-FL/
├── halp_fl.py                   # Main implementation and experiment orchestrator
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # MIT license
│
├── data/                        # Datasets (auto-downloaded on first run)
│   ├── MNIST/
│   ├── FashionMNIST/
│   └── CIFAR-10/
│
├── figures/                     # Generated visualizations
│   ├── overall_comparison.png
│   ├── privacy_utility_tradeoff.png
│   ├── communication_overhead.png
│   ├── collusion_resistance.png
│   ├── gradient_leakage.png
│   └── ablation_study.png
│
└── results/                     # JSON-formatted experiment results
    ├── halpfl_results_accuracy_comparison_*.json
    ├── halpfl_results_privacy_utility_*.json
    └── halpfl_results_ablation_*.json
```

---

## Experiments

The framework executes seven experiments:

| # | Experiment | Output figure |
|---|---|---|
| 1 | Accuracy comparison at $\varepsilon = 1$ and $\varepsilon = 3$ | `overall_comparison.png` |
| 2 | Privacy–utility trade-off across $\varepsilon \in \{0.5, 1, 2, 3, 4, 5\}$ | `privacy_utility_tradeoff.png` |
| 3 | Encryption/decryption time for key lengths {256, 512, 1024} bits | `communication_overhead.png` |
| 4 | Communication overhead per round, normalized to FedAvg | `communication_overhead.png` |
| 5 | Collusion resistance vs. number of malicious clients | `collusion_resistance.png` |
| 6 | Gradient inversion attack resilience | `gradient_leakage.png` |
| 7 | Ablation study on CIFAR-10 at $\varepsilon = 1$ | `ablation_study.png` |

All experiments use the Dirichlet partition $\alpha = 0.5$ for the non-IID setting.

---

## Configuration

The default parameters match the reference configuration. They can be modified in `halp_fl.py`:

```python
# Federated learning settings
n_clients       = 5           # Number of participating clients
n_rounds        = 30          # Communication rounds
local_epochs    = {'mnist': 10, 'fashion': 20, 'cifar10': 20}
batch_size      = 64
learning_rate   = 0.001       # SGD learning rate

# Privacy settings
epsilon_values  = [1, 2, 3]   # Cumulative DP budget for the DP-protected subset
delta           = 1e-5        # DP failure probability
adjustment_a    = 0.8         # Round-adaptivity factor
hardening_beta  = 0.5         # Closed-loop hardening strength
smoothing_eta   = 0.3         # Risk estimate smoothing factor
anomaly_lambda  = 3           # Anomaly threshold for risk detection

# Threshold HE settings
threshold_t     = 3           # t-of-n threshold (t = floor(N/2) + 1)
key_lengths     = [256, 512, 1024]   # Paillier key lengths to evaluate
```

To run with a custom configuration, edit these values and invoke the full mode:

```bash
python halp_fl.py --full
```

---

## Implementation Notes

### The deployed allocation is a privacy-priority heuristic

The framework distinguishes between two allocations:

- **Utility-optimal reference**: minimizes total noise variance subject to the budget constraint; assigns *larger* budgets to high-sensitivity clients. This is the mathematically optimal allocation under idealized assumptions.
- **Deployed privacy-priority heuristic**: assigns *smaller* budgets to high-sensitivity clients, and includes the closed-loop hardening factor $\gamma(r_i)$. This is what HALP-FL actually uses.

The two allocations optimize different objectives and are not approximations of one another. The code implements the deployed heuristic. The utility-optimal reference is provided as a comparison in the ablation experiment.

The per-client, per-round budget is computed as:

```python
def compute_epsilon(i, t, epsilon, T, a, r_i, beta, s_i, h_i, p_t):
    """
    Compute the client-specific privacy budget with closed-loop hardening.

        gamma(r_i)  = 1 - beta * r_i
        w_tilde_t   = max( (t^a / sum(tau^a)) * (1 - s_i) * (1 - h_i) * p_t, 0.01 )
        eps_i,t     = epsilon * gamma(r_i) * w_tilde_t / sum_tau(w_tilde_tau)
    """
    gamma      = 1.0 - beta * r_i
    w_tilde    = max((t ** a) / sum(tau ** a for tau in range(1, T + 1))
                     * (1 - s_i) * (1 - h_i) * p_t, 0.01)
    normalizer = sum(w_tilde_at(i, tau) for tau in range(1, T + 1))
    return epsilon * gamma * w_tilde / normalizer
```

Note that $\gamma(r_i)$ is the closed-loop factor driven by the running risk estimate $r_i \in [0, 1]$. The risk estimate is updated each round according to:

```python
def update_risk_estimate(r_i, z_t, eta, lambda_):
    """
    First-order low-pass filter on the anomaly signal.

        r_i <- (1 - eta) * r_i + eta * min(z_t / (2 * lambda), 1)
    """
    z_norm = min(z_t / (2.0 * lambda_), 1.0)
    return (1.0 - eta) * r_i + eta * z_norm
```

### Threshold Paillier

The threshold Paillier implementation follows the distributed key generation protocol of Fouque, Poupard, and Stern (2000), which is robust against up to $t - 1$ malicious participants during key generation. Decryption is non-interactive: each client produces a partial decryption independently, and the server combines any $t$ valid partial decryptions. In our experiments with $N = 5$, $t = 3$, we observed fewer than 1% decryption failures per round due to client dropouts.

### The verification layer

The Bulletproof range proof and the Paillier ciphertext are linked by a cross-group linkage proof whose concrete sound construction is left to future work. The verification code in this repository implements the range-proof component and the ciphertext-side check **separately**; it does not implement the linkage proof, and therefore does not provide the integrity guarantee that a sound linkage proof would provide. The verification layer should be treated as a proof-of-concept design rather than a deployable cryptographic mechanism.

---

## Output

### Generated figures

| File | Content |
|---|---|
| `overall_comparison.png` | Accuracy at $\varepsilon = 1$ and $\varepsilon = 3$ across all methods |
| `privacy_utility_tradeoff.png` | Accuracy vs. $\varepsilon$ on CIFAR-10 |
| `communication_overhead.png` | Normalized communication per round |
| `collusion_resistance.png` | Attack success rate vs. number of malicious clients |
| `gradient_leakage.png` | Reconstruction quality under different privacy settings |
| `ablation_study.png` | Component contributions on CIFAR-10 |

### Result files

Results are saved in JSON format with the following structure:

```json
{
  "experiment": "accuracy_comparison",
  "timestamp": "2025-09-25T14:32:11Z",
  "mode": "full",
  "results": {
    "mnist": {
      "halp": [accuracy_values],
      "adphe": [accuracy_values],
      "clfldp": [accuracy_values],
      "dp_fl": [accuracy_values]
    },
    "fashion": {...},
    "cifar10": {...}
  },
  "hyperparameters": {
    "n_clients": 5,
    "n_rounds": 30,
    "local_epochs": {"mnist": 10, "fashion": 20, "cifar10": 20},
    "learning_rate": 0.001,
    "adjustment_a": 0.8,
    "hardening_beta": 0.5,
    "smoothing_eta": 0.3,
    "anomaly_lambda": 3,
    "threshold_t": 3,
    "epsilon": 1
  }
}
```

The `mode` field is either `"full"` (reproduces the reported results) or `"preview"` (illustrative only). Only `"full"` mode results should be compared against the reported numbers.

---

## Reproducing the reported results

To reproduce the numbers reported in this repository:

```bash
python halp_fl.py --full
```

This runs the complete training pipeline with the reference hyperparameters and writes results to `results/` and figures to `figures/`.

**Expected results at $\varepsilon = 1$ (DP-protected subset):**

| Dataset | HALP-FL | ADPHE-FL (reimpl.) |
|---|---|---|
| MNIST | 99.69% | 96.00% |
| FashionMNIST | 93.47% | 88.00% |
| CIFAR-10 | 76.17% | 74.00% |

**Expected results at $\varepsilon = 3$ (DP-protected subset):**

| Dataset | HALP-FL | ADPHE-FL (reimpl.) | CLFLDP | DP-FL |
|---|---|---|---|---|
| MNIST | 99.71% | 96.80% | 95.00% | 90.00% |
| FashionMNIST | 93.89% | 89.10% | 81.00% | 80.00% |
| CIFAR-10 | 77.03% | 75.20% | 72.00% | 56.00% |

Results are means of five independent runs with different random seeds. Standard deviations on accuracy metrics are below 2% across runs.

**Reproducibility note.** The re-implemented ADPHE-FL baseline follows the unified protocol described below: identical hardware, model architectures, data partitioning (Dirichlet $\alpha = 0.5$), five-seed repetition, a common Bernoulli(0.5) DP/HE split, and a common mean-$k$ clipping and top-$k$ sparsification pipeline. The two methods differ only in how they allocate the DP budget across clients and rounds — which is precisely the effect the comparison is designed to isolate.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| CUDA out of memory | Set `device = 'cpu'` in `halp_fl.py`, or reduce `batch_size` |
| Dataset download failure | Check internet connection; manually download from [PyTorch Vision](https://pytorch.org/vision/stable/datasets.html) and place in `data/` |
| Figures not displaying on a headless server | Add `import matplotlib; matplotlib.use('Agg')` at the top of `halp_fl.py`; figures will be saved as PNG files |
| Slow execution | Full mode is CPU/GPU intensive; use `--preview` for a quick smoke test |
| Threshold decryption fails | Ensure at least `t` clients are online and reachable in the same round |

### Running on headless servers

```python
import matplotlib
matplotlib.use('Agg')
```

Place this at the top of `halp_fl.py`, and comment out any `plt.show()` calls. Figures are saved automatically as PNG files in `figures/`.

---

## Security and scope

The framework provides three distinct guarantees. They should not be conflated:

1. **Differential privacy for $\mathcal{W}_{\mathrm{DP}}$**: the DP-protected parameter subset satisfies $(\varepsilon, \delta)$-DP with $\varepsilon$ the cumulative budget over $T$ rounds. This guarantee is what the reported accuracy claims refer to.
2. **Cryptographic confidentiality for $\mathcal{W}_{\mathrm{HE}}$**: below the threshold $t$, the ciphertext is computationally hidden. At or above the threshold, the HE-protected aggregate is revealed exactly and carries no DP noise.
3. **Sub-threshold collusion resistance**: the framework resists up to $t - 1$ colluding clients. The residual-privacy law when the threshold is broken is derived in the code and reported in the output tables.

The following attacks are out of scope and are not evaluated: norm-constrained adaptive poisoning that deliberately stays below the detection threshold; backdoor and model-replacement attacks; gradient inversion with white-box knowledge of the server's aggregation; and clients misreporting their sensitivity score $s_i$ or non-IID degree $h_i$ to obtain a larger budget.

---

## License

This code is released under the MIT License. See `LICENSE` for details.

The threshold Paillier implementation is built on top of `python-paillier` (MIT License). The Bulletproof verification uses a Rust implementation with Python bindings; see `requirements.txt` for details.

---

## Reproducibility Checklist

-  Complete source code provided in `halp_fl.py`
-  All dependencies explicitly listed in `requirements.txt`
-  Automatic dataset download for MNIST, FashionMNIST, CIFAR-10
-  Fixed random seed (`torch.manual_seed(42)`, `numpy.random.seed(42)`) set at the start of `halp_fl.py` for bit-exact reproducibility
-  Preview mode (`--preview`) for fast pipeline verification without full training
-  Full mode (`--full`) reproduces the reported numbers at the reference hyperparameters
-  JSON result files include the full hyperparameter set for auditing
-  Figures regenerated from raw result files, not hard-coded

---

**Last Updated:** September 25, 2026
