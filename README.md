# Upstart AI Lending Fragility — Dissertation Code Repository

Companion code repository for the MSIN0032 dissertation **_The Fragility of Alternative Credit: An Analysis of AI-Driven Lending Performance Under Macroeconomic Stress (2023-2026)_**.

Six Python scripts implement the Part I pilot study, the three Part II solutions, the §3.3 vintage-benchmark robustness check, and the §5.5 Monte Carlo simulation. All scripts are deterministic (random seed 42) and reproduce the body's numerical outputs to within floating-point precision.

---

## Quick reference: file-to-body mapping

| File | Implements | Supports |
|---|---|---|
| `pilot_study.py` | Part I baseline pilot study (FICO vs ML, normal vs stress) | §1.1 235% calibration rise; Part I §3-§4 throughout |
| `solution_1.py` | Solution 1 adversarial-validation gate | §2 throughout (+14.41pp severely-adverse signal, 4.26pp threshold) |
| `solution_2.py` | Solution 2 PSI-triggered retraining demonstration | §3 throughout (one-quarter lead time, 23.1% reduction) |
| `solution_3.py` | Solution 3 vanilla split conformal + ACI | §4 throughout (25.48% abstention, ACI α 0.10 → 0.1094) |
| `vintage_benchmark.py` | Maturity-correction sensitivity analysis | §3.3 robustness check; Appendix B Table B.1 |
| `monte_carlo.py` | §5.5 layered-defence NPV simulation | §5.5 ($133M median NPV, 45.1% volatility reduction) |

---

## Setup

### Prerequisites

- Python 3.10 or higher
- Approximately 2GB free disk space (for dataset + outputs)
- Standard Python scientific stack (installed via `requirements.txt`)

### Installation

```bash
git clone https://github.com/uztaram/upstart-ai-lending-fragility.git
cd upstart-ai-lending-fragility
pip install -r requirements.txt
```

### Dataset

The pilot study uses LendingClub historical loan data (2007 through Q3 2020). The dataset is too large for direct GitHub hosting (approximately 1.1GB) and must be downloaded separately.

1. Visit [Kaggle: ethon0426/lending-club-20072020q1](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)
2. Download the dataset (requires free Kaggle account)
3. Place the CSV file in the repository root

See `DATA_README.md` for detailed instructions.

### Reproducing body outputs

Once the dataset is in place, run each script in this order from the repository root:

```bash
# Part I pilot study (foundational; produces baseline + stress results)
python pilot_study.py

# Vintage benchmark sensitivity (§3.3 robustness check)
python vintage_benchmark.py

# Solution 1 adversarial validation (§2)
python solution_1.py

# Solution 2 PSI monitoring (§3)
python solution_2.py

# Solution 3 conformal prediction with ACI (§4)
python solution_3.py

# Section 5 Monte Carlo (§5.5)
python monte_carlo.py
```

Each script writes its outputs to its own subfolder (e.g. `pilot_study_outputs/`, `solution_1_outputs/`). CSV filenames match the body's table labels and PNG filenames match the body's figure labels.

---

## Function-level documentation

### `pilot_study.py` — Part I baseline pilot study

| Function | Lines | Purpose | Supports body claim |
|---|---|---|---|
| `calculate_psi(expected, actual, bins=10)` | 215-230 | Population Stability Index between two distributions | §3.1 PSI definition; §3.2 quarterly PSI trajectory |
| `calculate_calibration_error(y_true, y_prob, n_bins=10)` | 233-244 | Expected Calibration Error (ECE) using ten equal-frequency bins | §1.1 235% ECE rise; §2.1 4.26pp threshold; §3.4 7.81pp/6.00pp; §4.4 6.18pp/5.28pp |
| `preprocess_data(df)` | 304-508 | 18-month observation-window filter; feature engineering | All sample sizes (n = 443,065 in §2.1; n = 653,380 in §3.1) |
| `split_data(df)` | 513-534 | Train/calibration/test split aligned with 2012-2016 / 2016 / 2017 / 2018Q1-2019Q2 windows | §3.1 empirical setup |
| `prepare_features(df_train, df_test, feature_list)` | 539-572 | Imputation, standardisation, feature alignment | All model training procedures |

### `solution_1.py` — Solution 1 adversarial validation gate

| Function | Lines | Purpose | Supports body claim |
|---|---|---|---|
| `apply_scenario(df_baseline, scenario_spec)` | 171-220 | CCAR severity transformations (baseline / adverse / severely adverse) | §2.1 scenario construction |
| `build_ground_truth_lookup(df_stress_period, n_fico_bins, n_dti_bins)` | 222-260 | FICO-by-DTI lookup grid for proxy actual rates | §2.1 proxy truth; Appendix A.1 four-step calculation |
| `assign_proxy_truth(df_synthetic, lookup, fico_edges, dti_edges)` | 262-285 | Maps synthetic borrowers to proxy actual rates | §2.1 counterfactual setup |
| `score_scenario(model, scaler, imputer, feature_list, df_scenario)` | 287-298 | Scores candidate model on transformed borrowers | §2.1 model scoring |
| `calibration_error_against_proxy(y_prob, proxy_rates, n_bins=10)` | 300-352 | Calibration error between predicted and proxy actual | §2.1 deterioration metric |
| `construct_adversarial_augmented_training_set(...)` | 426-547 | Adversarially augmented training data for fixing the gate breach | §2.3 augmented model construction |
| `run_adversarial_validation(...)` | 548-594 | Computes deterioration signal at the 4.26pp threshold | §2.1 +14.41pp severely-adverse; §2.3 augmented passes |
| `evaluate_benign_performance(...)` | 601-639 | Benign 2017 counter-test (AUC + rate-error preservation) | §2.3 Figure 2.3; AUC 0.6735 → 0.6672; rate error +1.98pp → +0.97pp |
| `plot_deterioration_comparison(results_a, results_b, filename)` | 641-713 | Generates Figure 2.1 | §2.2 Figure 2.1 |
| `plot_predicted_vs_actual_comparison(results_a, results_b, filename)` | 715-781 | Generates Figure 2.2 | §2.2 Figure 2.2 |
| `plot_benign_performance(benign_a, benign_b, filename)` | 783-832 | Generates Figure 2.3 | §2.3 Figure 2.3 |

### `solution_2.py` — Solution 2 PSI-triggered retraining

| Function | Purpose | Supports body claim |
|---|---|---|
| `get_quarterly_subsets(df, start_year, end_year, end_quarter)` | Quarterly partition of monitoring window | §3.1 six-quarter monitoring window |
| `build_monitoring_set(df_full)` | Constructs n = 653,380 monitoring set after 18-month filter | §3.1 sample size |
| `prepare_features_for_training(df_train, feature_list, target)` | Feature alignment with 2012-2016 reference | §3 candidate model training |
| `train_neural_net(X_train, y_train, label)` | MLP training with fixed seeds (Model A and Model B) | §3.4 Models A and B construction |
| `compute_output_psi_and_calibration(...)` | Per-quarter calibration error trajectory | §3.4 Table 3.3; Figure 3.3 quarterly trajectory |
| `evaluate_benign_period(...)` | 2017 benign counter-test for Model B | §3.4 retraining doesn't damage benign performance |

### `solution_3.py` — Solution 3 conformal prediction wrapper

Implements vanilla split conformal, Adaptive Conformal Inference (Gibbs and Candès, 2021), and the layered-defence configuration combining ACI with the Solution 2 PSI-retrained model.

| Function | Lines | Purpose | Supports body claim |
|---|---|---|---|
| `compute_nonconformity_scores(y_true, y_prob)` | 206-210 | Nonconformity score s = \|y − p\| | §4.1 nonconformity score; Appendix C.2 step 1 |
| `conformal_threshold(calibration_scores, alpha)` | 213-218 | τ as ⌈(n+1)(1-α)⌉/n empirical quantile | §4.1 threshold formula; §4.2 vanilla τ derivation |
| `conformal_prediction_sets(y_prob, tau)` | 221-235 | Returns prediction set; abstains if both labels included | §4.2 abstention rate 25.48%; §4.4 layered 20.38% |
| `evaluate_conformal_quarter(y_prob, y_true, tau)` | 353-380 | Per-quarter coverage, abstention, calibration metrics | §4.2 vanilla per-quarter results; §4.4 layered per-quarter |
| `run_aci_through_quarters(...)` | 429-485 | Walks quarters applying ACI online α-update | §4.3 α-trajectory 0.10 → 0.1094; §4.4 layered ACI |
| `run_vanilla_through_quarters(...)` | 488-530 | Walks quarters with fixed α (vanilla baseline) | §4.2 vanilla baseline |
| `build_calibration_scores(...)` | 532-554 | Constructs calibration sample from 2016 held-out data | §4.1 calibration sample setup |

### `vintage_benchmark.py` — Maturity-correction sensitivity

| Function | Purpose | Supports body claim |
|---|---|---|
| `quarter_specific_correction_factor(quarter_label, base_factor=1.45)` | Per-quarter maturity factor (1.30× to 1.45×) | §3.3 robustness check; Appendix B Table B.1 |
| `apply_correction(observed_rate, factor)` | Applies maturity correction to observed actual rate | Appendix B.4 corrected actuals |
| Script-level analysis | Computes corrected calibration errors per quarter | Appendix B Table B.1 (4.01pp Model A, 3.14pp Model B aggregates) |

### `monte_carlo.py` — Section 5 layered-defence NPV simulation

5,000 iterations, fixed random seed, triangular parameter distributions per Appendix D.2.

| Function | Purpose | Supports body claim |
|---|---|---|
| `run_iteration()` | Single iteration: samples nine parameters, computes combined volatility reduction, annual cash flows, 5-year NPV at 12% discount | §5.5 Table 5.2 (P10 $82M, median $133M, P90 $198M); Appendix D.3 worked example |
| Script-level aggregation | 5,000 iterations into headline percentiles and probabilities | §5.5 P(NPV > 0) = 100%; P(meeting 50% target) = 10.9%; mean reduction 45.1% |

---

## Reproducibility

All scripts are deterministic. Random seed 42 is set at the start of each entry point. Re-running any script on a clean checkout produces bit-identical numerical outputs to those reported in the body, modulo floating-point operations within documented precision (typically within 1e-10 absolute tolerance).

Output CSVs corresponding to body tables and PNG files corresponding to body figures are produced as side-effects of the relevant scripts and are committed to the repository under per-script subfolders.

### Pre-computed outputs

Each script's output subfolder contains all CSVs and figures from a clean run. Markers can verify reproducibility either by:

1. **Direct comparison**: inspect each `*_outputs/` folder and compare to the body
2. **Full reproduction**: download the dataset, run the scripts, compare to committed outputs

---

## Dependencies

Python 3.10+. Key libraries: scikit-learn, numpy, pandas, matplotlib, scipy. Full list with version pins in `requirements.txt`.

---

## File structure

```
upstart-ai-lending-fragility/
├── README.md                     ← This file
├── DATA_README.md                ← Dataset download instructions
├── requirements.txt              ← Python dependencies
├── LICENSE                       ← MIT licence
│
├── pilot_study.py                ← Part I baseline pilot study
├── solution_1.py                 ← Solution 1 adversarial validation
├── solution_2.py                 ← Solution 2 PSI monitoring
├── solution_3.py                 ← Solution 3 conformal prediction
├── vintage_benchmark.py          ← §3.3 robustness check
├── monte_carlo.py                ← §5.5 NPV simulation
│
├── pilot_study_outputs/          ← Pre-computed pilot study results
├── solution_1_outputs/           ← Pre-computed Solution 1 results
├── solution_2_outputs/           ← Pre-computed Solution 2 results
├── solution_3_outputs/           ← Pre-computed Solution 3 results
├── vintage_benchmark_outputs/    ← Pre-computed robustness check
└── monte_carlo_results.csv       ← Pre-computed NPV simulation
```

---

## Citation

If you reference this code or its analytical outputs, please cite the parent dissertation:

> Marat Daminov (2026) _The Fragility of Alternative Credit: An Analysis of AI-Driven Lending Performance Under Macroeconomic Stress (2023-2026)_. MSIN0032 Management Science Dissertation, University College London.

---

## Licence

MIT — see `LICENSE` file. Code may be reused with attribution.

---

## Contact

For questions about reproducibility or code interpretation, contact the dissertation author through the UCL School of Management.
