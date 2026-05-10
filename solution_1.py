#!/usr/bin/env python3
"""
MSIN0032 Dissertation - Part II Empirical Test
==============================================
Solution 1: Adversarial Validation with Synthetic Stress Scenarios

This script implements the empirical test of Solution 1 from Part II of the
dissertation. It evaluates whether adversarial validation can be used both
diagnostically (as a release-pipeline gate that flags brittle models before
deployment) and prescriptively (as a training-data construction principle
that reduces stress-induced calibration deterioration).

METHODOLOGY OVERVIEW

    Two ML models are trained and compared. Both share architecture, hyper-
    parameters, and random seeds; only the training data differs.
    
        Model A (control):  trained on the original 2012-2016 LendingClub
                            sample, identical to the Part I pilot study.
        
        Model B (augmented): trained on the same 2012-2016 sample plus a
                             25% synthetic-stress augmentation drawn from
                             CCAR Adverse and Severely Adverse scenarios,
                             with default labels assigned via a FICO x DTI
                             proxy lookup built from real 2018-2020
                             stress-period borrowers.
    
    Both models are scored against three CCAR scenarios (Baseline, Adverse,
    Severely Adverse) applied to the 2017 benign-period borrower population.
    The metric of interest is calibration deterioration (delta calibration
    error from the baseline scenario to each stressed scenario), evaluated
    against the 4.26pp failure-onset threshold derived in Section 2.
    
    A counter-test on the un-transformed 2017 population establishes that
    adversarial augmentation does not degrade benign-regime performance.

INTELLECTUAL CONTRIBUTION

    Adversarial validation appears in the model-risk literature primarily
    as a detection mechanism: a gate that approves or rejects candidate
    models. The empirical test here demonstrates a stronger claim, that
    the same synthetic scenarios used for the validation gate can be fed
    back into training-data construction. This is the credit-risk analogue
    of adversarial training in computer vision (Goodfellow et al. 2015;
    Madry et al. 2018).

DESIGN CHOICES

    25% augmentation rate: balance between injecting enough stress signal
    that the model learns to handle it, and not flooding the training set
    with synthetic data that swamps the real distribution.
    
    Half-adverse / half-severely-adverse split within the augmented sample:
    matches the two-scenario structure of the Section 2 validation gate so
    the model is exposed to examples across the full severity range.
    
    Same architecture (MLPClassifier, hidden layers 100/50/25, ReLU, Adam)
    as the Part I pilot study: keeps the comparison clean. The only thing
    that differs between Model A and Model B is the training data, so any
    change in behaviour is attributable to the augmentation itself.
    
    Proxy default labels for synthetic borrowers: the methodologically
    delicate step. We cannot observe true outcomes for synthetic borrowers
    because they do not exist. The label assignment uses a FICO x DTI
    lookup constructed from real 2018-2020 stress-period borrowers, so
    each synthetic borrower is labelled according to how real borrowers
    with the same stressed profile actually performed under stress.

OUTPUTS

    A subdirectory `solution_1_outputs/` is created next to this script.
    Figures: deterioration_comparison.png, predicted_vs_actual.png,
    benign_performance.png. Tables: results_per_scenario.csv,
    summary_comparison.csv.

USAGE

    python solution_1.py
    
    Requires pilot_study.py in the same directory. The LendingClub CSV
    path is configured inside pilot_study.py.

Author: Marat Daminov
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ============================================================================
# PATH SETUP
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from pilot_study import (
        TARGET, EARLY_DQ_TARGET, RANDOM_STATE, COLORS,
        FICO_FEATURES, ML_FEATURES_CORRECTED,
        find_data_file, load_data, preprocess_data, split_data,
    )
except ImportError as e:
    print(f"[ERROR] Could not import from pilot_study: {e}")
    print("        Make sure pilot_study.py is in the same directory "
          "as this script.")
    sys.exit(1)

warnings.filterwarnings('ignore')


# ============================================================================
# ADVERSARIAL VALIDATION COMPONENTS
# ----------------------------------------------------------------------------
# These were previously imported from a separate v2 file. They are inlined
# here so that solution_1.py is fully self-contained and depends only on
# pilot_study.py.
# ============================================================================

# Failure threshold from Section 2 of Part II:
#   Normal-period baseline ECE = 2.13pp (mean of 2017Q1-Q4)
#   Failure onset = 2x baseline = 4.26pp
# Used as the threshold for deterioration-vs-baseline.
CALIBRATION_FAILURE_THRESHOLD_PP = 4.26

# Years from which the proxy ground-truth lookup is constructed.
GROUND_TRUTH_START_YEAR = 2018
GROUND_TRUTH_END_YEAR = 2020

# Three CCAR-style adversarial scenarios. Severities calibrated against
# Bhutta and Keys (2016) recession-period feature shifts and the Federal
# Reserve CCAR scenario specifications (Federal Reserve, 2025).
SCENARIOS = {
    'baseline': {
        'description': 'No shock (calibration sanity check)',
        'fico_delta':       0,
        'income_pct_delta': 0.0,
        'dti_delta':        0,
        'util_delta':       0,
        'inq_delta':        0,
    },
    'adverse': {
        'description': 'Moderate recession (CCAR Adverse)',
        'fico_delta':       -20,
        'income_pct_delta': -0.05,
        'dti_delta':        +5,
        'util_delta':       +10,
        'inq_delta':        +1,
    },
    'severely_adverse': {
        'description': 'Major shock (CCAR Severely Adverse / 2008 analog)',
        'fico_delta':       -40,
        'income_pct_delta': -0.15,
        'dti_delta':        +10,
        'util_delta':       +20,
        'inq_delta':        +2,
    },
}


def apply_scenario(df_baseline, scenario_spec):
    """
    Apply CCAR-style feature transformations to a baseline borrower
    population. Returns a new DataFrame with transformed features but the
    same loan IDs and labels. Transformations reflect documented effects
    of macroeconomic stress on consumer credit profiles
    (Bhutta and Keys, 2016; Federal Reserve, 2025).
    """
    df_stressed = df_baseline.copy()

    df_stressed['fico_range_high'] = (
        df_stressed['fico_range_high'] + scenario_spec['fico_delta']
    ).clip(lower=300, upper=850)

    df_stressed['annual_inc'] = (
        df_stressed['annual_inc'] * (1 + scenario_spec['income_pct_delta'])
    ).clip(lower=0)

    df_stressed['dti'] = (
        df_stressed['dti'] + scenario_spec['dti_delta']
    ).clip(lower=0, upper=80)

    df_stressed['revol_util'] = (
        df_stressed['revol_util'] + scenario_spec['util_delta']
    ).clip(lower=0, upper=150)

    df_stressed['inq_last_6mths'] = (
        df_stressed['inq_last_6mths'] + scenario_spec['inq_delta']
    ).clip(lower=0, upper=20)

    # Re-derive engineered features that depend on transformed inputs
    df_stressed['log_income'] = np.log1p(df_stressed['annual_inc'].clip(lower=0))
    df_stressed['fico_squared'] = df_stressed['fico_range_high'] ** 2
    df_stressed['dti_fico_interaction'] = (
        df_stressed['dti'].fillna(0) * df_stressed['fico_range_high'].fillna(0)
    )
    monthly_inc = df_stressed['annual_inc'].fillna(0) / 12.0
    df_stressed['payment_to_income'] = (
        df_stressed['installment'] / monthly_inc.clip(lower=1)
    ).clip(upper=2.0)
    df_stressed['utilization_squared'] = (
        df_stressed['revol_util'].fillna(0) / 100
    ) ** 2
    df_stressed['debt_burden'] = (
        df_stressed['revol_bal'].fillna(0) /
        df_stressed['annual_inc'].fillna(1).clip(lower=1)
    ).clip(upper=5.0)

    return df_stressed


def build_ground_truth_lookup(df_stress_period, n_fico_bins=10, n_dti_bins=5):
    """
    Build a (FICO bin, DTI bin) -> realised default rate lookup from
    2018-2020 borrowers. Used as proxy ground truth for the calibration
    measurement. The lookup carries a downward bias from loan immaturity
    that is approximately constant across scenarios, which is why the
    primary metric is deterioration-vs-baseline (the bias appears in both
    terms and cancels).
    """
    df = df_stress_period[
        ['fico_range_high', 'dti', EARLY_DQ_TARGET]
    ].dropna()

    fico_edges = np.unique(np.percentile(
        df['fico_range_high'], np.linspace(0, 100, n_fico_bins + 1)
    ))
    dti_edges = np.unique(np.percentile(
        df['dti'], np.linspace(0, 100, n_dti_bins + 1)
    ))

    df['fico_bin'] = pd.cut(df['fico_range_high'], bins=fico_edges,
                             labels=False, include_lowest=True)
    df['dti_bin'] = pd.cut(df['dti'], bins=dti_edges,
                            labels=False, include_lowest=True)

    lookup = (
        df.groupby(['fico_bin', 'dti_bin'])[EARLY_DQ_TARGET]
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'realised_default_rate', 'count': 'n_loans'})
    )

    print(f"\n  Proxy ground truth built from {len(df):,} real "
          f"{GROUND_TRUTH_START_YEAR}-{GROUND_TRUTH_END_YEAR} borrowers")
    print(f"  Lookup grid: {len(lookup):,} (FICO x DTI) cells, "
          f"avg {lookup['n_loans'].mean():.0f} loans per cell")

    return lookup, fico_edges, dti_edges


def assign_proxy_truth(df_synthetic, lookup, fico_edges, dti_edges):
    """Assign proxy actual default rate to each synthetic borrower."""
    df = df_synthetic.copy()

    df['fico_bin'] = pd.cut(df['fico_range_high'], bins=fico_edges,
                             labels=False, include_lowest=True)
    df['dti_bin'] = pd.cut(df['dti'], bins=dti_edges,
                            labels=False, include_lowest=True)

    overall_rate = (
        (lookup['realised_default_rate'] * lookup['n_loans']).sum() /
        lookup['n_loans'].sum()
    )

    df = df.merge(
        lookup[['fico_bin', 'dti_bin', 'realised_default_rate']],
        on=['fico_bin', 'dti_bin'],
        how='left',
    )
    df['realised_default_rate'] = df['realised_default_rate'].fillna(overall_rate)
    df.rename(columns={'realised_default_rate': 'proxy_actual_rate'}, inplace=True)

    return df, overall_rate


def score_scenario(model, scaler, imputer, feature_list, df_scenario):
    """Score a synthetic scenario through the trained model."""
    sub = df_scenario[feature_list].copy()
    for f in feature_list:
        if f not in sub.columns:
            sub[f] = 0
    imputed = pd.DataFrame(
        imputer.transform(sub), columns=feature_list, index=sub.index
    )
    scaled = scaler.transform(imputed)
    return model.predict_proba(scaled)[:, 1]


def calibration_error_against_proxy(y_prob, proxy_rates, n_bins=10):
    """
    Weighted mean absolute deviation between predicted and proxy actual
    default rate, binned by predicted probability. Returns percentage
    points.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bin_edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    total_err = 0.0
    total_weight = 0
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        bin_pred_mean = y_prob[mask].mean()
        bin_actual_mean = proxy_rates[mask].mean()
        weight = mask.sum()
        total_err += weight * abs(bin_pred_mean - bin_actual_mean)
        total_weight += weight

    if total_weight == 0:
        return np.nan
    return (total_err / total_weight) * 100

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "solution_1_outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Adversarial augmentation hyperparameters
AUGMENTATION_FRACTION = 0.25   # 25% of training set sampled for augmentation
ADVERSE_SHARE = 0.50           # of augmented sample, half get adverse,
                                # half severely_adverse

plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 10,
})


# ============================================================================
# FEATURE PREP (replicates v3 exactly, but we need to expose imputer/scaler
# fitted only on Model A's training data, then re-fit for Model B)
# ============================================================================

def prepare_features_for_training(df_train, feature_list, target):
    """
    Fit imputer + scaler on a training set, return scaled X_train, y_train,
    plus the imputer + scaler so we can apply them at scoring time.

    This is the same pipeline as v3's prepare_features, but we don't need
    the test split here because we score scenarios separately.
    """
    available = [f for f in feature_list if f in df_train.columns]
    missing = [f for f in feature_list if f not in df_train.columns]
    if missing:
        for f in missing:
            df_train[f] = 0
        available = feature_list

    X_train = df_train[available].copy()
    y_train = df_train[target].values

    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=available, index=X_train.index
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)

    return X_train_scaled, y_train, available, scaler, imputer


# ============================================================================
# MODEL TRAINING (locked-in MLPClassifier matching v3 exactly)
# ============================================================================

def train_neural_net(X_train, y_train, label="MODEL"):
    """
    Train MLPClassifier with the same hyperparameters as v3's
    train_ml_model_neural_net. Identical architecture across Model A
    (clean training) and Model B (adversarial-augmented training) so
    that any difference in behaviour is attributable to training data
    only, not architecture.
    """
    print(f"\n  Training {label} (MLPClassifier)...")
    print(f"    Architecture: 3 layers (100, 50, 25 neurons)")
    print(f"    Activation: ReLU; Optimizer: Adam")
    print(f"    Training set size: {len(y_train):,} examples")
    print(f"    Default rate in training: {y_train.mean():.1%}")

    model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X_train, y_train)
    print(f"    Converged in {model.n_iter_} iterations")
    print(f"    Final training loss: {model.loss_:.4f}")
    return model


# ============================================================================
# ADVERSARIAL AUGMENTATION (the fix)
# ============================================================================

def construct_adversarial_augmented_training_set(
    df_clean_train, truth_lookup, fico_edges, dti_edges,
    augmentation_fraction=AUGMENTATION_FRACTION,
    adverse_share=ADVERSE_SHARE,
):
    """
    Build the training set for Model B (the adversarially hardened model).

    Steps:
      1. Sample a fraction of the clean training borrowers (default 25%).
      2. Split this sample 50/50: half get the adverse transformation,
         half get severely_adverse.
      3. Apply CCAR-style transformations via apply_scenario().
      4. Re-assign default labels for the transformed borrowers using the
         FICO x DTI proxy lookup constructed from real 2018-2020 stressed
         borrowers (same lookup v2 uses for proxy ground truth at
         validation time).
      5. Convert the proxy lookup rates into binary labels via Bernoulli
         draws (so the augmented set has the same {0, 1} label structure
         as the original training data).
      6. Concatenate clean + augmented sets.

    Returns the combined DataFrame ready for prepare_features_for_training.

    METHODOLOGICAL NOTE:
      The proxy lookup gives REALISED default rates for FICO/DTI cells in
      2018-2020. Step 5 converts these rates back into {0, 1} labels for
      individual synthetic borrowers via Bernoulli sampling. This adds
      controlled stochasticity but preserves the expected default rate per
      cell. Without this step we would be giving the model fractional
      labels (e.g., 0.34) which the binary classifier cannot consume.
    """
    print(f"\n  Constructing adversarial-augmented training set...")
    print(f"    Augmentation fraction: {augmentation_fraction:.0%}")
    print(f"    Within augmented sample: {adverse_share:.0%} adverse, "
          f"{1-adverse_share:.0%} severely_adverse")

    rng = np.random.RandomState(RANDOM_STATE)

    # Step 1: sample borrowers to augment
    n_to_augment = int(len(df_clean_train) * augmentation_fraction)
    sample_idx = rng.choice(
        df_clean_train.index, size=n_to_augment, replace=False
    )
    df_to_augment = df_clean_train.loc[sample_idx].copy()

    # Step 2: split sample into adverse vs severely_adverse halves
    n_adverse = int(n_to_augment * adverse_share)
    adverse_idx = df_to_augment.index[:n_adverse]
    severely_idx = df_to_augment.index[n_adverse:]

    df_adverse_pop = df_to_augment.loc[adverse_idx].copy()
    df_severely_pop = df_to_augment.loc[severely_idx].copy()

    # Step 3: apply CCAR transformations
    df_adverse_aug = apply_scenario(df_adverse_pop, SCENARIOS['adverse'])
    df_severely_aug = apply_scenario(df_severely_pop, SCENARIOS['severely_adverse'])

    print(f"    Generated {len(df_adverse_aug):,} adverse-stressed examples")
    print(f"    Generated {len(df_severely_aug):,} severely-adverse-stressed examples")

    # Step 4: assign proxy default rates
    df_adverse_with_truth, _ = assign_proxy_truth(
        df_adverse_aug, truth_lookup, fico_edges, dti_edges
    )
    df_severely_with_truth, _ = assign_proxy_truth(
        df_severely_aug, truth_lookup, fico_edges, dti_edges
    )

    # Step 5: convert proxy rates into {0, 1} labels via Bernoulli draws
    # Using a different RNG seed for each subset to keep noise independent
    rng2 = np.random.RandomState(RANDOM_STATE + 1)
    rng3 = np.random.RandomState(RANDOM_STATE + 2)

    p_adverse = df_adverse_with_truth['proxy_actual_rate'].values
    p_severely = df_severely_with_truth['proxy_actual_rate'].values

    df_adverse_with_truth[TARGET] = (
        rng2.uniform(size=len(p_adverse)) < p_adverse
    ).astype(int)
    df_severely_with_truth[TARGET] = (
        rng3.uniform(size=len(p_severely)) < p_severely
    ).astype(int)

    # Mirror the same draw onto the EARLY_DQ_TARGET so any downstream code
    # that reads it sees a consistent label. (Conservative choice.)
    df_adverse_with_truth[EARLY_DQ_TARGET] = df_adverse_with_truth[TARGET]
    df_severely_with_truth[EARLY_DQ_TARGET] = df_severely_with_truth[TARGET]

    print(f"    Adverse augmented default rate:           "
          f"{df_adverse_with_truth[TARGET].mean():.1%}")
    print(f"    Severely-adverse augmented default rate:  "
          f"{df_severely_with_truth[TARGET].mean():.1%}")

    # Step 6: combine clean + augmented training data
    # Drop helper columns added by assign_proxy_truth
    drop_cols = ['proxy_actual_rate', 'fico_bin', 'dti_bin']
    for d in drop_cols:
        if d in df_adverse_with_truth.columns:
            df_adverse_with_truth = df_adverse_with_truth.drop(columns=[d])
        if d in df_severely_with_truth.columns:
            df_severely_with_truth = df_severely_with_truth.drop(columns=[d])

    df_augmented_full = pd.concat(
        [df_clean_train, df_adverse_with_truth, df_severely_with_truth],
        axis=0, ignore_index=True
    )

    print(f"    Clean training set:      {len(df_clean_train):,} loans")
    print(f"    Adversarial additions:   "
          f"{len(df_adverse_with_truth) + len(df_severely_with_truth):,} loans")
    print(f"    Combined training set:   {len(df_augmented_full):,} loans "
          f"(+{augmentation_fraction:.0%} growth)")
    print(f"    Combined default rate:   {df_augmented_full[TARGET].mean():.1%}")

    return df_augmented_full


# ============================================================================
# VALIDATION GATE (reuses v2 scenario logic - same gate the original model failed)
# ============================================================================

def run_adversarial_validation(
    model, scaler, imputer, feature_list, df_normal,
    truth_lookup, fico_edges, dti_edges, label="MODEL"
):
    """
    Run the v2-identical adversarial validation gate against a trained
    model, returning per-scenario calibration error and deterioration.
    """
    print(f"\n  Running adversarial validation on {label}...")

    rows = []
    for scenario_name, scenario_spec in SCENARIOS.items():
        df_scenario = apply_scenario(df_normal, scenario_spec)
        df_scenario_with_truth, _ = assign_proxy_truth(
            df_scenario, truth_lookup, fico_edges, dti_edges
        )

        y_prob = score_scenario(
            model, scaler, imputer, feature_list, df_scenario_with_truth
        )
        proxy_rates = df_scenario_with_truth['proxy_actual_rate'].values

        cal_err_pp = calibration_error_against_proxy(y_prob, proxy_rates)
        mean_pred = float(np.mean(y_prob))
        mean_actual = float(np.mean(proxy_rates))

        rows.append({
            'model': label,
            'scenario': scenario_name,
            'mean_predicted_default_rate': round(mean_pred, 4),
            'mean_proxy_actual_rate': round(mean_actual, 4),
            'rate_error_pp': round((mean_pred - mean_actual) * 100, 2),
            'calibration_error_pp': round(cal_err_pp, 2),
            'n_borrowers': len(df_scenario_with_truth),
        })

    df = pd.DataFrame(rows)
    baseline_err = float(
        df.loc[df['scenario'] == 'baseline', 'calibration_error_pp'].iloc[0]
    )
    df['deterioration_vs_baseline_pp'] = (
        df['calibration_error_pp'] - baseline_err
    ).round(2)
    df['flags_failure'] = (
        df['deterioration_vs_baseline_pp'] >= CALIBRATION_FAILURE_THRESHOLD_PP
    )
    return df, baseline_err


# ============================================================================
# COUNTER-TEST: BENIGN-REGIME PERFORMANCE
# ============================================================================

def evaluate_benign_performance(
    model, scaler, imputer, feature_list, df_normal, label="MODEL"
):
    """
    Score the model on UNTRANSFORMED 2017 borrowers (no scenario applied).
    Reports AUC against actual labels and the model's mean predicted
    default rate vs the actual default rate.

    This is the counter-test: if adversarial augmentation hurts benign
    performance, it shows up here. We need MODEL B to be approximately
    as good as MODEL A on this metric.
    """
    df = df_normal.copy()
    y_true = df[EARLY_DQ_TARGET].values
    y_prob = score_scenario(model, scaler, imputer, feature_list, df)

    auc = roc_auc_score(y_true, y_prob)
    mean_pred = float(np.mean(y_prob))
    mean_actual = float(np.mean(y_true))
    rate_err_pp = (mean_pred - mean_actual) * 100

    print(f"\n  {label} benign performance on 2017:")
    print(f"    AUC:                 {auc:.4f}")
    print(f"    Mean predicted DR:   {mean_pred:.2%}")
    print(f"    Actual DR:           {mean_actual:.2%}")
    print(f"    Rate error:          {rate_err_pp:+.2f}pp")

    return {
        'model': label,
        'auc': round(auc, 4),
        'mean_predicted_default_rate': round(mean_pred, 4),
        'mean_actual_default_rate': round(mean_actual, 4),
        'rate_error_pp': round(rate_err_pp, 2),
    }


# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_deterioration_comparison(results_a, results_b, filename):
    """
    HEADLINE FIGURE: side-by-side comparison of stress-induced calibration
    deterioration for MODEL A (clean training) vs MODEL B (adversarial
    augmented training), with the failure threshold marked.
    """
    scenarios_order = ['baseline', 'adverse', 'severely_adverse']
    scenarios_label = ['Baseline', 'Adverse', 'Severely Adverse']

    a_deter = []
    b_deter = []
    for s in scenarios_order:
        a_deter.append(float(
            results_a.loc[results_a['scenario'] == s,
                          'deterioration_vs_baseline_pp'].iloc[0]
        ))
        b_deter.append(float(
            results_b.loc[results_b['scenario'] == s,
                          'deterioration_vs_baseline_pp'].iloc[0]
        ))

    x = np.arange(len(scenarios_order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 7))
    bars_a = ax.bar(
        x - width/2, a_deter, width,
        label='Model A: Clean training (control)',
        color='#C44E52', edgecolor='white', linewidth=1.2
    )
    bars_b = ax.bar(
        x + width/2, b_deter, width,
        label='Model B: Adversarial-augmented training (fix)',
        color='#4C72B0', edgecolor='white', linewidth=1.2
    )

    ax.axhline(
        CALIBRATION_FAILURE_THRESHOLD_PP, color='red', linestyle='--',
        linewidth=1.5, alpha=0.7,
        label=f'Failure-onset threshold ({CALIBRATION_FAILURE_THRESHOLD_PP:.1f}pp)'
    )
    ax.axhline(0, color='black', linewidth=0.7, alpha=0.5)

    for bar, val in zip(bars_a, a_deter):
        h = bar.get_height()
        offset = 0.3 if h >= 0 else -0.3
        va = 'bottom' if h >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                f'{val:+.1f}', ha='center', va=va, fontsize=9, fontweight='bold')
    for bar, val in zip(bars_b, b_deter):
        h = bar.get_height()
        offset = 0.3 if h >= 0 else -0.3
        va = 'bottom' if h >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                f'{val:+.1f}', ha='center', va=va, fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_label)
    ax.set_ylabel('Stress-Induced Calibration Deterioration (pp)')
    ax.set_title(
        'Stress-induced calibration deterioration across CCAR scenarios\n'
        'Model B (adversarially trained) vs Model A (control)',
        fontsize=12
    )
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    out = os.path.join(FIGURES_DIR, filename)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def plot_predicted_vs_actual_comparison(results_a, results_b, filename):
    """
    Two-panel reliability plot: for each scenario, show predicted DR vs
    proxy actual DR for both models. The control (Model A) should show
    growing divergence under stress; the fix (Model B) should show
    smaller divergence.
    """
    scenarios_order = ['baseline', 'adverse', 'severely_adverse']
    scenarios_label = ['Baseline', 'Adverse', 'Severely\nAdverse']

    a_pred = [float(results_a.loc[results_a['scenario'] == s,
                                   'mean_predicted_default_rate'].iloc[0]) * 100
              for s in scenarios_order]
    a_actual = [float(results_a.loc[results_a['scenario'] == s,
                                     'mean_proxy_actual_rate'].iloc[0]) * 100
                for s in scenarios_order]
    b_pred = [float(results_b.loc[results_b['scenario'] == s,
                                   'mean_predicted_default_rate'].iloc[0]) * 100
              for s in scenarios_order]
    b_actual = [float(results_b.loc[results_b['scenario'] == s,
                                     'mean_proxy_actual_rate'].iloc[0]) * 100
                for s in scenarios_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    x = np.arange(len(scenarios_order))
    width = 0.38

    # Model A
    ax1.bar(x - width/2, a_pred, width, label='Predicted',
            color='#C44E52', edgecolor='white', linewidth=1.0)
    ax1.bar(x + width/2, a_actual, width, label='Proxy actual',
            color='#888888', edgecolor='white', linewidth=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios_label)
    ax1.set_ylabel('Default rate (%)')
    ax1.set_title('Model A: Clean training (control)\nGrowing overprediction under stress')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (p, a) in enumerate(zip(a_pred, a_actual)):
        ax1.text(i - width/2, p + 0.5, f'{p:.1f}%', ha='center', fontsize=9)
        ax1.text(i + width/2, a + 0.5, f'{a:.1f}%', ha='center', fontsize=9)

    # Model B
    ax2.bar(x - width/2, b_pred, width, label='Predicted',
            color='#4C72B0', edgecolor='white', linewidth=1.0)
    ax2.bar(x + width/2, b_actual, width, label='Proxy actual',
            color='#888888', edgecolor='white', linewidth=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios_label)
    ax2.set_title('Model B: adversarial-augmented training\n'
                  'Reduced overprediction under stress')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (p, a) in enumerate(zip(b_pred, b_actual)):
        ax2.text(i - width/2, p + 0.5, f'{p:.1f}%', ha='center', fontsize=9)
        ax2.text(i + width/2, a + 0.5, f'{a:.1f}%', ha='center', fontsize=9)

    fig.suptitle('Predicted vs proxy actual default rates: control vs adversarial-augmented model',
                 fontsize=12, y=1.00)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def plot_benign_performance(benign_a, benign_b, filename):
    """
    Counter-test visual: AUC and mean rate error on UNTRANSFORMED 2017
    benchmarks for both models. This shows whether the fix has a cost in
    benign-regime accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    labels = ['Model A\n(clean)', 'Model B\n(adv-augmented)']
    aucs = [benign_a['auc'], benign_b['auc']]
    rate_errs = [benign_a['rate_error_pp'], benign_b['rate_error_pp']]

    bars1 = ax1.bar(labels, aucs, color=['#C44E52', '#4C72B0'],
                    edgecolor='white', linewidth=1.2)
    ax1.set_ylim(0.5, max(aucs) + 0.05)
    ax1.set_ylabel('AUC on 2017 benign benchmark')
    ax1.set_title('Benign-regime ranking ability\n(higher = better)')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005, f'{val:.4f}',
                 ha='center', fontsize=10, fontweight='bold')

    bars2 = ax2.bar(labels, rate_errs, color=['#C44E52', '#4C72B0'],
                    edgecolor='white', linewidth=1.2)
    ax2.axhline(0, color='black', linewidth=0.6, alpha=0.5)
    ax2.set_ylabel('Mean rate error on 2017 benign (pp)')
    ax2.set_title('Benign-regime calibration\n(closer to 0 = better)')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, rate_errs):
        h = bar.get_height()
        offset = 0.15 if h >= 0 else -0.15
        va = 'bottom' if h >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2,
                 h + offset, f'{val:+.2f}pp',
                 ha='center', va=va, fontsize=10, fontweight='bold')

    fig.suptitle('Counter-test: does adversarial augmentation hurt benign performance?',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    start_time = datetime.now()

    print(f"\n{'=' * 80}")
    print("  SOLUTION 1: ADVERSARIAL VALIDATION WITH SYNTHETIC STRESS SCENARIOS")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    # ----- STEP 1: Load + preprocess data (same as v2/v3) -----
    print(f"\n{'-' * 70}")
    print("  STEP 1: Loading LendingClub data")
    print(f"{'-' * 70}")
    filepath, is_extended = find_data_file()
    if not filepath:
        print("\n[ERROR] No LendingClub data file found.")
        sys.exit(1)
    df_raw = load_data(filepath)
    df = preprocess_data(df_raw)
    df_train, df_normal, df_stress = split_data(df)

    # ----- STEP 2: Build proxy ground truth lookup (same as v2) -----
    print(f"\n{'-' * 70}")
    print("  STEP 2: Building proxy ground truth (FICO x DTI lookup)")
    print(f"{'-' * 70}")
    truth_lookup, fico_edges, dti_edges = build_ground_truth_lookup(
        df_stress, n_fico_bins=10, n_dti_bins=5
    )

    # ----- STEP 3: Train MODEL A (control: clean training, identical to v2) -----
    print(f"\n{'-' * 70}")
    print("  STEP 3: Training MODEL A (control - clean training)")
    print(f"{'-' * 70}")
    X_train_a, y_train_a, feat_a, scaler_a, imputer_a = (
        prepare_features_for_training(df_train, ML_FEATURES_CORRECTED, TARGET)
    )
    model_a = train_neural_net(X_train_a, y_train_a, label="MODEL A (control)")

    # ----- STEP 4: Construct adversarial-augmented training data -----
    print(f"\n{'-' * 70}")
    print("  STEP 4: Constructing adversarial-augmented training data for MODEL B")
    print(f"{'-' * 70}")
    df_train_augmented = construct_adversarial_augmented_training_set(
        df_train, truth_lookup, fico_edges, dti_edges
    )

    # ----- STEP 5: Train MODEL B (adversarially hardened) -----
    print(f"\n{'-' * 70}")
    print("  STEP 5: Training MODEL B (adversarially hardened)")
    print(f"{'-' * 70}")
    X_train_b, y_train_b, feat_b, scaler_b, imputer_b = (
        prepare_features_for_training(
            df_train_augmented, ML_FEATURES_CORRECTED, TARGET
        )
    )
    model_b = train_neural_net(X_train_b, y_train_b, label="MODEL B (adv-augmented)")

    # ----- STEP 6: Run identical adversarial validation on BOTH models -----
    print(f"\n{'-' * 70}")
    print("  STEP 6: Adversarial validation gate - applied to BOTH models")
    print(f"{'-' * 70}")
    results_a, baseline_err_a = run_adversarial_validation(
        model_a, scaler_a, imputer_a, feat_a, df_normal,
        truth_lookup, fico_edges, dti_edges, label="MODEL A (control)"
    )
    results_b, baseline_err_b = run_adversarial_validation(
        model_b, scaler_b, imputer_b, feat_b, df_normal,
        truth_lookup, fico_edges, dti_edges, label="MODEL B (adv-augmented)"
    )

    # Save per-scenario results
    full_results = pd.concat([results_a, results_b], axis=0, ignore_index=True)
    full_results.to_csv(
        os.path.join(TABLES_DIR, 'results_per_scenario.csv'), index=False
    )

    # ----- STEP 7: Counter-test on benign 2017 baseline -----
    print(f"\n{'-' * 70}")
    print("  STEP 7: Counter-test - benign-regime performance on 2017")
    print(f"{'-' * 70}")
    benign_a = evaluate_benign_performance(
        model_a, scaler_a, imputer_a, feat_a, df_normal, label="MODEL A (control)"
    )
    benign_b = evaluate_benign_performance(
        model_b, scaler_b, imputer_b, feat_b, df_normal, label="MODEL B (adv-augmented)"
    )

    # ----- STEP 8: Build summary table -----
    severely_a = results_a[results_a['scenario'] == 'severely_adverse'].iloc[0]
    severely_b = results_b[results_b['scenario'] == 'severely_adverse'].iloc[0]
    adverse_a = results_a[results_a['scenario'] == 'adverse'].iloc[0]
    adverse_b = results_b[results_b['scenario'] == 'adverse'].iloc[0]

    summary_rows = [
        {
            'metric': 'Adverse deterioration (pp)',
            'model_a_control': adverse_a['deterioration_vs_baseline_pp'],
            'model_b_adv_augmented': adverse_b['deterioration_vs_baseline_pp'],
            'failure_threshold': CALIBRATION_FAILURE_THRESHOLD_PP,
        },
        {
            'metric': 'Severely Adverse deterioration (pp)',
            'model_a_control': severely_a['deterioration_vs_baseline_pp'],
            'model_b_adv_augmented': severely_b['deterioration_vs_baseline_pp'],
            'failure_threshold': CALIBRATION_FAILURE_THRESHOLD_PP,
        },
        {
            'metric': 'Severely Adverse: flags failure?',
            'model_a_control': bool(severely_a['flags_failure']),
            'model_b_adv_augmented': bool(severely_b['flags_failure']),
            'failure_threshold': '-',
        },
        {
            'metric': 'Benign 2017 AUC',
            'model_a_control': benign_a['auc'],
            'model_b_adv_augmented': benign_b['auc'],
            'failure_threshold': '-',
        },
        {
            'metric': 'Benign 2017 rate error (pp)',
            'model_a_control': benign_a['rate_error_pp'],
            'model_b_adv_augmented': benign_b['rate_error_pp'],
            'failure_threshold': '-',
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(TABLES_DIR, 'summary_comparison.csv'), index=False
    )

    # ----- STEP 9: Visualisations -----
    print(f"\n{'-' * 70}")
    print("  STEP 9: Generating visualisations")
    print(f"{'-' * 70}")
    plot_deterioration_comparison(
        results_a, results_b, 'deterioration_comparison.png'
    )
    plot_predicted_vs_actual_comparison(
        results_a, results_b, 'predicted_vs_actual.png'
    )
    plot_benign_performance(
        benign_a, benign_b, 'benign_performance.png'
    )

    # ----- FINAL SUMMARY -----
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 80}")
    print("  SUMMARY: SOLUTION 1 RESULTS")
    print(f"{'=' * 80}")
    print(f"\n  Failure-onset threshold: {CALIBRATION_FAILURE_THRESHOLD_PP:.2f}pp")
    print(f"\n  Stress-induced deterioration (severely adverse scenario):")
    print(f"    Model A (clean training):       "
          f"{severely_a['deterioration_vs_baseline_pp']:+.2f}pp"
          f"  ({'FLAGS FAILURE' if severely_a['flags_failure'] else 'passes'})")
    print(f"    Model B (adv-augmented):        "
          f"{severely_b['deterioration_vs_baseline_pp']:+.2f}pp"
          f"  ({'FLAGS FAILURE' if severely_b['flags_failure'] else 'passes'})")

    if severely_a['deterioration_vs_baseline_pp'] != 0:
        reduction_pct = 100 * (
            1 - severely_b['deterioration_vs_baseline_pp']
            / severely_a['deterioration_vs_baseline_pp']
        )
        print(f"\n  Relative reduction in stress deterioration: "
              f"{reduction_pct:+.1f}%")

    print(f"\n  Benign-regime counter-test (2017 untransformed):")
    print(f"    Model A AUC: {benign_a['auc']:.4f}  |  rate error: "
          f"{benign_a['rate_error_pp']:+.2f}pp")
    print(f"    Model B AUC: {benign_b['auc']:.4f}  |  rate error: "
          f"{benign_b['rate_error_pp']:+.2f}pp")
    auc_delta = benign_b['auc'] - benign_a['auc']
    if abs(auc_delta) < 0.01:
        print(f"    AUC change: {auc_delta:+.4f}  -- "
              f"hardening did not materially affect benign discrimination")
    else:
        print(f"    AUC change: {auc_delta:+.4f}  -- "
              f"flag for prose: hardening had a measurable effect on benign AUC")

    print(f"\n  Outputs:")
    print(f"    Tables:  {TABLES_DIR}/")
    print(f"    Figures: {FIGURES_DIR}/")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()