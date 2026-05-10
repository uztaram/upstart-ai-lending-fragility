#!/usr/bin/env python3
"""
MSIN0032 Dissertation - Part II Empirical Test
==============================================
Solution 2: Real-Time Distribution Monitoring with Adaptive Recalibration

This script implements the empirical test of Solution 2 from Part II of the
dissertation. It evaluates whether Population Stability Index (PSI) monitoring
of input feature distributions, combined with PSI-triggered model retraining,
materially reduces stress-period calibration failure under macroeconomic
regime shift.

METHODOLOGY OVERVIEW

    Two ML models with identical architecture are trained and compared. Both
    use the MLPClassifier neural network defined in pilot_study.py. The only
    difference between them is the training-data window:
    
        Model A (control / no retraining):
            Training data: 2012-2016 only.
            Represents the status quo where PSI is not monitored, so the
            model continues in production unchanged through the 2018-2020
            stress window.
        
        Model B (PSI-triggered retraining):
            Training data: 2012-2017.
            Represents the alternative where a 2017Q4 PSI breach triggered
            a retraining cycle completed in early 2018, before the stress
            period began.
    
    Both models are evaluated quarterly across 2018Q1 to 2020Q3, with per-
    quarter calibration error tracked against a 4.26pp failure-onset
    threshold (twice the 2017 baseline calibration error of 2.13pp). The
    headline comparison is whether Model B's quarterly trajectory stays
    below threshold longer than Model A's.
    
    A counter-test on benign 2017 data confirms the additional training
    year does not introduce regressions in benign-regime performance.

INTELLECTUAL CONTRIBUTION

    PSI monitoring is widely deployed in industry (Siddiqi 2006), but the
    literature focuses overwhelmingly on threshold selection and detection
    reliability. The remediation question (what should happen when the
    alarm fires) is comparatively under-treated. This script provides a
    concrete answer: simulating the 2017Q4 alarm-triggered retraining
    cycle and measuring whether it would have prevented the documented
    2018Q1 calibration failure onset.

DESIGN CHOICES

    One-year retraining horizon (add 2017 only): matches what would
    realistically happen in operations. The 2017Q4 alarm fires; an
    early-2018 retrain has access to 2012-2017 issued loans (the data
    available at the time the alarm fires). Using 2018 data in training
    would give Model B information unavailable at the moment of decision.
    
    Same architecture and hyperparameters as the Part I pilot study:
    isolates the effect of the retraining decision. Any change in
    stress-period behaviour is attributable to the additional training
    year, not architectural differences.
    
    Quarterly evaluation cadence matches typical model-monitoring practice
    and gives the reader a per-quarter trajectory that can be visually
    compared against the failure threshold.

OUTPUTS

    A subdirectory `solution_2_outputs/` is created next to this script.
    Figures: calibration_trajectory.png, predicted_vs_actual_quarterly.png,
    benign_performance.png. Tables: quarterly_results.csv,
    summary_comparison.csv.

USAGE

    python solution_2.py
    
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
        calculate_calibration_error,
    )
except ImportError as e:
    print(f"[ERROR] Could not import from pilot_study: {e}")
    print("        Make sure pilot_study.py is in the same directory "
          "as this script.")
    sys.exit(1)

warnings.filterwarnings('ignore')


# ============================================================================
# PSI-MONITORING COMPONENTS
# ----------------------------------------------------------------------------
# These were previously imported from a separate v2 file. They are inlined
# here so that solution_2.py is fully self-contained and depends only on
# pilot_study.py.
# ============================================================================

# Monitoring window: 2017Q1 through 2020Q3 (continuous).
MONITORING_START_YEAR = 2017
MONITORING_END_YEAR = 2020
MONITORING_END_QUARTER = 3   # LendingClub data ends at Q3 2020

# PSI thresholds (Siddiqi 2006 industry standard).
PSI_THRESHOLDS = {
    'stable':   0.10,   # Below: no action
    'moderate': 0.25,   # Above: investigate / retrain
    'critical': 0.50,   # Above: emergency recalibration
}

# Calibration-failure ground truth: first quarter where ECE >= 2x normal baseline.
CALIBRATION_FAILURE_MULTIPLIER = 2.0


def quarter_label(year, quarter):
    """Convert (year, quarter) -> 'YYYYQ#' string."""
    return f"{int(year)}Q{int(quarter)}"


def quarter_index(year, quarter):
    """Convert (year, quarter) -> integer index for time-series ordering."""
    return int(year) * 4 + int(quarter) - 1


def get_quarterly_subsets(df, start_year, end_year, end_quarter):
    """Yield (label, year, q, subset) for each quarter in range (continuous)."""
    for year in range(start_year, end_year + 1):
        for q in range(1, 5):
            if year == end_year and q > end_quarter:
                break
            mask = (df['issue_year'] == year) & (df['issue_quarter'] == q)
            subset = df[mask]
            if len(subset) > 0:
                yield quarter_label(year, q), year, q, subset


def build_monitoring_set(df_full):
    """
    Build a continuous monitoring time series from the preprocessed pilot
    dataframe. The pilot's split_data drops 2018 by design (training is
    2012-2016, normal is 2017, stress is 2019-2020). Solution 2 needs the
    full 2017-2020 window for quarterly PSI evaluation, so the monitoring
    set is rebuilt directly here.
    """
    mask = (
        (df_full['issue_year'] >= MONITORING_START_YEAR) &
        (
            (df_full['issue_year'] < MONITORING_END_YEAR) |
            ((df_full['issue_year'] == MONITORING_END_YEAR) &
             (df_full['issue_quarter'] <= MONITORING_END_QUARTER))
        )
    )
    monitoring = df_full[mask].copy()

    print(f"\n  Monitoring window: {len(monitoring):,} loans, "
          f"{MONITORING_START_YEAR}Q1 - {MONITORING_END_YEAR}Q{MONITORING_END_QUARTER}")
    print(f"  Year breakdown:")
    for year in range(MONITORING_START_YEAR, MONITORING_END_YEAR + 1):
        n = (monitoring['issue_year'] == year).sum()
        print(f"    {year}: {n:>8,} loans")

    return monitoring

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "solution_2_outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Failure threshold (same as v2 monitoring + Solution 1 fix demo)
CALIBRATION_FAILURE_THRESHOLD_PP = 4.26

# Stress evaluation window (the period both models are evaluated against)
STRESS_EVAL_START_YEAR = 2018
STRESS_EVAL_END_YEAR = 2020
STRESS_EVAL_END_QUARTER = 3

# Counter-test: benign 2017 evaluation
BENIGN_EVAL_YEAR = 2017

plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 10,
})


# ============================================================================
# FEATURE PREPARATION (replicates v3 prepare_features for training-only use)
# ============================================================================

def prepare_features_for_training(df_train, feature_list, target):
    """
    Fit imputer + scaler on a training set, return scaled X_train, y_train,
    plus the imputer + scaler so we can apply them at scoring time.

    Identical pipeline to v3 / Solution 1 fix demo.
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
    (2012-2016 training) and Model B (2012-2017 training) so that any
    difference in behaviour is attributable to training data scope only,
    not architecture.
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
# QUARTERLY EVALUATION (per-quarter calibration error during stress)
# ============================================================================

def evaluate_model_on_quarterly_stress(
    model, scaler, imputer, feature_list, df_full,
    start_year, end_year, end_quarter, target_col, label="MODEL"
):
    """
    Score a model on each quarter from start_year-Q1 to end_year-end_quarter.
    Returns a DataFrame with per-quarter:
      - n_loans
      - predicted default rate (mean)
      - actual default rate (mean) -- using EARLY_DQ_TARGET to match v2 PSI logic
      - rate error (pp)
      - calibration error (ECE, in pp)
      - AUC
    """
    print(f"\n  Evaluating {label} quarterly through "
          f"{start_year}Q1 - {end_year}Q{end_quarter}...")

    rows = []
    for qlabel, year, q, subset in get_quarterly_subsets(
        df_full, start_year, end_year, end_quarter
    ):
        sub_features = subset[feature_list].copy()
        for f in feature_list:
            if f not in sub_features.columns:
                sub_features[f] = 0

        sub_imputed = pd.DataFrame(
            imputer.transform(sub_features),
            columns=feature_list, index=sub_features.index
        )
        sub_scaled = scaler.transform(sub_imputed)

        y_prob = model.predict_proba(sub_scaled)[:, 1]
        y_true = subset[target_col].values

        cal_err = calculate_calibration_error(y_true, y_prob)
        cal_err_pp = (cal_err * 100
                      if cal_err is not None and not np.isnan(cal_err)
                      else np.nan)

        try:
            auc = (roc_auc_score(y_true, y_prob)
                   if len(np.unique(y_true)) > 1 else np.nan)
        except Exception:
            auc = np.nan

        mean_pred = float(np.mean(y_prob))
        mean_actual = float(np.mean(y_true))

        rows.append({
            'model': label,
            'quarter': qlabel,
            'year': year,
            'q': q,
            'q_idx': quarter_index(year, q),
            'n_loans': len(subset),
            'predicted_default_rate': round(mean_pred, 4),
            'actual_default_rate': round(mean_actual, 4),
            'rate_error_pp': round((mean_pred - mean_actual) * 100, 2),
            'calibration_error_pp': (round(cal_err_pp, 2)
                                      if not np.isnan(cal_err_pp) else np.nan),
            'auc_roc': round(auc, 4) if not np.isnan(auc) else np.nan,
            'breaches_failure_threshold':
                bool(cal_err_pp >= CALIBRATION_FAILURE_THRESHOLD_PP)
                if not np.isnan(cal_err_pp) else False,
        })

    return pd.DataFrame(rows)


# ============================================================================
# COUNTER-TEST: BENIGN-REGIME PERFORMANCE ON 2017
# ============================================================================

def evaluate_benign_2017_performance(
    model, scaler, imputer, feature_list, df_normal_2017, target_col, label="MODEL"
):
    """
    Score the model on 2017 borrowers (the benign year).

    NOTE: For Model B this is partially in-sample because Model B's training
    data INCLUDES 2017. We acknowledge this and report it primarily for
    Model A (out-of-sample for it), with Model B's number reported for
    completeness. The substantive counter-test is whether Model B doesn't
    REGRESS on benign data - if AUC stays comparable to Model A's, the
    retraining hasn't damaged anything.
    """
    df = df_normal_2017.copy()
    y_true = df[target_col].values

    sub_features = df[feature_list].copy()
    for f in feature_list:
        if f not in sub_features.columns:
            sub_features[f] = 0

    sub_imputed = pd.DataFrame(
        imputer.transform(sub_features), columns=feature_list, index=sub_features.index
    )
    sub_scaled = scaler.transform(sub_imputed)
    y_prob = model.predict_proba(sub_scaled)[:, 1]

    auc = roc_auc_score(y_true, y_prob)
    mean_pred = float(np.mean(y_prob))
    mean_actual = float(np.mean(y_true))
    rate_err_pp = (mean_pred - mean_actual) * 100

    print(f"\n  {label} benign 2017 performance:")
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

def plot_calibration_trajectory(results_a, results_b, filename):
    """
    HEADLINE FIGURE: per-quarter calibration error for Model A vs Model B
    through 2018-2020 stress, with the failure threshold marked.

    The story this figure tells: if Model B's curve sits meaningfully below
    Model A's, then PSI-triggered retraining was actionable. If they
    overlap, the alarm bought no remediation.
    """
    quarters = results_a['quarter'].tolist()
    a_cal = results_a['calibration_error_pp'].tolist()
    b_cal = results_b['calibration_error_pp'].tolist()

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(quarters, a_cal, marker='o', linewidth=2.5, color='#C44E52',
            label='Model A: 2012-2016 training (no PSI-triggered retraining)',
            markersize=8, markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(quarters, b_cal, marker='s', linewidth=2.5, color='#4C72B0',
            label='Model B: 2012-2017 training (PSI-triggered retraining)',
            markersize=8, markeredgecolor='white', markeredgewidth=1.5)

    ax.axhline(
        CALIBRATION_FAILURE_THRESHOLD_PP, color='red', linestyle='--',
        linewidth=1.5, alpha=0.7,
        label=f'Failure-onset threshold ({CALIBRATION_FAILURE_THRESHOLD_PP:.1f}pp)'
    )

    # Annotate first-breach quarters where applicable
    for series_label, cal_list, color in [
        ('Model A', a_cal, '#C44E52'), ('Model B', b_cal, '#4C72B0')
    ]:
        for i, val in enumerate(cal_list):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if val >= CALIBRATION_FAILURE_THRESHOLD_PP:
                ax.annotate(
                    f'{series_label}\nfirst breach',
                    xy=(quarters[i], val),
                    xytext=(0, 20), textcoords='offset points',
                    ha='center', fontsize=8, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1)
                )
                break  # only annotate first breach per series

    ax.set_xlabel('Quarter')
    ax.set_ylabel('Calibration error (pp)')
    ax.set_title(
        'Quarterly calibration error trajectory through 2018-2020 stress\n'
        'Model B (PSI-triggered retraining) vs Model A (no retraining)',
        fontsize=12
    )
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def plot_predicted_vs_actual_quarterly(results_a, results_b, filename):
    """
    Two-panel: per-quarter predicted vs actual default rates for both models.
    Shows whether Model B reduces the systematic overprediction that Model A
    exhibits during stress.
    """
    quarters = results_a['quarter'].tolist()

    a_pred = (results_a['predicted_default_rate'] * 100).tolist()
    a_actual = (results_a['actual_default_rate'] * 100).tolist()
    b_pred = (results_b['predicted_default_rate'] * 100).tolist()
    b_actual = (results_b['actual_default_rate'] * 100).tolist()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    ax1.plot(quarters, a_pred, marker='o', linewidth=2, color='#C44E52',
             label='Predicted DR', markersize=7)
    ax1.plot(quarters, a_actual, marker='o', linewidth=2, color='#888888',
             label='Actual DR', markersize=7, linestyle='--')
    ax1.fill_between(range(len(quarters)), a_pred, a_actual, alpha=0.15,
                      color='#C44E52')
    ax1.set_ylabel('Default rate (%)')
    ax1.set_title('Model A: 2012-2016 training (control)\n'
                  'Predicted vs actual default rates per quarter')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.plot(quarters, b_pred, marker='s', linewidth=2, color='#4C72B0',
             label='Predicted DR', markersize=7)
    ax2.plot(quarters, b_actual, marker='s', linewidth=2, color='#888888',
             label='Actual DR', markersize=7, linestyle='--')
    ax2.fill_between(range(len(quarters)), b_pred, b_actual, alpha=0.15,
                      color='#4C72B0')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Default rate (%)')
    ax2.set_title('Model B: 2012-2017 training (PSI-triggered retraining)\n'
                  'Predicted vs actual default rates per quarter')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def plot_benign_performance(benign_a, benign_b, filename):
    """
    Counter-test visual: AUC and rate error on 2017 benign data for both
    models. Note that Model B has 2017 in its training set so this is a
    leak-aware comparison; the substantive question is whether Model A
    (out-of-sample on 2017) is meaningfully WORSE than Model B, which
    would indicate the retraining produced a strict improvement on
    benign data without harming it.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    labels = ['Model A\n(2012-2016)', 'Model B\n(2012-2017)']
    aucs = [benign_a['auc'], benign_b['auc']]
    rate_errs = [benign_a['rate_error_pp'], benign_b['rate_error_pp']]

    bars1 = ax1.bar(labels, aucs, color=['#C44E52', '#4C72B0'],
                    edgecolor='white', linewidth=1.2)
    ax1.set_ylim(0.5, max(aucs) + 0.05)
    ax1.set_ylabel('AUC on 2017 benign')
    ax1.set_title('Benign-regime ranking ability\n(higher = better)')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005, f'{val:.4f}',
                 ha='center', fontsize=10, fontweight='bold')

    bars2 = ax2.bar(labels, rate_errs, color=['#C44E52', '#4C72B0'],
                    edgecolor='white', linewidth=1.2)
    ax2.axhline(0, color='black', linewidth=0.6, alpha=0.5)
    ax2.set_ylabel('Rate error on 2017 benign (pp)')
    ax2.set_title('Benign-regime calibration\n(closer to 0 = better)')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, rate_errs):
        h = bar.get_height()
        offset = 0.15 if h >= 0 else -0.15
        va = 'bottom' if h >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2,
                 h + offset, f'{val:+.2f}pp',
                 ha='center', va=va, fontsize=10, fontweight='bold')

    fig.suptitle(
        'Counter-test: does PSI-triggered retraining hurt benign performance?\n'
        '(NOTE: Model B has 2017 in training - test is conservative; '
        'the question is whether Model A is meaningfully worse on its own '
        'out-of-sample 2017 than Model B is on its in-sample 2017.)',
        fontsize=10, y=1.02
    )
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
    print("  SOLUTION 2: PSI MONITORING WITH ADAPTIVE RECALIBRATION")
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
    df_train_2012_2016, df_normal_2017, df_stress_2019_2020 = split_data(df)

    # Build the continuous monitoring set (includes 2018, which v3's split drops)
    print(f"\n{'-' * 70}")
    print("  STEP 2: Building continuous monitoring set (2017Q1-2020Q3)")
    print(f"{'-' * 70}")
    df_monitor = build_monitoring_set(df)

    # ----- STEP 3: Build training sets for both models -----
    print(f"\n{'-' * 70}")
    print("  STEP 3: Constructing training sets")
    print(f"{'-' * 70}")

    # Model A: 2012-2016 (status quo)
    df_train_a = df_train_2012_2016.copy()

    # Model B: 2012-2017 (PSI alarm fires 2017Q4 -> retrain in early 2018
    #                     with all data available at that moment)
    df_train_b = df[
        (df['issue_year'] >= 2012) & (df['issue_year'] <= 2017)
    ].copy()

    print(f"\n  Model A training set (2012-2016): {len(df_train_a):,} loans")
    print(f"    Default rate: {df_train_a[TARGET].mean():.1%}")
    print(f"\n  Model B training set (2012-2017): {len(df_train_b):,} loans "
          f"(+{len(df_train_b) - len(df_train_a):,} from 2017)")
    print(f"    Default rate: {df_train_b[TARGET].mean():.1%}")

    # ----- STEP 4: Train Model A (control) -----
    print(f"\n{'-' * 70}")
    print("  STEP 4: Training MODEL A (control - 2012-2016)")
    print(f"{'-' * 70}")
    X_train_a, y_train_a, feat_a, scaler_a, imputer_a = (
        prepare_features_for_training(df_train_a, ML_FEATURES_CORRECTED, TARGET)
    )
    model_a = train_neural_net(X_train_a, y_train_a, label="MODEL A (2012-2016)")

    # ----- STEP 5: Train Model B (PSI-triggered retraining) -----
    print(f"\n{'-' * 70}")
    print("  STEP 5: Training MODEL B (PSI-triggered retraining - 2012-2017)")
    print(f"{'-' * 70}")
    X_train_b, y_train_b, feat_b, scaler_b, imputer_b = (
        prepare_features_for_training(df_train_b, ML_FEATURES_CORRECTED, TARGET)
    )
    model_b = train_neural_net(X_train_b, y_train_b, label="MODEL B (2012-2017)")

    # ----- STEP 6: Quarterly stress evaluation (2018-2020) for BOTH models -----
    print(f"\n{'-' * 70}")
    print("  STEP 6: Quarterly stress evaluation (2018Q1-2020Q3)")
    print(f"{'-' * 70}")

    # We use EARLY_DQ_TARGET to match v2 PSI script's evaluation logic
    # (early delinquency is the available signal in 2019-2020 loans that
    #  haven't reached full term yet)
    results_a = evaluate_model_on_quarterly_stress(
        model_a, scaler_a, imputer_a, feat_a, df_monitor,
        STRESS_EVAL_START_YEAR, STRESS_EVAL_END_YEAR, STRESS_EVAL_END_QUARTER,
        EARLY_DQ_TARGET, label="MODEL A (2012-2016)"
    )
    results_b = evaluate_model_on_quarterly_stress(
        model_b, scaler_b, imputer_b, feat_b, df_monitor,
        STRESS_EVAL_START_YEAR, STRESS_EVAL_END_YEAR, STRESS_EVAL_END_QUARTER,
        EARLY_DQ_TARGET, label="MODEL B (2012-2017)"
    )

    full_results = pd.concat([results_a, results_b], axis=0, ignore_index=True)
    full_results.to_csv(
        os.path.join(TABLES_DIR, 'quarterly_results.csv'), index=False
    )

    # Print quarterly comparison
    print(f"\n  Quarterly calibration error (pp):")
    print(f"  {'Quarter':<10}{'Model A':>12}{'Model B':>12}{'Improvement':>15}")
    print(f"  {'-' * 49}")
    for i in range(len(results_a)):
        qa = results_a.iloc[i]
        qb = results_b.iloc[i]
        a_val = qa['calibration_error_pp']
        b_val = qb['calibration_error_pp']
        improvement = a_val - b_val if not (np.isnan(a_val) or np.isnan(b_val)) else np.nan
        print(f"  {qa['quarter']:<10}{a_val:>10.2f}pp"
              f"{b_val:>10.2f}pp"
              f"{improvement:>+13.2f}pp")

    # ----- STEP 7: Counter-test on benign 2017 -----
    print(f"\n{'-' * 70}")
    print("  STEP 7: Counter-test - benign-regime performance on 2017")
    print(f"{'-' * 70}")
    print("  NOTE: 2017 is in Model B's training set (in-sample);")
    print("        out-of-sample for Model A. Conservative comparison.")
    benign_a = evaluate_benign_2017_performance(
        model_a, scaler_a, imputer_a, feat_a, df_normal_2017, EARLY_DQ_TARGET,
        label="MODEL A (2012-2016)"
    )
    benign_b = evaluate_benign_2017_performance(
        model_b, scaler_b, imputer_b, feat_b, df_normal_2017, EARLY_DQ_TARGET,
        label="MODEL B (2012-2017)"
    )

    # ----- STEP 8: Build summary table -----
    a_mean_stress = float(results_a['calibration_error_pp'].mean())
    b_mean_stress = float(results_b['calibration_error_pp'].mean())
    a_max_stress = float(results_a['calibration_error_pp'].max())
    b_max_stress = float(results_b['calibration_error_pp'].max())
    a_breaches = int(results_a['breaches_failure_threshold'].sum())
    b_breaches = int(results_b['breaches_failure_threshold'].sum())

    summary_rows = [
        {
            'metric': 'Mean stress-period calibration error (pp)',
            'model_a_control': round(a_mean_stress, 2),
            'model_b_psi_retraining': round(b_mean_stress, 2),
            'improvement_pp': round(a_mean_stress - b_mean_stress, 2),
        },
        {
            'metric': 'Peak stress-period calibration error (pp)',
            'model_a_control': round(a_max_stress, 2),
            'model_b_psi_retraining': round(b_max_stress, 2),
            'improvement_pp': round(a_max_stress - b_max_stress, 2),
        },
        {
            'metric': 'Quarters breaching 4.26pp threshold',
            'model_a_control': a_breaches,
            'model_b_psi_retraining': b_breaches,
            'improvement_pp': a_breaches - b_breaches,
        },
        {
            'metric': 'Benign 2017 AUC',
            'model_a_control': benign_a['auc'],
            'model_b_psi_retraining': benign_b['auc'],
            'improvement_pp': '-',
        },
        {
            'metric': 'Benign 2017 rate error (pp)',
            'model_a_control': benign_a['rate_error_pp'],
            'model_b_psi_retraining': benign_b['rate_error_pp'],
            'improvement_pp': '-',
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
    plot_calibration_trajectory(
        results_a, results_b, 'calibration_trajectory.png'
    )
    plot_predicted_vs_actual_quarterly(
        results_a, results_b, 'predicted_vs_actual_quarterly.png'
    )
    plot_benign_performance(
        benign_a, benign_b, 'benign_performance.png'
    )

    # ----- FINAL SUMMARY -----
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 80}")
    print("  SUMMARY: SOLUTION 2 RESULTS")
    print(f"{'=' * 80}")
    print(f"\n  Failure-onset threshold: {CALIBRATION_FAILURE_THRESHOLD_PP:.2f}pp")
    print(f"\n  Stress-period calibration error (2018Q1 - 2020Q3):")
    print(f"    Mean:")
    print(f"      Model A (no retraining):       {a_mean_stress:.2f}pp")
    print(f"      Model B (PSI retraining):      {b_mean_stress:.2f}pp")
    if a_mean_stress != 0:
        rel_red = 100 * (a_mean_stress - b_mean_stress) / a_mean_stress
        print(f"      Relative improvement:          {rel_red:+.1f}%")
    print(f"    Peak:")
    print(f"      Model A:                       {a_max_stress:.2f}pp")
    print(f"      Model B:                       {b_max_stress:.2f}pp")
    print(f"    Quarters breaching threshold:")
    print(f"      Model A:                       {a_breaches} of {len(results_a)}")
    print(f"      Model B:                       {b_breaches} of {len(results_b)}")

    print(f"\n  Benign 2017 counter-test:")
    print(f"    Model A AUC: {benign_a['auc']:.4f}  |  rate error: "
          f"{benign_a['rate_error_pp']:+.2f}pp  (out-of-sample)")
    print(f"    Model B AUC: {benign_b['auc']:.4f}  |  rate error: "
          f"{benign_b['rate_error_pp']:+.2f}pp  (in-sample)")

    print(f"\n  Interpretation guide:")
    if a_mean_stress > b_mean_stress:
        print(f"    Mean stress-period calibration error decreased "
              f"{a_mean_stress - b_mean_stress:.2f}pp")
        print(f"    PSI-triggered retraining was actionable: the alarm")
        print(f"    bought meaningful stress-period calibration improvement.")
    else:
        print(f"    Mean stress-period calibration error did NOT decrease.")
        print(f"    PSI alarm was informational but not directly actionable")
        print(f"    via simple retraining alone - flag for prose discussion.")

    print(f"\n  Outputs:")
    print(f"    Tables:  {TABLES_DIR}/")
    print(f"    Figures: {FIGURES_DIR}/")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
