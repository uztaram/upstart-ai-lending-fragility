#!/usr/bin/env python3
"""
MSIN0032 Dissertation - Part II Empirical Test
==============================================
Solution 3: Confidence-Conditional Decisioning via Adaptive Conformal Inference

This script implements the empirical test of Solution 3 from Part II of the
dissertation. It evaluates whether Adaptive Conformal Inference (Gibbs and
Candès, 2021) provides reliable per-decision confidence wrappers under
macroeconomic regime shift, where vanilla split conformal prediction's
exchangeability assumption breaks.

THEORETICAL MOTIVATION

    Standard split conformal prediction (Vovk et al., 2005; Angelopoulos and
    Bates, 2023) provides a finite-sample marginal coverage guarantee
    P(y in C(x)) >= 1 - alpha under exchangeability between calibration and
    test data. Regime shift violates exchangeability: 2016 calibration
    borrowers are not exchangeable with 2018-2020 stress-period borrowers.
    Adaptive Conformal Inference relaxes the exchangeability requirement to
    a long-run marginal coverage guarantee that holds under arbitrary
    distribution shift, achieved by a simple online learning update on the
    alpha parameter at each evaluation step.

ALGORITHM (Gibbs and Candès, 2021)

    Initialise alpha_1 = alpha_target (e.g. 0.10 for 90% coverage).
    For each evaluation quarter t = 1, 2, ...:
        1. Form prediction set C_t(x_t) using current alpha_t and the
           frozen 2016 calibration set.
        2. Observe miscoverage err_t (= 1 if y_t not in C_t else 0)
           averaged across the quarter.
        3. Update: alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t),
           where gamma = 0.05 is the learning rate.
    
    The Gibbs and Candès (2021) Proposition 4.1 gives long-run miscoverage
    convergence to alpha_target irrespective of distribution shift.

METHODOLOGY OVERVIEW

    Three configurations are compared against a Model A (2012-2016 trained
    MLPClassifier, identical to the pilot study):
    
        Config 1 (control):    Vanilla split conformal, fixed alpha = 0.10
        Config 2 (ACI alone):  ACI with alpha_target = 0.10, gamma = 0.05
        Config 3 (layered):    ACI on top of Model B (the 2012-2017
                               PSI-retrained model from Solution 2),
                               representing the integrated layered defence
    
    All three are evaluated quarterly through 2018Q1-2020Q3. Per quarter we
    measure: empirical coverage rate, abstention rate, calibration error
    on retained (confident) cases, and default rate of abstained versus
    confident cases.

DESIGN CHOICES

    gamma = 0.05: standard choice in Gibbs and Candès (2021) experiments;
    larger gamma adapts faster but with more variance.
    
    alpha_target = 0.10: 90% target coverage, matching prior conformal
    work in this dissertation and consistent with credit-risk practice.
    
    Quarterly batch update: in pure ACI the update is per-observation; for
    credit risk we use the natural quarterly cadence (Bhatnagar et al.,
    2023, batched-ACI variant). The long-run guarantee is preserved.
    
    Calibration set fixed at 2016: not refreshed across stress quarters,
    isolating the effect of the alpha-adaptation rule from the confound
    of updated calibration data.

OUTPUTS

    A subdirectory `solution_3_outputs/` is created next to this script.
    Figures: aci_alpha_trajectory.png, aci_vs_vanilla_coverage.png,
    aci_vs_vanilla_calibration.png, aci_vs_vanilla_abstention.png,
    layered_defence.png, method_card.png.
    Tables: aci_quarterly_results.csv, aci_summary_comparison.csv.

USAGE

    python solution_3.py
    
    Requires pilot_study.py in the same directory. The LendingClub CSV
    path is configured inside pilot_study.py.

Author: Marat Daminov

REFERENCES
    Gibbs, I. and Candès, E. J. (2021) 'Adaptive Conformal Inference Under
        Distribution Shift'. NeurIPS 2021.
    Vovk, V., Gammerman, A. and Shafer, G. (2005). Algorithmic Learning in
        a Random World. Springer.
    Angelopoulos, A. N. and Bates, S. (2023). Conformal Prediction: A
        Gentle Introduction. Foundations and Trends in ML.
    Bhatnagar et al. (2023). 'Improved Online Conformal Prediction via
        Strongly Adaptive Online Learning'. ICML 2023.
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
# QUARTERLY MONITORING HELPERS
# ----------------------------------------------------------------------------
# These were previously imported from a separate v2 file. They are inlined
# here so that solution_3.py is fully self-contained and depends only on
# pilot_study.py.
# ============================================================================

# Monitoring window: 2018Q1 through 2020Q3 (the stress evaluation window).
_MONITORING_START_YEAR = 2018
_MONITORING_END_YEAR = 2020
_MONITORING_END_QUARTER = 3


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
    Build the continuous quarterly monitoring set covering 2018Q1-2020Q3.
    The pilot's split_data drops 2018 (training is 2012-2016, normal is 2017,
    stress is 2019-2020), so the monitoring set is rebuilt directly here.
    """
    mask = (
        (df_full['issue_year'] >= _MONITORING_START_YEAR) &
        (
            (df_full['issue_year'] < _MONITORING_END_YEAR) |
            ((df_full['issue_year'] == _MONITORING_END_YEAR) &
             (df_full['issue_quarter'] <= _MONITORING_END_QUARTER))
        )
    )
    monitoring = df_full[mask].copy()

    print(f"\n  Monitoring window: {len(monitoring):,} loans, "
          f"{_MONITORING_START_YEAR}Q1 - {_MONITORING_END_YEAR}Q{_MONITORING_END_QUARTER}")
    print(f"  Year breakdown:")
    for year in range(_MONITORING_START_YEAR, _MONITORING_END_YEAR + 1):
        n = (monitoring['issue_year'] == year).sum()
        print(f"    {year}: {n:>8,} loans")

    return monitoring


# ============================================================================
# CONFORMAL PREDICTION HELPERS
# ----------------------------------------------------------------------------
# Standard split conformal building blocks. Inlined here so solution_3.py
# does not depend on any external script beyond pilot_study.py.
# ============================================================================

def compute_nonconformity_scores(y_true, y_prob):
    """Standard binary classification nonconformity score: |y - p|."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return np.abs(y_true - y_prob)


def conformal_threshold(calibration_scores, alpha):
    """Finite-sample-corrected (1 - alpha) quantile per Vovk et al. (2005)."""
    n = len(calibration_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(calibration_scores, q_level))


def conformal_prediction_sets(y_prob, tau):
    """
    Build prediction sets {0}, {1}, {0, 1} (abstain) or empty set from
    predicted probabilities and a conformal threshold tau.
    """
    y_prob = np.asarray(y_prob)
    contains_1 = (1 - y_prob) <= tau
    contains_0 = y_prob <= tau
    set_size = contains_0.astype(int) + contains_1.astype(int)
    return pd.DataFrame({
        'p_default': y_prob,
        'contains_0': contains_0,
        'contains_1': contains_1,
        'set_size': set_size,
    })


def score_period(model, scaler, imputer, feature_list, df_period):
    """Score an evaluation period through the trained model."""
    sub = df_period[feature_list].copy()
    for f in feature_list:
        if f not in sub.columns:
            sub[f] = 0
    imputed = pd.DataFrame(
        imputer.transform(sub), columns=feature_list, index=sub.index
    )
    scaled = scaler.transform(imputed)
    return model.predict_proba(scaled)[:, 1]


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "solution_3_outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ACI hyperparameters
ALPHA_TARGET = 0.10        # target miscoverage rate (90% target coverage)
ACI_GAMMA = 0.05           # ACI learning rate (Gibbs & Candès 2021)
ACI_ALPHA_MIN = 0.001      # numerical floor (alpha cannot be <= 0)
ACI_ALPHA_MAX = 0.999      # numerical ceiling (alpha cannot be >= 1)

# Stress evaluation window
STRESS_EVAL_START_YEAR = 2018
STRESS_EVAL_END_YEAR = 2020
STRESS_EVAL_END_QUARTER = 3

# Calibration year (held-out subset of training period for conformal calibration)
CALIBRATION_YEAR = 2016
CALIBRATION_FRACTION = 0.20   # use 20% of 2016 loans as calibration set

# Layered defence: also evaluate ACI on a Model B (PSI-retrained) variant
INCLUDE_LAYERED_DEFENCE = True

plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 10,
})

# Visual palette - consistent with prior fix scripts
PALETTE = {
    'vanilla': '#C44E52',         # red - control
    'aci': '#4C72B0',             # blue - fix
    'layered': '#55A868',          # green - layered defence
    'target': '#444444',           # dark grey - reference lines
    'threshold': '#888888',        # mid grey - other reference
}


# ============================================================================
# FEATURE PREP & MODEL TRAINING (consistent with Solutions 1 & 2 fix demos)
# ============================================================================

def prepare_features_for_training(df_train, feature_list, target):
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


def train_neural_net(X_train, y_train, label="MODEL"):
    """MLPClassifier matching v3 / Solutions 1 & 2 fix demos exactly."""
    print(f"\n  Training {label} (MLPClassifier)...")
    print(f"    Architecture: 3 layers (100, 50, 25 neurons)")
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
    return model


# ============================================================================
# ADAPTIVE CONFORMAL INFERENCE - the heart of the fix
# ============================================================================

def evaluate_conformal_quarter(y_prob, y_true, tau):
    """
    Evaluate one quarter through a fixed conformal threshold tau.

    Returns:
        dict with abstention_rate, miscoverage_rate, default rates,
        calibration error on retained, etc.

    Coverage definition for binary classification with {0,1,abstain}
    prediction sets:
        - If set_size == 1: covered iff true label is the singleton
        - If set_size == 2 (abstain): trivially covered (set is {0,1})
        - If set_size == 0 (empty): always miscovered
    """
    df_pred = conformal_prediction_sets(y_prob, tau)
    df_pred['y_true'] = np.asarray(y_true)

    # Coverage: did the prediction set contain the true label?
    df_pred['covered'] = (
        ((df_pred['set_size'] == 2)) |
        ((df_pred['y_true'] == 1) & df_pred['contains_1']) |
        ((df_pred['y_true'] == 0) & df_pred['contains_0'])
    )

    confident = df_pred[df_pred['set_size'] == 1]
    abstained = df_pred[df_pred['set_size'] == 2]
    empty = df_pred[df_pred['set_size'] == 0]

    n_total = len(df_pred)
    coverage_rate = float(df_pred['covered'].mean())
    miscoverage_rate = 1 - coverage_rate
    abstention_rate = len(abstained) / n_total if n_total > 0 else 0
    empty_rate = len(empty) / n_total if n_total > 0 else 0

    cal_full_raw = calculate_calibration_error(
        df_pred['y_true'].values, df_pred['p_default'].values
    )
    cal_full_pp = (cal_full_raw * 100
                   if cal_full_raw is not None and not np.isnan(cal_full_raw)
                   else np.nan)

    if len(confident) > 0:
        cal_ret_raw = calculate_calibration_error(
            confident['y_true'].values, confident['p_default'].values
        )
        cal_ret_pp = (cal_ret_raw * 100
                      if cal_ret_raw is not None and not np.isnan(cal_ret_raw)
                      else np.nan)
    else:
        cal_ret_pp = np.nan

    return {
        'n_total': n_total,
        'n_confident': len(confident),
        'n_abstained': len(abstained),
        'n_empty': len(empty),
        'coverage_rate': round(coverage_rate, 4),
        'miscoverage_rate': round(miscoverage_rate, 4),
        'abstention_rate': round(abstention_rate, 4),
        'empty_set_rate': round(empty_rate, 4),
        'default_rate_full': round(float(df_pred['y_true'].mean()), 4),
        'default_rate_confident': (
            round(float(confident['y_true'].mean()), 4)
            if len(confident) > 0 else np.nan
        ),
        'default_rate_abstained': (
            round(float(abstained['y_true'].mean()), 4)
            if len(abstained) > 0 else np.nan
        ),
        'cal_err_full_pp': round(cal_full_pp, 2)
                           if not np.isnan(cal_full_pp) else np.nan,
        'cal_err_retained_pp': round(cal_ret_pp, 2)
                                if not np.isnan(cal_ret_pp) else np.nan,
    }


def run_aci_through_quarters(
    model, scaler, imputer, feature_list, df_monitor,
    cal_scores, alpha_target, gamma,
    start_year, end_year, end_quarter, target_col,
    label="ACI",
):
    """
    Walk quarter-by-quarter through the stress period applying Adaptive
    Conformal Inference. At each quarter:
      1. Compute current threshold tau from current alpha
      2. Score quarter through model
      3. Form prediction sets, measure miscoverage
      4. Update alpha via Gibbs & Candès (2021) rule:
            alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

    Returns a per-quarter DataFrame with alpha, tau, and full evaluation
    metrics tracking.
    """
    print(f"\n  Running {label} through {start_year}Q1 - {end_year}Q{end_quarter}...")

    alpha_t = alpha_target  # initialise
    rows = []

    for qlabel, year, q, subset in get_quarterly_subsets(
        df_monitor, start_year, end_year, end_quarter
    ):
        # 1. Compute threshold from current alpha
        tau_t = conformal_threshold(cal_scores, alpha_t)

        # 2. Score this quarter
        y_prob = score_period(model, scaler, imputer, feature_list, subset)
        y_true = subset[target_col].values

        # 3. Evaluate
        metrics = evaluate_conformal_quarter(y_prob, y_true, tau_t)

        # 4. ACI update for NEXT quarter
        err_t = metrics['miscoverage_rate']  # batched ACI: use mean miscoverage
        alpha_next = alpha_t + gamma * (alpha_target - err_t)
        # Clip to numerical bounds
        alpha_next = np.clip(alpha_next, ACI_ALPHA_MIN, ACI_ALPHA_MAX)

        row = {
            'quarter': qlabel,
            'year': year,
            'q': q,
            'q_idx': quarter_index(year, q),
            'alpha_t': round(alpha_t, 4),
            'tau_t': round(tau_t, 4),
            'alpha_next': round(alpha_next, 4),
        }
        row.update(metrics)
        rows.append(row)

        alpha_t = alpha_next  # advance for next quarter

    return pd.DataFrame(rows)


def run_vanilla_through_quarters(
    model, scaler, imputer, feature_list, df_monitor,
    cal_scores, alpha_fixed,
    start_year, end_year, end_quarter, target_col,
    label="VANILLA",
):
    """
    Walk quarter-by-quarter applying VANILLA split conformal prediction
    with a fixed alpha throughout (no adaptation). This is the control
    against which ACI is compared.
    """
    print(f"\n  Running {label} (fixed alpha={alpha_fixed:.2f}) through "
          f"{start_year}Q1 - {end_year}Q{end_quarter}...")

    tau_fixed = conformal_threshold(cal_scores, alpha_fixed)
    rows = []

    for qlabel, year, q, subset in get_quarterly_subsets(
        df_monitor, start_year, end_year, end_quarter
    ):
        y_prob = score_period(model, scaler, imputer, feature_list, subset)
        y_true = subset[target_col].values

        metrics = evaluate_conformal_quarter(y_prob, y_true, tau_fixed)

        row = {
            'quarter': qlabel,
            'year': year,
            'q': q,
            'q_idx': quarter_index(year, q),
            'alpha_t': alpha_fixed,
            'tau_t': round(tau_fixed, 4),
            'alpha_next': alpha_fixed,
        }
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# CALIBRATION SET CONSTRUCTION
# ============================================================================

def build_calibration_scores(model, scaler, imputer, feature_list,
                              df_calibration, target_col):
    """
    Score the calibration set and compute nonconformity scores. These
    scores form the empirical distribution from which conformal
    thresholds are extracted.
    """
    print(f"\n  Building calibration set ({len(df_calibration):,} loans)...")
    y_prob_cal = score_period(
        model, scaler, imputer, feature_list, df_calibration
    )
    y_true_cal = df_calibration[target_col].values
    scores = compute_nonconformity_scores(y_true_cal, y_prob_cal)
    print(f"    Nonconformity score distribution:")
    print(f"      min:    {np.min(scores):.4f}")
    print(f"      median: {np.median(scores):.4f}")
    print(f"      max:    {np.max(scores):.4f}")
    return scores


# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_alpha_trajectory(aci_df, vanilla_df, filename):
    """
    HEADLINE 1: how alpha adapts over time. The mechanism made visible.
    Shows ACI's alpha walking quarter-by-quarter, with vanilla's flat
    line as reference.
    """
    quarters = aci_df['quarter'].tolist()
    aci_alpha = aci_df['alpha_t'].tolist()
    vanilla_alpha = vanilla_df['alpha_t'].tolist()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(quarters, aci_alpha, marker='s', linewidth=2.5,
            color=PALETTE['aci'], label='ACI: alpha adapts each quarter',
            markersize=8, markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(quarters, vanilla_alpha, marker='o', linewidth=2,
            color=PALETTE['vanilla'], linestyle='--',
            label='Vanilla conformal: fixed alpha = 0.10',
            markersize=7, alpha=0.8)
    ax.axhline(ALPHA_TARGET, color=PALETTE['target'], linestyle=':',
               linewidth=1.2, alpha=0.6,
               label=f'Target alpha = {ALPHA_TARGET}')

    # Annotate alpha excursions
    for i, (q, a) in enumerate(zip(quarters, aci_alpha)):
        if abs(a - ALPHA_TARGET) > 0.02:  # annotate notable departures
            ax.annotate(f'{a:.3f}', xy=(q, a), xytext=(0, 8),
                        textcoords='offset points', ha='center',
                        fontsize=8, color=PALETTE['aci'])

    ax.set_xlabel('Quarter')
    ax.set_ylabel('Alpha (target miscoverage rate)')
    ax.set_title(
        'Adaptive Conformal Inference: alpha trajectory\n'
        f'ACI dynamically updates alpha quarter by quarter '
        f'(gamma = {ACI_GAMMA}); vanilla holds fixed.',
        fontsize=12
    )
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def plot_coverage_comparison(aci_df, vanilla_df, filename):
    """
    HEADLINE 2: empirical coverage rate. The central theoretical
    guarantee tested. Vanilla coverage should drift away from target
    under regime shift; ACI's should stay close.
    """
    quarters = aci_df['quarter'].tolist()
    aci_cov = (aci_df['coverage_rate'] * 100).tolist()
    van_cov = (vanilla_df['coverage_rate'] * 100).tolist()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(quarters, aci_cov, marker='s', linewidth=2.5,
            color=PALETTE['aci'], label='ACI empirical coverage',
            markersize=8, markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(quarters, van_cov, marker='o', linewidth=2.5,
            color=PALETTE['vanilla'], label='Vanilla conformal empirical coverage',
            markersize=8, markeredgecolor='white', markeredgewidth=1.5)

    target_cov = (1 - ALPHA_TARGET) * 100
    ax.axhline(target_cov, color=PALETTE['target'], linestyle='--',
               linewidth=1.5, alpha=0.7,
               label=f'Target coverage = {target_cov:.0f}%')

    # Shaded acceptable band: target +- 2pp
    ax.fill_between(range(len(quarters)),
                     target_cov - 2, target_cov + 2,
                     alpha=0.10, color=PALETTE['target'],
                     label='Target band (+/- 2pp)')

    ax.set_xlabel('Quarter')
    ax.set_ylabel('Empirical coverage rate (%)')
    ax.set_title(
        'Empirical Coverage: ACI vs Vanilla Conformal\n'
        'The central theoretical claim of ACI - long-run coverage at target '
        'under arbitrary distribution shift',
        fontsize=12
    )
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def plot_calibration_comparison(aci_df, vanilla_df, filename):
    """Per-quarter calibration error on retained (confident) cases."""
    quarters = aci_df['quarter'].tolist()
    aci_cal = aci_df['cal_err_retained_pp'].tolist()
    van_cal = vanilla_df['cal_err_retained_pp'].tolist()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(quarters, aci_cal, marker='s', linewidth=2.5,
            color=PALETTE['aci'], label='ACI (calibration on retained)',
            markersize=7)
    ax.plot(quarters, van_cal, marker='o', linewidth=2.5,
            color=PALETTE['vanilla'], label='Vanilla (calibration on retained)',
            markersize=7)
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Calibration error on retained (pp)')
    ax.set_title(
        'Calibration Error on Auto-Decisioned (Retained) Cases\n'
        'Per quarter, only confident decisions are evaluated',
        fontsize=12
    )
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def plot_abstention_comparison(aci_df, vanilla_df, filename):
    """Per-quarter abstention rate (the operational cost dimension)."""
    quarters = aci_df['quarter'].tolist()
    aci_abst = (aci_df['abstention_rate'] * 100).tolist()
    van_abst = (vanilla_df['abstention_rate'] * 100).tolist()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(quarters, aci_abst, marker='s', linewidth=2.5,
            color=PALETTE['aci'], label='ACI', markersize=7)
    ax.plot(quarters, van_abst, marker='o', linewidth=2.5,
            color=PALETTE['vanilla'], label='Vanilla', markersize=7)
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Abstention rate (%)')
    ax.set_title(
        'Abstention Rate: ACI vs Vanilla\n'
        'The operational cost dimension - higher abstention = more human review',
        fontsize=12
    )
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def plot_layered_defence(aci_a_df, aci_b_df, vanilla_df, filename):
    """
    Layered defence visualisation: ACI on top of Model A vs ACI on top of
    Model B (the PSI-retrained model from Solution 2 fix demo). Shows
    whether the two solutions stack.
    """
    quarters = aci_a_df['quarter'].tolist()

    # Coverage rate panel
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    target_cov = (1 - ALPHA_TARGET) * 100

    ax1.plot(quarters, (aci_a_df['coverage_rate'] * 100).tolist(),
             marker='s', linewidth=2.5, color=PALETTE['aci'],
             label='ACI on Model A (2012-2016 training)', markersize=7)
    ax1.plot(quarters, (aci_b_df['coverage_rate'] * 100).tolist(),
             marker='D', linewidth=2.5, color=PALETTE['layered'],
             label='ACI on Model B (PSI-retrained, 2012-2017)', markersize=7)
    ax1.plot(quarters, (vanilla_df['coverage_rate'] * 100).tolist(),
             marker='o', linewidth=2, color=PALETTE['vanilla'],
             label='Vanilla on Model A (control)', markersize=6,
             alpha=0.7, linestyle='--')
    ax1.axhline(target_cov, color=PALETTE['target'], linestyle=':',
                linewidth=1.2, alpha=0.6, label=f'Target ({target_cov:.0f}%)')
    ax1.set_ylabel('Empirical coverage (%)')
    ax1.set_title('Coverage: layered defence')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(quarters, aci_a_df['cal_err_retained_pp'].tolist(),
             marker='s', linewidth=2.5, color=PALETTE['aci'],
             label='ACI on Model A', markersize=7)
    ax2.plot(quarters, aci_b_df['cal_err_retained_pp'].tolist(),
             marker='D', linewidth=2.5, color=PALETTE['layered'],
             label='ACI on Model B', markersize=7)
    ax2.plot(quarters, vanilla_df['cal_err_retained_pp'].tolist(),
             marker='o', linewidth=2, color=PALETTE['vanilla'],
             label='Vanilla on Model A', markersize=6,
             alpha=0.7, linestyle='--')
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Calibration error (pp)')
    ax2.set_title('Calibration on retained: layered defence')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    fig.suptitle(
        'Layered Defence: ACI alone vs ACI + PSI-Triggered Retraining',
        fontsize=12, y=1.00
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, filename)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def plot_method_card(filename):
    """
    Showcase visualisation: ACI algorithm pseudocode rendered as a
    figure suitable for prose / presentation use. Demonstrates
    technical communication skills (a coursework deliverable).
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis('off')

    # Box backgrounds
    bg1 = patches.FancyBboxPatch((0.02, 0.78), 0.96, 0.18,
                                  boxstyle="round,pad=0.02",
                                  linewidth=1.2,
                                  facecolor='#F0F5FA', edgecolor=PALETTE['aci'])
    ax.add_patch(bg1)

    bg2 = patches.FancyBboxPatch((0.02, 0.18), 0.96, 0.55,
                                  boxstyle="round,pad=0.02",
                                  linewidth=1.2,
                                  facecolor='#FAFAFA', edgecolor='#666666')
    ax.add_patch(bg2)

    bg3 = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.13,
                                  boxstyle="round,pad=0.02",
                                  linewidth=1.2,
                                  facecolor='#E9F5EE', edgecolor=PALETTE['layered'])
    ax.add_patch(bg3)

    ax.text(0.5, 0.93, 'Adaptive Conformal Inference (Gibbs & Candès, 2021)',
            ha='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.86,
            'Online learning rule that maintains long-run coverage at target\n'
            'under arbitrary distribution shift',
            ha='center', fontsize=10, style='italic')

    # Algorithm steps
    steps = [
        ('Inputs:',
         '  - Trained model f, calibration nonconformity scores S\n'
         '  - Target miscoverage alpha_target (e.g., 0.10)\n'
         '  - Learning rate gamma (e.g., 0.05)'),
        ('Initialise:',
         '  alpha_1 := alpha_target'),
        ('At each time step t = 1, 2, ...:',
         '  1. tau_t := finite-sample (1 - alpha_t) quantile of S\n'
         '  2. Form prediction set C_t(x_t) using tau_t\n'
         '  3. Observe y_t and compute err_t = 1 if y_t not in C_t(x_t) else 0\n'
         '  4. Update: alpha_{t+1} := alpha_t + gamma * (alpha_target - err_t)'),
    ]

    y = 0.68
    for header, body in steps:
        ax.text(0.05, y, header, fontsize=11, fontweight='bold',
                color=PALETTE['aci'])
        y -= 0.025
        for line in body.split('\n'):
            ax.text(0.06, y, line, fontsize=9.5, family='monospace')
            y -= 0.022
        y -= 0.005

    ax.text(0.5, 0.10,
            'Theoretical Guarantee:',
            ha='center', fontsize=10, fontweight='bold', color=PALETTE['layered'])
    ax.text(0.5, 0.06,
            '|empirical_miscoverage - alpha_target| <= (alpha + gamma) / (gamma * T)\n'
            'for any distribution shift between train, calibration, and test data',
            ha='center', fontsize=9, family='monospace')

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
    print("  SOLUTION 3: ADAPTIVE CONFORMAL INFERENCE")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    # ----- STEP 1: Load + preprocess data -----
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
    df_monitor = build_monitoring_set(df)

    # ----- STEP 2: Construct training and calibration sets -----
    print(f"\n{'-' * 70}")
    print("  STEP 2: Constructing training and calibration sets")
    print(f"{'-' * 70}")

    # Hold out a fraction of CALIBRATION_YEAR (=2016) loans as calibration set;
    # train on everything else from 2012-2016
    df_2016 = df_train_2012_2016[df_train_2012_2016['issue_year'] == CALIBRATION_YEAR]
    df_2012_2015 = df_train_2012_2016[df_train_2012_2016['issue_year'] != CALIBRATION_YEAR]

    rng = np.random.RandomState(RANDOM_STATE)
    cal_idx = rng.choice(
        df_2016.index,
        size=int(len(df_2016) * CALIBRATION_FRACTION),
        replace=False,
    )
    df_calibration = df_2016.loc[cal_idx]
    df_2016_train_part = df_2016.drop(index=cal_idx)
    df_train_a = pd.concat([df_2012_2015, df_2016_train_part], ignore_index=True)

    print(f"\n  Model A training:    {len(df_train_a):,} loans (2012-2016 minus calibration holdout)")
    print(f"  Calibration set:     {len(df_calibration):,} loans (2016 holdout)")

    # Model B training set (2012-2017) for the layered defence test
    df_train_b = df[
        (df['issue_year'] >= 2012) & (df['issue_year'] <= 2017)
    ].copy()
    # Remove calibration loans from Model B training as well
    df_train_b = df_train_b[~df_train_b.index.isin(df_calibration.index)]
    print(f"  Model B training:    {len(df_train_b):,} loans (2012-2017 minus calibration holdout)")

    # ----- STEP 3: Train Model A (the conformal-wrapped model) -----
    print(f"\n{'-' * 70}")
    print("  STEP 3: Training MODEL A (2012-2016)")
    print(f"{'-' * 70}")
    X_train_a, y_train_a, feat_a, scaler_a, imputer_a = (
        prepare_features_for_training(df_train_a, ML_FEATURES_CORRECTED, TARGET)
    )
    model_a = train_neural_net(X_train_a, y_train_a, label="MODEL A")

    # ----- STEP 4: Build calibration nonconformity scores via Model A -----
    print(f"\n{'-' * 70}")
    print("  STEP 4: Building calibration nonconformity scores")
    print(f"{'-' * 70}")
    cal_scores_a = build_calibration_scores(
        model_a, scaler_a, imputer_a, feat_a, df_calibration, TARGET
    )

    # ----- STEP 5: Run vanilla conformal as control -----
    print(f"\n{'-' * 70}")
    print("  STEP 5: Vanilla conformal (control, fixed alpha)")
    print(f"{'-' * 70}")
    vanilla_results = run_vanilla_through_quarters(
        model_a, scaler_a, imputer_a, feat_a, df_monitor,
        cal_scores_a, ALPHA_TARGET,
        STRESS_EVAL_START_YEAR, STRESS_EVAL_END_YEAR, STRESS_EVAL_END_QUARTER,
        EARLY_DQ_TARGET, label="VANILLA on Model A",
    )

    # ----- STEP 6: Run ACI as fix -----
    print(f"\n{'-' * 70}")
    print("  STEP 6: Adaptive Conformal Inference")
    print(f"{'-' * 70}")
    aci_a_results = run_aci_through_quarters(
        model_a, scaler_a, imputer_a, feat_a, df_monitor,
        cal_scores_a, ALPHA_TARGET, ACI_GAMMA,
        STRESS_EVAL_START_YEAR, STRESS_EVAL_END_YEAR, STRESS_EVAL_END_QUARTER,
        EARLY_DQ_TARGET, label="ACI on Model A",
    )

    # ----- STEP 7 (optional): Layered defence - ACI on Model B -----
    if INCLUDE_LAYERED_DEFENCE:
        print(f"\n{'-' * 70}")
        print("  STEP 7: Layered defence - train Model B and apply ACI")
        print(f"{'-' * 70}")

        # Train Model B (PSI-retrained equivalent: 2012-2017)
        X_train_b, y_train_b, feat_b, scaler_b, imputer_b = (
            prepare_features_for_training(df_train_b, ML_FEATURES_CORRECTED, TARGET)
        )
        model_b = train_neural_net(X_train_b, y_train_b, label="MODEL B (2012-2017)")

        # Calibration nonconformity for Model B (using SAME calibration loans)
        cal_scores_b = build_calibration_scores(
            model_b, scaler_b, imputer_b, feat_b, df_calibration, TARGET
        )

        aci_b_results = run_aci_through_quarters(
            model_b, scaler_b, imputer_b, feat_b, df_monitor,
            cal_scores_b, ALPHA_TARGET, ACI_GAMMA,
            STRESS_EVAL_START_YEAR, STRESS_EVAL_END_YEAR, STRESS_EVAL_END_QUARTER,
            EARLY_DQ_TARGET, label="ACI on Model B",
        )
    else:
        aci_b_results = None

    # ----- STEP 8: Save tables -----
    print(f"\n{'-' * 70}")
    print("  STEP 8: Saving per-quarter results")
    print(f"{'-' * 70}")

    vanilla_results['config'] = 'vanilla_modelA'
    aci_a_results['config'] = 'aci_modelA'
    if aci_b_results is not None:
        aci_b_results['config'] = 'aci_modelB_layered'
        all_results = pd.concat(
            [vanilla_results, aci_a_results, aci_b_results],
            axis=0, ignore_index=True,
        )
    else:
        all_results = pd.concat(
            [vanilla_results, aci_a_results], axis=0, ignore_index=True
        )

    all_results.to_csv(
        os.path.join(TABLES_DIR, 'aci_quarterly_results.csv'), index=False
    )

    # ----- STEP 9: Build summary table -----
    def summarise(df_, label):
        return {
            'config': label,
            'mean_coverage_rate': round(float(df_['coverage_rate'].mean()), 4),
            'coverage_deviation_from_target_pp':
                round(abs(df_['coverage_rate'].mean() - (1 - ALPHA_TARGET)) * 100, 2),
            'mean_abstention_rate': round(float(df_['abstention_rate'].mean()), 4),
            'mean_cal_err_retained_pp':
                round(float(df_['cal_err_retained_pp'].mean()), 2),
            'max_cal_err_retained_pp':
                round(float(df_['cal_err_retained_pp'].max()), 2),
            'mean_cal_err_full_pp':
                round(float(df_['cal_err_full_pp'].mean()), 2),
        }

    summary_rows = [
        summarise(vanilla_results, 'Vanilla on Model A (control)'),
        summarise(aci_a_results, 'ACI on Model A (FIX)'),
    ]
    if aci_b_results is not None:
        summary_rows.append(
            summarise(aci_b_results, 'ACI on Model B (LAYERED DEFENCE)')
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(TABLES_DIR, 'aci_summary_comparison.csv'), index=False
    )

    # Print summary
    print(f"\n  Quarterly trajectory (vanilla vs ACI):")
    print(f"  {'Quarter':<10}{'Van cov':>10}{'ACI cov':>10}"
          f"{'Van abst':>11}{'ACI abst':>11}{'ACI alpha':>11}")
    print(f"  {'-' * 64}")
    for i in range(len(vanilla_results)):
        v = vanilla_results.iloc[i]
        a = aci_a_results.iloc[i]
        print(f"  {v['quarter']:<10}"
              f"{v['coverage_rate']*100:>9.1f}%"
              f"{a['coverage_rate']*100:>9.1f}%"
              f"{v['abstention_rate']*100:>10.1f}%"
              f"{a['abstention_rate']*100:>10.1f}%"
              f"{a['alpha_t']:>11.4f}")

    # ----- STEP 10: Visualisations -----
    print(f"\n{'-' * 70}")
    print("  STEP 10: Generating visualisations")
    print(f"{'-' * 70}")
    plot_alpha_trajectory(aci_a_results, vanilla_results,
                           'aci_alpha_trajectory.png')
    plot_coverage_comparison(aci_a_results, vanilla_results,
                              'aci_vs_vanilla_coverage.png')
    plot_calibration_comparison(aci_a_results, vanilla_results,
                                 'aci_vs_vanilla_calibration.png')
    plot_abstention_comparison(aci_a_results, vanilla_results,
                                'aci_vs_vanilla_abstention.png')
    if aci_b_results is not None:
        plot_layered_defence(aci_a_results, aci_b_results, vanilla_results,
                              'layered_defence.png')
    plot_method_card('method_card.png')

    # ----- FINAL SUMMARY -----
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 80}")
    print("  SUMMARY: SOLUTION 3 RESULTS (Adaptive Conformal Inference)")
    print(f"{'=' * 80}")

    target_cov_pct = (1 - ALPHA_TARGET) * 100
    van_mean_cov = vanilla_results['coverage_rate'].mean() * 100
    aci_mean_cov = aci_a_results['coverage_rate'].mean() * 100
    van_dev = abs(van_mean_cov - target_cov_pct)
    aci_dev = abs(aci_mean_cov - target_cov_pct)

    print(f"\n  Target coverage: {target_cov_pct:.0f}%")
    print(f"\n  Mean empirical coverage (2018Q1 - 2020Q3):")
    print(f"    Vanilla:   {van_mean_cov:>5.2f}%   (dev from target: {van_dev:.2f}pp)")
    print(f"    ACI:       {aci_mean_cov:>5.2f}%   (dev from target: {aci_dev:.2f}pp)")

    if van_dev > 0:
        rel_imp = 100 * (van_dev - aci_dev) / van_dev
        print(f"    ACI reduces deviation from target coverage by {rel_imp:+.1f}%")

    print(f"\n  Mean abstention rate:")
    print(f"    Vanilla:   {vanilla_results['abstention_rate'].mean()*100:>5.2f}%")
    print(f"    ACI:       {aci_a_results['abstention_rate'].mean()*100:>5.2f}%")

    print(f"\n  Mean calibration error on retained (confident decisions):")
    print(f"    Vanilla:   {vanilla_results['cal_err_retained_pp'].mean():>5.2f}pp")
    print(f"    ACI:       {aci_a_results['cal_err_retained_pp'].mean():>5.2f}pp")

    if aci_b_results is not None:
        print(f"\n  Layered defence (ACI on Model B = ACI + PSI-triggered retraining):")
        print(f"    Mean coverage:     {aci_b_results['coverage_rate'].mean()*100:>5.2f}%")
        print(f"    Mean abstention:   {aci_b_results['abstention_rate'].mean()*100:>5.2f}%")
        print(f"    Mean cal_err ret:  {aci_b_results['cal_err_retained_pp'].mean():>5.2f}pp")

    print(f"\n  ACI alpha trajectory:")
    print(f"    Initial:   {aci_a_results.iloc[0]['alpha_t']:.4f}")
    print(f"    Final:     {aci_a_results.iloc[-1]['alpha_t']:.4f}")
    print(f"    Min:       {aci_a_results['alpha_t'].min():.4f}")
    print(f"    Max:       {aci_a_results['alpha_t'].max():.4f}")

    print(f"\n  Outputs:")
    print(f"    Tables:  {TABLES_DIR}/")
    print(f"    Figures: {FIGURES_DIR}/")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
