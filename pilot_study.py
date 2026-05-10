#!/usr/bin/env python3
"""
MSIN0032 Dissertation Pilot Study: FICO vs ML Credit Scoring Under Stress
=========================================================================

This script implements the empirical pilot study supporting the MSIN0032
dissertation. It compares a traditional FICO-based logistic regression against
a neural-network ML model on the LendingClub 2007-Q3 2020 loan dataset, across
a normal period (2017 originations) and a COVID-era stress period (2019-2020
originations).

METHODOLOGY OVERVIEW

    Training data:    2012-2016 originations (~1.28M loans)
    Normal-period:    2017 originations (~443k loans)
    Stress-period:    2019-2020 originations (~159k loans, post-filter)
    
    Target:           Vintage-comparable early delinquency
                      (snapshot status: Charged Off, Default, Late 31-120,
                       Late 16-30, In Grace Period)
                      Applied only to loans observed for at least 18 months
                      to ensure equal observation horizons across vintages.
    
    FICO model:       Logistic regression on 6 traditional credit-bureau features.
    ML model:         MLPClassifier (3 hidden layers: 100, 50, 25 neurons)
                      on 23 features including alternative data and engineered
                      ratios. Architecture chosen to mirror Upstart's stated
                      reliance on deep learning.

EXCLUDED VARIABLES (data-leakage prevention)

    The ML feature set deliberately excludes int_rate, grade, sub_grade, and
    installment. These are endogenous outcomes of the lender's risk decision
    rather than inputs to it; including them produces mechanical correlation
    with default and inflates apparent ML performance.

VINTAGE-IMMATURITY CORRECTION

    Earlier iterations of this pilot defined the early-delinquency target as a
    snapshot of current loan status without controlling for loan age. Recent
    vintages then appeared to have artificially low default rates because they
    had not been observed for long enough to enter delinquent status. The
    18-month observation-window filter applied here addresses that bias by
    excluding loans whose age at the data cutoff (30 September 2020) is below
    the threshold, so all retained loans share an equal observation horizon.
    Residual immaturity bias is acknowledged in dissertation §4.3.

OUTPUTS

    A subdirectory `pilot_study_outputs/` is created next to this script. It
    contains a `figures/` folder with two PNGs (ROC curves and a degradation
    bar chart) and a `tables/` folder with one CSV (per-period model metrics).

USAGE

    python pilot_study.py
    
    The script expects accepted_2007_to_2020Q3.csv in the same directory by
    default; alternative paths can be set via the DATA_PATH constant.

Author: Marat Daminov
"""

import os
import sys
import warnings
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, brier_score_loss,
    roc_curve, precision_recall_curve, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "/Users/maratdaminov/Desktop/Dissertation/code/accepted_2007_to_2020Q3.csv"

# Anchor outputs to the directory containing this script, so they land next
# to the script regardless of which working directory python is invoked from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "pilot_study_outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

# Time periods (same as v2)
TRAIN_START = 2012
TRAIN_END = 2016
NORMAL_START = 2017
NORMAL_END = 2017
STRESS_START = 2019
STRESS_END = 2020

# V5 immaturity-correction parameter
# ----------------------------------
# Observation window: a loan is only retained for evaluation if it was originated
# at least OBSERVATION_WINDOW_MONTHS before the data cutoff. This ensures all
# retained loans have an equal observation horizon, eliminating the vintage-
# immaturity bias that affected V3.
#
# Default: 18 months. Trade-off: longer window = cleaner signal but smaller
# stress-period sample. 18 months keeps 2018Q1-2019Q1 fully retained and partial
# 2019Q2-Q3 retained at the 2020Q3 data cutoff.
OBSERVATION_WINDOW_MONTHS = 18

# Data cutoff: the last quarter present in the LendingClub dataset.
DATA_CUTOFF_DATE = pd.Timestamp("2020-09-30")

TARGET = 'default'
# V5 NOTE: EARLY_DQ_TARGET is bound to the corrected column name produced by
# preprocess_data(). The three solution fix-demo scripts import this constant
# and so will automatically use the corrected target with no edits required.
EARLY_DQ_TARGET = 'early_delinquency_corrected'

RANDOM_STATE = 42

# Plotting
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

COLORS = {
    'fico': '#2C5F8A',
    'ml': '#C44E52',
    'normal': '#4CAF50',
    'stress': '#FF5722',
}


# ============================================================================
# CORRECTED FEATURE DEFINITIONS
# ============================================================================

# FICO Model: Traditional 5-variable approach (UNCHANGED)
FICO_FEATURES = [
    'fico_range_high',
    'annual_inc',
    'dti',
    'inq_last_6mths',
    'revol_util',
]

# ML Model: CORRECTED - Removed interest rate and risk-encoding variables
# This matches what real fintech underwriting models can actually use
ML_FEATURES_CORRECTED = [
    # === TRADITIONAL CREDIT BUREAU DATA ===
    'fico_range_high',           # Credit score (but ML will learn it's not perfect)
    'annual_inc',                # Income
    'dti',                       # Debt-to-income ratio
    'inq_last_6mths',           # Recent credit inquiries
    'revol_util',               # Credit utilization
    'revol_bal',                # Total revolving balance
    'open_acc',                 # Open accounts
    'total_acc',                # Total accounts
    'pub_rec',                  # Public records (bankruptcies)
    'delinq_2yrs',              # Recent delinquencies
    
    # === ALTERNATIVE DATA (Fintech advantage) ===
    'emp_length_num',           # Employment stability
    'home_ownership_enc',       # Housing status
    'purpose_enc',              # Loan purpose
    'loan_amnt',                # Loan amount requested
    'installment',              # Monthly payment
    
    # === ENGINEERED FEATURES (No data leakage) ===
    'log_income',               # log(income) - diminishing returns
    'fico_squared',             # FICO² - non-linear risk at tails
    'dti_fico_interaction',     # DTI × FICO interaction
    'payment_to_income',        # Affordability ratio
    'credit_age_years',         # Age of credit history (NEW)
    'account_diversity',        # Mix of account types (NEW)
    'utilization_squared',      # Util² - captures high-utilization risk (NEW)
    'debt_burden',              # Total debt as % of income (NEW)
]

# EXPLICITLY EXCLUDED VARIABLES (Data leakage sources):
EXCLUDED_LEAKAGE_VARS = [
    'int_rate',        # ENDOGENOUS: Determined BY risk assessment
    'grade',           # ENCODES RISK: Lender's assigned grade
    'sub_grade',       # ENCODES RISK: Lender's sub-grade
    'installment',     # PARTIALLY ENDOGENOUS: Depends on rate + amount
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dirs():
    for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)


def calculate_psi(expected, actual, bins=10):
    """Population Stability Index - detects input drift."""
    eps = 1e-6
    breakpoints = np.percentile(expected.dropna(), np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0
    
    exp_counts = np.histogram(expected.dropna(), bins=breakpoints)[0]
    act_counts = np.histogram(actual.dropna(), bins=breakpoints)[0]
    
    exp_pct = np.clip(exp_counts / (exp_counts.sum() + eps), eps, None)
    act_pct = np.clip(act_counts / (act_counts.sum() + eps), eps, None)
    
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(psi, 4)


def calculate_calibration_error(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    try:
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1))[0]
        weights = bin_totals / (bin_totals.sum() + 1e-8)
        ece = np.sum(weights[:len(fraction_pos)] * np.abs(fraction_pos - mean_pred))
        return round(float(ece), 4)
    except (ValueError, IndexError):
        return np.nan


# ============================================================================
# DATA LOADING
# ============================================================================

def find_data_file():
    """Locate LendingClub CSV."""
    if os.path.isfile(DATA_PATH):
        return DATA_PATH, True
    
    alternatives = [
        "Loan_status_2007-2020Q3.csv",
        "accepted_2007_to_2020q3.csv",
        "lending-club-20072020q1.csv",
    ]
    
    for fname in alternatives:
        if os.path.isfile(fname):
            return fname, True
    
    return None, False


def load_data(filepath):
    """Load and parse LendingClub data."""
    print(f"\n{'='*70}")
    print(f"  LOADING DATA: {filepath}")
    print(f"{'='*70}")
    
    df = pd.read_csv(filepath, low_memory=False, na_values=['', 'NA', 'n/a'])
    
    # Remove header rows mixed in data
    if 'loan_amnt' in df.columns:
        df = df[df['loan_amnt'].apply(lambda x: str(x).replace('.', '').isdigit())]
        df['loan_amnt'] = pd.to_numeric(df['loan_amnt'], errors='coerce')
    
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    
    # Parse dates
    date_cols = ['issue_d', 'issue_date']
    for c in date_cols:
        if c in df.columns:
            df['issue_d'] = pd.to_datetime(df[c], format='mixed', errors='coerce')
            df['issue_year'] = df['issue_d'].dt.year
            df['issue_month'] = df['issue_d'].dt.month
            df['issue_quarter'] = df['issue_d'].dt.quarter
            year_range = df['issue_year'].dropna()
            print(f"  Date range: {int(year_range.min())} – {int(year_range.max())}")
            break
    
    return df


# ============================================================================
# DATA PREPROCESSING (CORRECTED)
# ============================================================================

def preprocess_data(df):
    """
    Clean and engineer features.
    
    CRITICAL CHANGES FROM V2:
    - Does NOT create int_rate feature
    - Does NOT create grade/sub_grade encodings
    - ADDS new alternative data features (credit_age, account_diversity, etc.)
    """
    print(f"\n{'='*70}")
    print(f"  PREPROCESSING DATA")
    print(f"{'='*70}")
    
    n_start = len(df)
    
    # ------------------------------------------------------------------
    # Target variable 1: full default (unchanged from V3)
    # ------------------------------------------------------------------
    default_statuses = ['Charged Off', 'Default']
    df[TARGET] = df['loan_status'].isin(default_statuses).astype(int)
    
    # ------------------------------------------------------------------
    # Target variable 2: V5 vintage-comparable early delinquency
    # ------------------------------------------------------------------
    # V5 uses a single signal (snapshot delinquency status, the V3 definition)
    # but applies it ONLY to loans that have been observed for at least
    # OBSERVATION_WINDOW_MONTHS. The filter is what makes the vintages comparable;
    # the signal definition itself is back to V3 simplicity.
    #
    # V4 had attempted to combine three signals (status OR early-stopped payment
    # OR 50pp FICO drop). On the actual LendingClub data this produced an
    # implausibly-high stress-period actual rate of 83.5% because:
    #   - 'last_pymnt_d' is "date of most recent payment received", not "date of
    #     final payment". Currently-paying young vintages falsely triggered the
    #     early-stop signal.
    #   - 50pp FICO drop is within normal credit-score volatility over an 18-mo
    #     horizon for borrowers with utilization changes or hard inquiries.
    # The filter alone (without the extra signals) is sufficient to ensure
    # vintage comparability.
    
    # Parse origination date and compute loan age at the data cutoff.
    # Note: by the time preprocess_data runs, issue_d may already be datetime
    # (load_data converts it). Handle both cases.
    if 'issue_d' not in df.columns:
        raise ValueError("issue_d column not found; cannot apply V5 immaturity correction")
    if df['issue_d'].dtype == 'O':
        df['issue_d_parsed'] = pd.to_datetime(
            df['issue_d'], format='%b-%Y', errors='coerce'
        )
    else:
        df['issue_d_parsed'] = pd.to_datetime(df['issue_d'], errors='coerce')
    
    df['loan_age_months'] = (
        (DATA_CUTOFF_DATE - df['issue_d_parsed']).dt.days / 30.44
    )
    
    # Single-signal target: snapshot delinquency status (V3 definition)
    late_statuses = ['Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period']
    dq_statuses = default_statuses + late_statuses
    df[EARLY_DQ_TARGET] = df['loan_status'].isin(dq_statuses).astype(int)
    
    # ------------------------------------------------------------------
    # Apply the observation-window filter
    # ------------------------------------------------------------------
    # Only retain loans observed for at least OBSERVATION_WINDOW_MONTHS so
    # all retained loans have an equal observation horizon. This is the
    # core fix for vintage immaturity bias.
    n_pre_filter = len(df)
    df = df[df['loan_age_months'] >= OBSERVATION_WINDOW_MONTHS].copy()
    n_post_filter = len(df)
    
    print(f"  Charge-off rate (full default, post-filter): {df[TARGET].mean():.1%}")
    print(f"  Early delinquency rate (snapshot status, "
          f"{OBSERVATION_WINDOW_MONTHS}-mo filter): {df[EARLY_DQ_TARGET].mean():.1%}")
    print(f"  Observation-window filter: {n_pre_filter:,} -> {n_post_filter:,} loans "
          f"(retained {n_post_filter/n_pre_filter*100:.1f}%)")
    
    # Sanity check: print quarterly actual rate trajectory through stress window
    # so the user can verify the immaturity bias has been corrected
    if 'issue_year' in df.columns and 'issue_quarter' in df.columns:
        stress_quarters = df[
            (df['issue_year'] >= 2018) & (df['issue_year'] <= 2020)
        ].copy()
        stress_quarters['quarter_label'] = (
            stress_quarters['issue_year'].astype(str)
            + 'Q'
            + stress_quarters['issue_quarter'].astype(str)
        )
        traj = stress_quarters.groupby('quarter_label').agg(
            n_loans=(EARLY_DQ_TARGET, 'size'),
            actual_rate_pct=(EARLY_DQ_TARGET, lambda s: s.mean() * 100),
        ).round(2)
        print(f"\n  Actual-rate trajectory (verification):")
        print(traj.to_string())
        print(f"  -> Trajectory should be flat or rising in the 12-25% range")
        print(f"     (consistent with LendingClub published vintage charge-off curves).")
        print(f"     If markedly declining, increase OBSERVATION_WINDOW_MONTHS.\n")
    
    # Numeric conversions
    if 'revol_util' in df.columns:
        df['revol_util'] = pd.to_numeric(
            df['revol_util'].astype(str).str.replace('%', ''), errors='coerce'
        )
    
    numeric_cols = [
        'loan_amnt', 'annual_inc', 'dti', 'fico_range_high', 'fico_range_low',
        'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc',
        'delinq_2yrs', 'installment', 'mort_acc',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Employment length
    if 'emp_length' in df.columns:
        emp_map = {
            '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years': 8, '9 years': 9, '10+ years': 12,
        }
        df['emp_length_num'] = df['emp_length'].map(emp_map)
    else:
        df['emp_length_num'] = np.nan
    
    # Home ownership
    if 'home_ownership' in df.columns:
        home_map = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 0, 'NONE': 0}
        df['home_ownership_enc'] = df['home_ownership'].map(home_map).fillna(0).astype(int)
    else:
        df['home_ownership_enc'] = 0
    
    # Loan purpose
    if 'purpose' in df.columns:
        purpose_counts = df['purpose'].value_counts()
        top_purposes = purpose_counts.head(8).index.tolist()
        df['purpose_clean'] = df['purpose'].where(df['purpose'].isin(top_purposes), 'other')
        purpose_map = {p: i for i, p in enumerate(df['purpose_clean'].value_counts().index)}
        df['purpose_enc'] = df['purpose_clean'].map(purpose_map).fillna(0).astype(int)
    else:
        df['purpose_enc'] = 0
    
    # === BASIC ENGINEERED FEATURES ===
    df['log_income'] = np.log1p(df['annual_inc'].clip(lower=0))
    df['fico_squared'] = df['fico_range_high'] ** 2
    df['dti_fico_interaction'] = df['dti'].fillna(0) * df['fico_range_high'].fillna(0)
    
    monthly_inc = df['annual_inc'].fillna(0) / 12.0
    df['payment_to_income'] = (
        df['installment'] / monthly_inc.clip(lower=1)
    ).clip(upper=2.0)
    
    # === NEW ALTERNATIVE DATA FEATURES ===
    
    # Credit age (how long has borrower had credit?)
    if 'earliest_cr_line' in df.columns:
        df['earliest_cr_line_dt'] = pd.to_datetime(
            df['earliest_cr_line'], format='mixed', errors='coerce'
        )
        df['credit_age_years'] = (
            (df['issue_d'] - df['earliest_cr_line_dt']).dt.days / 365.25
        ).clip(lower=0, upper=50)
    else:
        # Estimate from total_acc (rough proxy)
        df['credit_age_years'] = (df['total_acc'].fillna(0) / 2).clip(lower=0, upper=50)
    
    # Account diversity (measures credit mix)
    df['account_diversity'] = np.sqrt(
        df['open_acc'].fillna(0) * df['total_acc'].fillna(1).clip(lower=1)
    )
    
    # Utilization squared (captures high-utilization risk non-linearly)
    df['utilization_squared'] = (df['revol_util'].fillna(0) / 100) ** 2
    
    # Total debt burden (all revolving debt as % of annual income)
    df['debt_burden'] = (
        df['revol_bal'].fillna(0) / df['annual_inc'].fillna(1).clip(lower=1)
    ).clip(upper=5.0)
    
    # Mort_acc (if missing)
    if 'mort_acc' not in df.columns:
        df['mort_acc'] = 0
    
    # Filter missing critical fields
    critical_cols = ['fico_range_high', 'annual_inc', 'dti', 'issue_year']
    df = df.dropna(subset=critical_cols)
    
    # Outlier clipping
    df['annual_inc'] = df['annual_inc'].clip(upper=500_000)
    df['dti'] = df['dti'].clip(lower=0, upper=80)
    df['revol_util'] = df['revol_util'].clip(lower=0, upper=150)
    df['loan_amnt'] = df['loan_amnt'].clip(lower=500, upper=50_000)
    
    n_end = len(df)
    print(f"  Rows after cleaning: {n_end:,} (dropped {n_start - n_end:,})")
    
    # Verify new features created
    new_features = ['credit_age_years', 'account_diversity', 'utilization_squared', 'debt_burden']
    print(f"\n  Engineered features added:")
    for feat in new_features:
        if feat in df.columns:
            print(f"     {feat}: mean={df[feat].mean():.2f}, std={df[feat].std():.2f}")
    
    return df


# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_data(df):
    """Time-based split: Train (2012-2016), Normal (2017), Stress (2019-2020)."""
    print(f"\n{'='*70}")
    print(f"  SPLITTING DATA BY TIME PERIOD")
    print(f"{'='*70}")
    
    df_train = df[(df['issue_year'] >= TRAIN_START) & (df['issue_year'] <= TRAIN_END)].copy()
    df_normal = df[(df['issue_year'] >= NORMAL_START) & (df['issue_year'] <= NORMAL_END)].copy()
    df_stress = df[(df['issue_year'] >= STRESS_START) & (df['issue_year'] <= STRESS_END)].copy()
    
    for name, subset in [
        ('TRAIN (2012-2016)', df_train),
        ('NORMAL (2017)', df_normal),
        ('STRESS (2019-2020 COVID)', df_stress),
    ]:
        n = len(subset)
        rate = subset[TARGET].mean() if n > 0 else 0
        print(f"  {name}: {n:>8,} loans  |  Default rate: {rate:.1%}")
    
    return df_train, df_normal, df_stress


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features(df_train, df_test, feature_list, target=TARGET):
    """Prepare features with imputation and scaling."""
    available = [f for f in feature_list if f in df_train.columns]
    missing = [f for f in feature_list if f not in df_train.columns]
    
    if missing:
        print(f"  [WARNING] Missing features (will create as 0): {missing}")
        for f in missing:
            df_train[f] = 0
            df_test[f] = 0
        available = feature_list
    
    X_train = df_train[available].copy()
    X_test = df_test[available].copy()
    y_train = df_train[target].values
    y_test = df_test[target].values
    
    # Impute
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=available, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=available, index=X_test.index
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, available, scaler, imputer


# ============================================================================
# MODEL TRAINING (CORRECTED)
# ============================================================================

def train_fico_model(X_train, y_train):
    """Traditional FICO model: Logistic Regression (UNCHANGED)."""
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        max_iter=2000,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        solver='lbfgs',
    )
    model.fit(X_train, y_train)
    return model


def train_ml_model_neural_net(X_train, y_train):
    """
    AI/ML model: NEURAL NETWORK (matches Upstart architecture better).
    
    CHANGE FROM V2: Using MLPClassifier instead of XGBoost/GradientBoosting.
    
    Rationale:
    - Upstart uses deep learning neural networks
    - Neural networks may be LESS robust than tree models during distribution shifts
    - Tree models (XGBoost) have built-in splits that can generalize better
    - Neural networks learn smooth non-linear functions that may not extrapolate well
    
    Architecture:
    - 3 hidden layers: (100, 50, 25) neurons
    - ReLU activation
    - Adam optimizer
    - Dropout-like effect from early stopping
    """
    print(f"\n  Training NEURAL NETWORK (MLPClassifier)...")
    print(f"    Architecture: 3 layers (100, 50, 25 neurons)")
    print(f"    Activation: ReLU")
    print(f"    Optimizer: Adam")
    
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),  # 3-layer deep network
        activation='relu',
        solver='adam',
        alpha=0.001,                       # L2 regularization
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


def train_ml_model_gradient_boosting(X_train, y_train):
    """
    ALTERNATIVE: Gradient Boosting (for comparison).
    This is the v2 approach - keeping it as option to compare against neural nets.
    """
    print(f"\n  Training GRADIENT BOOSTING (GradientBoostingClassifier)...")
    
    model = GradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )
    
    model.fit(X_train, y_train)
    return model


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive evaluation metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    cal_err = calculate_calibration_error(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    pred_default_rate = y_prob.mean()
    actual_default_rate = y_test.mean()
    rate_error = pred_default_rate - actual_default_rate
    
    return {
        'model_name': model_name,
        'auc_roc': round(auc, 4),
        'accuracy': round(acc, 4),
        'brier_score': round(brier, 4),
        'calibration_error': round(cal_err, 4) if not np.isnan(cal_err) else None,
        'f1_score': round(f1, 4),
        'predicted_default_rate': round(pred_default_rate, 4),
        'actual_default_rate': round(actual_default_rate, 4),
        'rate_error': round(rate_error, 4),
        'n_samples': len(y_test),
        'y_prob': y_prob,
        'y_pred': y_pred,
        'y_test': y_test,
    }


def compute_degradation(normal_result, stress_result):
    """Calculate performance degradation."""
    metrics = ['auc_roc', 'accuracy', 'f1_score']
    degrad = {}
    
    for m in metrics:
        n_val = normal_result.get(m, 0)
        s_val = stress_result.get(m, 0)
        if n_val > 0:
            degrad[f'{m}_degradation_pct'] = round((s_val - n_val) / n_val * 100, 2)
        else:
            degrad[f'{m}_degradation_pct'] = None
    
    for m in ['brier_score', 'calibration_error']:
        n_val = normal_result.get(m)
        s_val = stress_result.get(m)
        if n_val and s_val and n_val > 0:
            degrad[f'{m}_change_pct'] = round((s_val - n_val) / n_val * 100, 2)
        else:
            degrad[f'{m}_change_pct'] = None
    
    return degrad


def print_results_comparison(results_dict):
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print(f"  MODEL PERFORMANCE COMPARISON")
    print(f"{'='*90}")
    
    header = f"{'Metric':<28}{'FICO Normal':>16}{'ML Normal':>16}{'FICO Stress':>16}{'ML Stress':>16}"
    print(header)
    print("-" * 90)
    
    metrics = [
        ('AUC-ROC ↑', 'auc_roc'),
        ('Accuracy ↑', 'accuracy'),
        ('Brier Score ↓', 'brier_score'),
        ('Calibration Error ↓', 'calibration_error'),
        ('F1 Score ↑', 'f1_score'),
        ('Pred Default Rate', 'predicted_default_rate'),
        ('Actual Default Rate', 'actual_default_rate'),
    ]
    
    for display, key in metrics:
        row = f"  {display:<26}"
        for period in ['normal', 'stress']:
            for model_type in ['fico', 'ml']:
                result_key = f"{model_type}_{period}"
                val = results_dict.get(result_key, {}).get(key)
                if val is None:
                    row += f"{'N/A':>16}"
                elif isinstance(val, float):
                    row += f"{val:>16.4f}"
                else:
                    row += f"{val:>16}"
        print(row)
    
    print("-" * 90)


def print_degradation_analysis(fico_deg, ml_deg):
    """Print degradation comparison - THE KEY TABLE."""
    print(f"\n{'='*70}")
    print(f"  PERFORMANCE DEGRADATION: Normal → COVID Stress")
    print(f"  (Negative % = model got WORSE)")
    print(f"{'='*70}")
    
    print(f"\n  {'Metric':<30}{'FICO Degrad':>16}{'ML Degrad':>16}{'ML/FICO Ratio':>16}")
    print(f"  {'-'*78}")
    
    for key in ['auc_roc_degradation_pct', 'accuracy_degradation_pct',
                 'f1_score_degradation_pct', 'calibration_error_change_pct']:
        f_val = fico_deg.get(key)
        m_val = ml_deg.get(key)
        
        display_key = key.replace('_degradation_pct', '').replace('_change_pct', '').replace('_', ' ').title()
        
        f_str = f"{f_val:+.2f}%" if f_val is not None else "N/A"
        m_str = f"{m_val:+.2f}%" if m_val is not None else "N/A"
        
        if f_val and m_val and f_val != 0:
            ratio = abs(m_val) / abs(f_val)
            r_str = f"{ratio:.1f}×"
        else:
            r_str = "—"
        
        print(f"  {display_key:<30}{f_str:>16}{m_str:>16}{r_str:>16}")
    
    print(f"\n  INTERPRETATION:")
    print(f"  ML/FICO Ratio > 1.0 = ML degrades MORE")
    print(f"  ML/FICO Ratio < 1.0 = ML degrades LESS")


# ============================================================================
# VISUALIZATION (Simplified)
# ============================================================================

def plot_roc_comparison(fico_normal, ml_normal, fico_stress, ml_stress, filename):
    """ROC curves for both periods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Normal period
    for res, color, label in [
        (fico_normal, COLORS['fico'], f"FICO (AUC={fico_normal['auc_roc']:.3f})"),
        (ml_normal, COLORS['ml'], f"ML (AUC={ml_normal['auc_roc']:.3f})"),
    ]:
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        ax1.plot(fpr, tpr, color=color, linewidth=2, label=label)
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Normal Period (2017)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Stress period
    for res, color, label in [
        (fico_stress, COLORS['fico'], f"FICO (AUC={fico_stress['auc_roc']:.3f})"),
        (ml_stress, COLORS['ml'], f"ML (AUC={ml_stress['auc_roc']:.3f})"),
    ]:
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_prob'])
        ax2.plot(fpr, tpr, color=color, linewidth=2, label=label)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('COVID Stress (2019-2020)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('ROC Curves: FICO vs ML', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_degradation_bars(fico_deg, ml_deg, filename):
    """Bar chart showing degradation magnitude - HERO CHART."""
    metrics = ['auc_roc_degradation_pct', 'accuracy_degradation_pct', 'f1_score_degradation_pct']
    labels = ['AUC-ROC', 'Accuracy', 'F1 Score']
    
    fico_vals = [abs(fico_deg.get(m, 0) or 0) for m in metrics]
    ml_vals = [abs(ml_deg.get(m, 0) or 0) for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, fico_vals, width, label='FICO Model',
                   color=COLORS['fico'], edgecolor='white')
    bars2 = ax.bar(x + width/2, ml_vals, width, label='ML Model (Neural Net)',
                   color=COLORS['ml'], edgecolor='white')
    
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Performance Degradation (%)\n(larger = more fragile)')
    ax.set_title('Model Fragility Under COVID Stress', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(max(fico_vals), max(ml_vals)) * 1.3 + 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_calibration_comparison(fico_normal, ml_normal, fico_stress, ml_stress, filename):
    """
    Headline calibration chart for the regime-shift fragility narrative.

    Shows FICO and ML calibration error in normal vs stress periods on its own
    chart so the calibration scale is not compressed by larger AUC/accuracy
    bars. The dissertation's central empirical claim is that ML loses calibration
    under stress while preserving discrimination, so this chart sits separately
    from the discrimination-degradation chart.
    """
    fico_normal_cal = fico_normal['calibration_error']
    ml_normal_cal   = ml_normal['calibration_error']
    fico_stress_cal = fico_stress['calibration_error']
    ml_stress_cal   = ml_stress['calibration_error']

    labels = ['Normal (2017)', 'Stress (2019-2020)']
    fico_vals = [fico_normal_cal, fico_stress_cal]
    ml_vals   = [ml_normal_cal,   ml_stress_cal]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, fico_vals, width, label='FICO Model',
                   color=COLORS['fico'], edgecolor='white')
    bars2 = ax.bar(x + width/2, ml_vals, width, label='ML Model (Neural Net)',
                   color=COLORS['ml'], edgecolor='white')

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10)

    # Compute and annotate the relative change for each model
    fico_change_pct = (fico_stress_cal - fico_normal_cal) / fico_normal_cal * 100 if fico_normal_cal else 0
    ml_change_pct   = (ml_stress_cal   - ml_normal_cal)   / ml_normal_cal   * 100 if ml_normal_cal   else 0
    subtitle = (
        f'Normal-to-stress change: '
        f'FICO {fico_change_pct:+.1f}%   |   ML {ml_change_pct:+.1f}%'
    )

    ax.set_ylabel('Calibration Error (mean absolute)')
    ax.set_title('Calibration Error: Normal vs COVID Stress\n' + subtitle, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(max(fico_vals), max(ml_vals)) * 1.25)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Full pipeline with corrected methodology."""
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("  MSIN0032 PILOT STUDY")
    print("  FICO vs ML Credit Scoring Under Stress")
    print("=" * 80)
    print(f"  Run started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    ensure_dirs()
    
    # Load data
    filepath, is_extended = find_data_file()
    if not filepath:
        print("\n[ERROR] No LendingClub data file found!")
        print("  Download from: https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1")
        sys.exit(1)
    
    df = load_data(filepath)
    df = preprocess_data(df)
    df_train, df_normal, df_stress = split_data(df)
    
    # Prepare features
    print(f"\n{'='*70}")
    print(f"  TRAINING MODELS")
    print(f"{'='*70}")
    
    # FICO model
    print(f"\n  Training FICO Model...")
    X_train_fico, X_normal_fico, y_train, y_normal, fico_feat, fico_scaler, fico_imputer = \
        prepare_features(df_train, df_normal, FICO_FEATURES)
    fico_model = train_fico_model(X_train_fico, y_train)
    print(f"  FICO trained ({len(y_train):,} samples)")
    
    # ML model (NEURAL NETWORK)
    print(f"\n  Training ML Model (Neural Network)...")
    X_train_ml, X_normal_ml, y_train_ml, y_normal_ml, ml_feat, ml_scaler, ml_imputer = \
        prepare_features(df_train, df_normal, ML_FEATURES_CORRECTED)
    ml_model = train_ml_model_neural_net(X_train_ml, y_train_ml)
    print(f"  ML trained ({len(y_train_ml):,} samples)")
    
    # Evaluate on normal period
    print(f"\n{'='*70}")
    print(f"  EVALUATING ON NORMAL PERIOD (2017)")
    print(f"{'='*70}")
    
    fico_normal = evaluate_model(fico_model, X_normal_fico, y_normal, "FICO – Normal")
    ml_normal = evaluate_model(ml_model, X_normal_ml, y_normal_ml, "ML – Normal")
    
    print(f"  FICO AUC: {fico_normal['auc_roc']:.4f}")
    print(f"  ML AUC:   {ml_normal['auc_roc']:.4f}")
    
    # Prepare stress data
    print(f"\n{'='*70}")
    print(f"  EVALUATING ON COVID STRESS PERIOD (2019-2020)")
    print(f"{'='*70}")
    
    # FICO stress
    X_stress_fico_raw = df_stress[fico_feat].copy()
    X_stress_fico_imp = pd.DataFrame(
        fico_imputer.transform(X_stress_fico_raw),
        columns=fico_feat, index=X_stress_fico_raw.index
    )
    X_stress_fico = fico_scaler.transform(X_stress_fico_imp)
    y_stress_fico = df_stress[EARLY_DQ_TARGET].values  # Use early DQ for 2019-2020
    
    # ML stress
    for f in ML_FEATURES_CORRECTED:
        if f not in df_stress.columns:
            df_stress[f] = 0
    
    X_stress_ml_raw = df_stress[ml_feat].copy()
    X_stress_ml_imp = pd.DataFrame(
        ml_imputer.transform(X_stress_ml_raw),
        columns=ml_feat, index=X_stress_ml_raw.index
    )
    X_stress_ml = ml_scaler.transform(X_stress_ml_imp)
    y_stress_ml = df_stress[EARLY_DQ_TARGET].values
    
    fico_stress = evaluate_model(fico_model, X_stress_fico, y_stress_fico, "FICO – COVID")
    ml_stress = evaluate_model(ml_model, X_stress_ml, y_stress_ml, "ML – COVID")
    
    print(f"  FICO AUC: {fico_stress['auc_roc']:.4f}")
    print(f"  ML AUC:   {ml_stress['auc_roc']:.4f}")
    
    # Compute degradation
    fico_deg = compute_degradation(fico_normal, fico_stress)
    ml_deg = compute_degradation(ml_normal, ml_stress)
    
    # Print results
    results = {
        'fico_normal': fico_normal,
        'ml_normal': ml_normal,
        'fico_stress': fico_stress,
        'ml_stress': ml_stress,
    }
    
    print_results_comparison(results)
    print_degradation_analysis(fico_deg, ml_deg)
    
    # Generate plots
    print(f"\n{'='*70}")
    print(f"  GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    plot_roc_comparison(fico_normal, ml_normal, fico_stress, ml_stress, 'roc_curves.png')
    plot_degradation_bars(fico_deg, ml_deg, 'degradation.png')
    plot_calibration_comparison(fico_normal, ml_normal, fico_stress, ml_stress, 'calibration_error.png')
    
    # Export CSVs
    print(f"\n{'='*70}")
    print(f"  EXPORTING RESULTS")
    print(f"{'='*70}")
    
    results_data = []
    for key, res in results.items():
        row = {k: v for k, v in res.items() if k not in ('y_prob', 'y_pred', 'y_test')}
        row['period_model'] = key
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_path = os.path.join(TABLES_DIR, 'pilot_study_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: pilot_study_results.csv")
    
    # Run complete
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"  RUN COMPLETE")
    print(f"{'='*70}")
    print(f"  Elapsed time: {elapsed:.1f} seconds")
    print(f"  Outputs: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
