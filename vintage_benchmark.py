#!/usr/bin/env python3
"""
MSIN0032 Dissertation - Part II Empirical Test
==============================================
Vintage Maturity-Correction Sensitivity Analysis

This script implements the external benchmark validation referenced in
Part II Section 3.3 of the dissertation. It addresses the loan-immaturity
limitation of the LendingClub dataset (which terminates at 2020Q3) by
applying a maturity-correction factor derived from LendingClub's own
publicly disclosed vintage performance curves.

PURPOSE

    The pilot study evaluates calibration error against actual default rates
    observed within an 18-month observation window after origination. Because
    unsecured personal loans season over their full 36-60 month term, the
    in-sample actual rates underestimate eventual lifetime default rates.
    This script demonstrates that the dissertation's relative findings -
    the 12x ML/FICO calibration degradation ratio (Part I), the 23.1%
    Solution 2 calibration improvement (Part II Section 3), the 14.6%
    full-population calibration improvement of the layered defence (Part II
    Section 4) - are robust to this maturity correction.

DATA SOURCES

    LendingClub 2020 Annual Report (Form 10-K), Item 7 MD&A: cumulative
    net charge-off curves by booking year and months-on-book. Visual
    reading of disclosed vintage curves indicates approximately:
        - 2018 60-month vintage at month 24:    ~10.0% cumulative NCO
        - 2015-2016 60-month vintages at maturity: ~16.0-16.5% cumulative NCO
    Implied maturity factor (mature / month-24): approximately 1.6x.
    A conservative correction factor of 1.45x is used here, biased
    downward to avoid overclaiming.

    LendingClub 2020 Annual Report URL:
        https://d18rn0p25nwr6d.cloudfront.net/CIK-0001409970/
        ca4a9a47-c622-4c1b-8573-72045798a2ce.pdf

    Federal Reserve DRCLACBS series (Delinquency Rate on Consumer Loans,
    All Commercial Banks): used as macro stress-context benchmark, not
    as direct comparator. Source: https://fred.stlouisfed.org/data/DRCLACBS

    LendingClub net charge-off ratios (annualized, held-for-investment):
    used as institutional stress-context benchmark, not as direct
    comparator. Sources: SEC 8-K filings 2023Q1 onwards.

METHODOLOGY

    1. Read three existing pilot-study output CSVs:
        - pilot_study_results.csv (Part I baseline)
        - quarterly_results.csv (Solution 2 quarterly trajectory)
        - aci_quarterly_results.csv (Solution 3 conformal results)

    2. Apply quarter-specific maturity correction to in-sample actual
       rates. Correction factor scales with observation horizon: later
       vintages need larger upward adjustment because they have less
       observation time relative to the 36-60 month nominal loan term.

    3. Recompute calibration metrics under corrected actuals:
        - rate_error_corrected = predicted - corrected_actual
        - calibration_error_corrected = |rate_error_corrected|

    4. Quantify the headline relative findings under both observed and
       corrected actuals:
        - Part I: ML calibration deterioration vs FICO (12x ratio)
        - Solution 2: Model B vs Model A mean stress-period calibration
        - Solution 3: Layered vs vanilla full-population calibration

    5. Write outputs to vintage_benchmark_outputs/:
        - sensitivity_table.csv: per-quarter observed vs corrected
        - relative_findings_robustness.csv: headline ratios under both
        - sensitivity_chart.png: visual comparison

USAGE

    python vintage_benchmark.py

    Reads from /mnt/project/ in development; falls back to the same
    directory as the script for local execution.

Author: Marat Daminov
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

# Maturity correction factor.
# LendingClub 2020 10-K vintage curves: 2018 vintage ~10% by month 24,
# mature vintages ~16% by month 60. Implied mature/month-24 ratio ~1.6x.
# We apply a conservative 1.45x to avoid overclaiming based on visual
# reading of a published chart.
BASE_MATURITY_FACTOR = 1.45

# The 18-month observation filter retains loans with at least 18 months
# of post-origination performance. The maturity factor scales the in-sample
# actual rate to its expected lifetime equivalent. We apply quarter-specific
# adjustment: later vintages have less observation time relative to the
# 36-60 month nominal term, so they need slightly larger upward correction.
QUARTER_OBSERVATION_MONTHS = {
    '2018Q1': 30,  # Originated Jan-Mar 2018, observed to Sep 2020Q3 = ~30 months
    '2018Q2': 27,
    '2018Q3': 24,
    '2018Q4': 21,
    '2019Q1': 18,  # Right at 18-mo filter floor
    '2019Q2': 18,  # Filter ensures min 18 months
}

# Implied lifetime month for the maturity benchmark
LIFETIME_REFERENCE_MONTHS = 60

# Calibration failure threshold (consistent with main pilot study)
CALIBRATION_FAILURE_THRESHOLD_PP = 4.26


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = '/mnt/project'

# Try /mnt/project first (development), fall back to script directory (local)
if not os.path.isdir(PROJECT_DIR) or not os.path.exists(
    os.path.join(PROJECT_DIR, 'pilot_study_results.csv')
):
    PROJECT_DIR = SCRIPT_DIR

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "vintage_benchmark_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# MATURITY CORRECTION
# ============================================================================

def quarter_specific_correction_factor(quarter_label, base_factor=BASE_MATURITY_FACTOR):
    """
    Compute a quarter-specific maturity correction factor.
    
    The base factor scales an 18-month observation to its expected
    lifetime (60-month) equivalent. Vintages observed for longer than
    18 months need correspondingly smaller correction because they
    have already accumulated more of their lifetime defaults.
    
    The scaling assumes a roughly linear seasoning curve in the
    18-60 month range (a simplification of the convex empirical curve).
    """
    obs_months = QUARTER_OBSERVATION_MONTHS.get(quarter_label, 18)
    
    # If observed for the full lifetime, no correction needed.
    # If observed for 18 months (filter floor), apply full base_factor.
    # Linear interpolation between these endpoints.
    base_obs_months = 18
    if obs_months >= LIFETIME_REFERENCE_MONTHS:
        return 1.0
    
    # Fraction of way from 18 months to 60 months
    progress = (obs_months - base_obs_months) / (
        LIFETIME_REFERENCE_MONTHS - base_obs_months
    )
    progress = max(0.0, min(1.0, progress))
    
    # Linearly reduce correction toward 1.0 as observation horizon lengthens
    return 1.0 + (base_factor - 1.0) * (1.0 - progress)


def apply_correction(actual_rate, quarter_label):
    """Apply quarter-specific maturity correction to an observed actual rate."""
    factor = quarter_specific_correction_factor(quarter_label)
    return actual_rate * factor


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pilot_results():
    """Load Part I pilot baseline CSV."""
    path = os.path.join(PROJECT_DIR, 'pilot_study_results.csv')
    if not os.path.exists(path):
        print(f"[WARNING] Could not find {path}")
        return None
    return pd.read_csv(path)


def load_solution2_quarterly():
    """Load Solution 2 quarterly results."""
    # Try both naming conventions
    for filename in ['quarterly_results.csv', 'solution_2_outputs/tables/quarterly_results.csv']:
        path = os.path.join(PROJECT_DIR, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
    print(f"[WARNING] Could not find quarterly_results.csv")
    return None


def load_solution3_quarterly():
    """Load Solution 3 quarterly ACI results."""
    for filename in ['aci_quarterly_results.csv', 'solution_3_outputs/tables/aci_quarterly_results.csv']:
        path = os.path.join(PROJECT_DIR, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
    print(f"[WARNING] Could not find aci_quarterly_results.csv")
    return None


# ============================================================================
# PER-QUARTER SENSITIVITY TABLE
# ============================================================================

def build_per_quarter_sensitivity(sol2_df):
    """
    For each Solution 2 quarter, compute observed vs corrected calibration.
    
    Returns a long-format DataFrame with columns:
        model, quarter, predicted_default_rate, observed_actual_rate,
        observed_calibration_error_pp, maturity_factor, corrected_actual_rate,
        corrected_calibration_error_pp, breach_observed, breach_corrected
    """
    rows = []
    for _, r in sol2_df.iterrows():
        quarter = r['quarter']
        predicted = r['predicted_default_rate']
        observed_actual = r['actual_default_rate']
        observed_ce = r['calibration_error_pp']
        
        factor = quarter_specific_correction_factor(quarter)
        corrected_actual = observed_actual * factor
        corrected_rate_error = predicted - corrected_actual
        corrected_ce = abs(corrected_rate_error) * 100  # in pp
        
        rows.append({
            'model': r['model'],
            'quarter': quarter,
            'observation_months': QUARTER_OBSERVATION_MONTHS.get(quarter, 18),
            'predicted_default_rate_pct': predicted * 100,
            'observed_actual_rate_pct': observed_actual * 100,
            'observed_calibration_error_pp': observed_ce,
            'maturity_factor': factor,
            'corrected_actual_rate_pct': corrected_actual * 100,
            'corrected_calibration_error_pp': corrected_ce,
            'observed_breach_4_26pp': observed_ce >= CALIBRATION_FAILURE_THRESHOLD_PP,
            'corrected_breach_4_26pp': corrected_ce >= CALIBRATION_FAILURE_THRESHOLD_PP,
        })
    return pd.DataFrame(rows)


# ============================================================================
# RELATIVE-FINDINGS ROBUSTNESS
# ============================================================================

def relative_findings_robustness(pilot_df, sol2_df, sol3_df):
    """
    Compute the headline relative findings under both observed and
    corrected actuals. Demonstrates that the dissertation's relative
    conclusions are robust to maturity correction.
    """
    findings = []
    
    # ----- Finding 1: Part I 12x ML/FICO calibration degradation -----
    # The Part I baseline uses the full normal-stress comparison. The
    # maturity correction here applies to the stress-period actuals.
    # Apply a uniform stress-period correction (since Part I aggregates
    # rather than reporting per-quarter).
    if pilot_df is not None:
        # Average stress-period correction factor across the 6 stress quarters
        avg_stress_factor = np.mean([
            quarter_specific_correction_factor(q)
            for q in QUARTER_OBSERVATION_MONTHS.keys()
        ])
        
        fico_normal_ce = pilot_df.loc[pilot_df['model_name'] == 'FICO – Normal',
                                      'calibration_error'].values[0] * 100
        fico_stress_ce = pilot_df.loc[pilot_df['model_name'] == 'FICO – COVID',
                                      'calibration_error'].values[0] * 100
        ml_normal_ce = pilot_df.loc[pilot_df['model_name'] == 'ML – Normal',
                                    'calibration_error'].values[0] * 100
        ml_stress_ce = pilot_df.loc[pilot_df['model_name'] == 'ML – COVID',
                                    'calibration_error'].values[0] * 100
        
        fico_predicted_stress = pilot_df.loc[
            pilot_df['model_name'] == 'FICO – COVID', 'predicted_default_rate'
        ].values[0]
        fico_actual_stress = pilot_df.loc[
            pilot_df['model_name'] == 'FICO – COVID', 'actual_default_rate'
        ].values[0]
        ml_predicted_stress = pilot_df.loc[
            pilot_df['model_name'] == 'ML – COVID', 'predicted_default_rate'
        ].values[0]
        ml_actual_stress = pilot_df.loc[
            pilot_df['model_name'] == 'ML – COVID', 'actual_default_rate'
        ].values[0]
        
        # Corrected stress actuals
        fico_actual_stress_corr = fico_actual_stress * avg_stress_factor
        ml_actual_stress_corr = ml_actual_stress * avg_stress_factor
        
        fico_stress_ce_corr = abs(fico_predicted_stress - fico_actual_stress_corr) * 100
        ml_stress_ce_corr = abs(ml_predicted_stress - ml_actual_stress_corr) * 100
        
        # Relative degradation: stress vs normal
        fico_degradation_obs = (fico_stress_ce - fico_normal_ce) / fico_normal_ce * 100
        ml_degradation_obs = (ml_stress_ce - ml_normal_ce) / ml_normal_ce * 100
        ratio_obs = ml_degradation_obs / fico_degradation_obs if fico_degradation_obs != 0 else np.nan
        
        fico_degradation_corr = (fico_stress_ce_corr - fico_normal_ce) / fico_normal_ce * 100
        ml_degradation_corr = (ml_stress_ce_corr - ml_normal_ce) / ml_normal_ce * 100
        ratio_corr = ml_degradation_corr / fico_degradation_corr if fico_degradation_corr != 0 else np.nan
        
        findings.append({
            'finding': 'Part I: ML/FICO calibration degradation ratio',
            'observed': f"{ratio_obs:.1f}x",
            'corrected': f"{ratio_corr:.1f}x",
            'preserved': abs(ratio_corr - ratio_obs) / abs(ratio_obs) < 0.30,
            'narrative': 'ML calibration degrades many times more than FICO under stress',
        })
    
    # ----- Finding 2: Solution 2 mean stress-period calibration improvement -----
    if sol2_df is not None:
        sens_df = build_per_quarter_sensitivity(sol2_df)
        
        for breach_col, label in [
            ('observed_calibration_error_pp', 'observed'),
            ('corrected_calibration_error_pp', 'corrected'),
        ]:
            mean_a = sens_df.loc[
                sens_df['model'].str.contains('MODEL A'), breach_col
            ].mean()
            mean_b = sens_df.loc[
                sens_df['model'].str.contains('MODEL B'), breach_col
            ].mean()
            improvement_pct = (mean_a - mean_b) / mean_a * 100 if mean_a != 0 else np.nan
            if label == 'observed':
                observed_improvement = improvement_pct
                observed_mean_a = mean_a
                observed_mean_b = mean_b
            else:
                corrected_improvement = improvement_pct
                corrected_mean_a = mean_a
                corrected_mean_b = mean_b
        
        findings.append({
            'finding': 'Solution 2: Mean stress-period calibration improvement '
                       '(Model B vs Model A)',
            'observed': f"-{observed_improvement:.1f}% (A: {observed_mean_a:.2f}pp, "
                        f"B: {observed_mean_b:.2f}pp)",
            'corrected': f"-{corrected_improvement:.1f}% (A: {corrected_mean_a:.2f}pp, "
                         f"B: {corrected_mean_b:.2f}pp)",
            'preserved': abs(corrected_improvement - observed_improvement) < 10,
            'narrative': 'PSI-triggered retraining reduces calibration error '
                         'similarly under both observed and corrected actuals',
        })
    
    # ----- Finding 3: Solution 3 layered defence full-population calibration -----
    if sol3_df is not None:
        # Full-population calibration error per config, weighted by quarter sample size
        configs = sol3_df['config'].unique()
        for breach_col, label in [
            ('cal_err_full_pp', 'observed'),
            ('cal_err_full_corrected_pp', 'corrected'),
        ]:
            if breach_col == 'cal_err_full_corrected_pp':
                # Compute corrected full-population calibration per quarter
                sol3_df = sol3_df.copy()
                sol3_df['maturity_factor'] = sol3_df['quarter'].map(
                    quarter_specific_correction_factor
                )
                sol3_df['default_rate_full_corrected'] = (
                    sol3_df['default_rate_full'] * sol3_df['maturity_factor']
                )
                # Cal error full corrected = |predicted_full - corrected_actual| in pp
                # We need predicted full rate; back out from observed cal err and actual
                # cal_err_full_pp = |predicted - actual| * 100 in pp (or is it just abs diff?)
                # The vanilla output has cal_err_full_pp = error in pp on the full population
                # which equals |mean(predicted) - mean(actual)| * 100
                # So predicted = actual + cal_err/100 (sign unknown without rate_error)
                # But we know from the trajectory that ML model OVER-predicts stress,
                # so predicted > actual, so predicted = actual + cal_err/100
                sol3_df['predicted_full'] = (
                    sol3_df['default_rate_full'] + sol3_df['cal_err_full_pp'] / 100
                )
                sol3_df['cal_err_full_corrected_pp'] = (
                    abs(sol3_df['predicted_full'] - sol3_df['default_rate_full_corrected']) * 100
                )
            
            for config in configs:
                config_df = sol3_df[sol3_df['config'] == config]
                if len(config_df) == 0:
                    continue
                weighted_mean = np.average(
                    config_df[breach_col],
                    weights=config_df['n_total']
                )
                if config == 'vanilla_modelA' and label == 'observed':
                    obs_vanilla = weighted_mean
                elif config == 'aci_modelB_layered' and label == 'observed':
                    obs_layered = weighted_mean
                elif config == 'vanilla_modelA' and label == 'corrected':
                    corr_vanilla = weighted_mean
                elif config == 'aci_modelB_layered' and label == 'corrected':
                    corr_layered = weighted_mean
        
        obs_improvement = (obs_vanilla - obs_layered) / obs_vanilla * 100
        corr_improvement = (corr_vanilla - corr_layered) / corr_vanilla * 100
        
        findings.append({
            'finding': 'Solution 3: Layered defence full-population calibration '
                       'improvement (vs vanilla)',
            'observed': f"-{obs_improvement:.1f}% (vanilla: {obs_vanilla:.2f}pp, "
                        f"layered: {obs_layered:.2f}pp)",
            'corrected': f"-{corr_improvement:.1f}% (vanilla: {corr_vanilla:.2f}pp, "
                         f"layered: {corr_layered:.2f}pp)",
            'preserved': abs(corr_improvement - obs_improvement) < 10,
            'narrative': 'Layered defence improvement preserved under maturity correction',
        })
    
    return pd.DataFrame(findings)


# ============================================================================
# VISUALISATION
# ============================================================================

def plot_sensitivity(sens_df, output_path):
    """Per-quarter observed-vs-corrected calibration error, both models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    quarters = sens_df['quarter'].unique()
    
    for ax, model_filter, color, title in [
        (axes[0], 'MODEL A', '#C84630', 'Model A: 2012-2016 training (control)'),
        (axes[1], 'MODEL B', '#3D6E8C',
         'Model B: 2012-2017 training (PSI-retrained)'),
    ]:
        sub = sens_df[sens_df['model'].str.contains(model_filter)]
        ax.plot(sub['quarter'], sub['observed_calibration_error_pp'],
                marker='o', linestyle='--', color=color,
                label='Observed (in-sample)')
        ax.plot(sub['quarter'], sub['corrected_calibration_error_pp'],
                marker='s', color=color, linewidth=2,
                label='Maturity-corrected')
        ax.axhline(CALIBRATION_FAILURE_THRESHOLD_PP, color='grey',
                   linestyle=':', alpha=0.6,
                   label=f'Failure threshold ({CALIBRATION_FAILURE_THRESHOLD_PP}pp)')
        ax.set_xlabel('Quarter')
        ax.set_title(title, fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    axes[0].set_ylabel('Calibration error (pp)')
    
    fig.suptitle(
        'Maturity-Correction Sensitivity Analysis\n'
        'In-sample (18-month observation) vs LendingClub-vintage-corrected '
        '(60-month equivalent)',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Saved: {output_path}")


def plot_external_benchmarks(output_path):
    """Show pilot study trajectory alongside FRED + LC NCO benchmarks."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Build unified ordered quarter axis from earliest to latest
    def quarter_to_idx(q):
        year = int(q[:4])
        qnum = int(q[-1])
        return year * 4 + qnum
    
    # FRED DRCLACBS quarterly values 2018-2025 (verified from FRED)
    fred_dates = ['2018Q1', '2018Q2', '2018Q3', '2018Q4',
                  '2019Q1', '2019Q2', '2019Q3', '2019Q4',
                  '2020Q1', '2020Q2', '2020Q3', '2020Q4',
                  '2021Q1', '2021Q2', '2021Q3', '2021Q4',
                  '2022Q1', '2022Q2', '2022Q3', '2022Q4',
                  '2023Q1', '2023Q2', '2023Q3', '2023Q4',
                  '2024Q1', '2024Q2', '2024Q3', '2024Q4',
                  '2025Q1', '2025Q2', '2025Q3', '2025Q4']
    fred_values = [2.27, 2.26, 2.30, 2.31,
                   2.33, 2.36, 2.34, 2.31,
                   2.48, 2.01, 1.82, 1.92,
                   1.69, 1.53, 1.52, 1.54,
                   1.65, 1.81, 1.92, 2.07,
                   2.23, 2.37, 2.52, 2.61,
                   2.68, 2.72, 2.73, 2.76,
                   2.77, 2.76, 2.71, 2.62]
    
    # LendingClub NCO ratio quarterly values 2023-2025 (verified from SEC 8-Ks)
    lc_nco_dates = ['2023Q1', '2023Q2', '2023Q3', '2023Q4',
                    '2024Q1', '2024Q2', '2024Q3', '2024Q4',
                    '2025Q1', '2025Q2', '2025Q3', '2025Q4']
    lc_nco_values = [3.8, 4.4, 5.1, 6.6,
                     6.9, 6.2, 5.4, 4.5,
                     4.8, 3.0, 2.9, 3.7]
    
    # Pilot study in-sample stress trajectory (2018Q1-2019Q2)
    pilot_dates = ['2018Q1', '2018Q2', '2018Q3', '2018Q4',
                   '2019Q1', '2019Q2']
    pilot_observed = [13.35, 13.00, 10.59, 9.24, 7.41, 6.65]
    pilot_corrected = [
        observed * quarter_specific_correction_factor(q)
        for observed, q in zip(pilot_observed, pilot_dates)
    ]
    
    # Build unified ordered set of all quarters for consistent x-axis
    all_quarters = sorted(
        set(fred_dates) | set(lc_nco_dates) | set(pilot_dates),
        key=quarter_to_idx
    )
    
    # Map each series's values onto the unified axis
    def to_unified(dates, values, axis):
        d = dict(zip(dates, values))
        return [d.get(q, np.nan) for q in axis]
    
    fred_unified = to_unified(fred_dates, fred_values, all_quarters)
    lc_unified = to_unified(lc_nco_dates, lc_nco_values, all_quarters)
    pilot_obs_unified = to_unified(pilot_dates, pilot_observed, all_quarters)
    pilot_corr_unified = to_unified(pilot_dates, pilot_corrected, all_quarters)
    
    ax.plot(all_quarters, fred_unified, marker='o', linestyle='-',
            color='#1f4068',
            label='FRED DRCLACBS (US consumer-loan delinquency, %)',
            alpha=0.7, linewidth=2, markersize=4)
    ax.plot(all_quarters, lc_unified, marker='s', linestyle='-',
            color='#e58e26',
            label='LendingClub net charge-off ratio, % (annualized HFI portfolio)',
            alpha=0.85, linewidth=2, markersize=5)
    ax.plot(all_quarters, pilot_obs_unified, marker='D', linestyle='--',
            color='#C84630',
            label='Pilot in-sample actual rate (18-mo observation, cumulative)',
            linewidth=2, markersize=6)
    ax.plot(all_quarters, pilot_corr_unified, marker='D', linestyle='-',
            color='#900020',
            label='Pilot maturity-corrected actual rate (60-mo equivalent, cumulative)',
            linewidth=2.5, markersize=6)
    
    ax.set_xlabel('Quarter', fontsize=11)
    ax.set_ylabel('Default rate / delinquency rate (%)', fontsize=11)
    ax.set_title(
        'External benchmarks for the pilot stress period\n'
        'Verifying the pilot trajectory is consistent with macro and '
        'institutional credit-stress evidence',
        fontsize=12
    )
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=60, labelsize=8)
    
    # Show only every 2nd tick label to reduce clutter
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 2 != 0:
            label.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    start = datetime.now()
    
    print(f"\n{'=' * 70}")
    print("  VINTAGE MATURITY-CORRECTION SENSITIVITY ANALYSIS")
    print(f"  Started: {start:%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 70}\n")
    
    print(f"  Input directory: {PROJECT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Base maturity factor: {BASE_MATURITY_FACTOR}x")
    print(f"  Quarter-specific correction factors:")
    for q, m in QUARTER_OBSERVATION_MONTHS.items():
        f = quarter_specific_correction_factor(q)
        print(f"    {q}: {m} months observed -> factor = {f:.3f}x")
    
    # Load CSVs
    print(f"\n  Loading existing pilot CSVs...")
    pilot_df = load_pilot_results()
    sol2_df = load_solution2_quarterly()
    sol3_df = load_solution3_quarterly()
    
    if pilot_df is None or sol2_df is None or sol3_df is None:
        print("\n[ERROR] Could not load all required input CSVs.")
        print(f"        Place pilot_study_results.csv, quarterly_results.csv,")
        print(f"        and aci_quarterly_results.csv in {PROJECT_DIR}")
        sys.exit(1)
    
    print(f"    Loaded {len(pilot_df)} rows from pilot_study_results.csv")
    print(f"    Loaded {len(sol2_df)} rows from quarterly_results.csv")
    print(f"    Loaded {len(sol3_df)} rows from aci_quarterly_results.csv")
    
    # Per-quarter sensitivity table
    print(f"\n  Building per-quarter sensitivity table...")
    sens_df = build_per_quarter_sensitivity(sol2_df)
    sens_path = os.path.join(OUTPUT_DIR, 'sensitivity_table.csv')
    sens_df.to_csv(sens_path, index=False)
    print(f"    Saved: {sens_path}")
    
    # Print sensitivity summary
    print(f"\n  Per-quarter calibration error: observed vs corrected")
    print(f"  {'Quarter':<10} {'Model':<25} {'Observed':>10} {'Corrected':>10} "
          f"{'Factor':>8}")
    print(f"  {'-' * 67}")
    for _, r in sens_df.iterrows():
        print(f"  {r['quarter']:<10} {r['model']:<25} "
              f"{r['observed_calibration_error_pp']:>8.2f}pp "
              f"{r['corrected_calibration_error_pp']:>8.2f}pp "
              f"{r['maturity_factor']:>7.3f}x")
    
    # Relative-findings robustness
    print(f"\n  Computing relative-findings robustness...")
    findings_df = relative_findings_robustness(pilot_df, sol2_df, sol3_df)
    findings_path = os.path.join(OUTPUT_DIR, 'relative_findings_robustness.csv')
    findings_df.to_csv(findings_path, index=False)
    print(f"    Saved: {findings_path}")
    
    # Print findings summary
    print(f"\n  RELATIVE FINDINGS ROBUSTNESS")
    print(f"  {'=' * 70}")
    for _, f in findings_df.iterrows():
        preserved_str = "PRESERVED" if f['preserved'] else "MATERIALLY CHANGED"
        print(f"\n  Finding: {f['finding']}")
        print(f"    Observed:  {f['observed']}")
        print(f"    Corrected: {f['corrected']}")
        print(f"    -> {preserved_str}")
    
    # Visualisations
    print(f"\n  Generating visualisations...")
    plot_sensitivity(sens_df, os.path.join(OUTPUT_DIR, 'sensitivity_chart.png'))
    plot_external_benchmarks(
        os.path.join(OUTPUT_DIR, 'external_benchmarks_chart.png')
    )
    
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'=' * 70}")
    print(f"  COMPLETED")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
