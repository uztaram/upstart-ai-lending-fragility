#!/usr/bin/env python3
"""
Monte Carlo simulation for layered defence — REVISED cost ranges
Updated from sourced cost-derivation appendices.

Cost changes from previous version:
  Solution 1: $0.5-1.5M -> $0.4-0.9M  (component-up derivation; mode $0.65M)
  Solution 2: $0.8-1.7M -> $0.3-0.8M  (component-up derivation; mode $0.55M)
  Solution 3: $5-25M    -> $3-10M     (revised volume calibration; mode $6.5M)

These ranges are now defensibly sourced (Appendix A.4, B.3, C.4).
Other parameters unchanged (volatility loss, contributions, lit benefit, discount).
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ITERATIONS = 5000
HORIZON_YEARS = 5
DISCOUNT_RATE = 0.12
TARGET_VOLATILITY_REDUCTION_PCT = 50.0

PARAMS = {
    'annual_loss_no_intervention_M':  (20.0, 40.0, 80.0),
    'sol1_contribution_pct':          (22.0, 27.0, 35.0),
    'sol2_contribution_pct':          (12.0, 17.0, 22.0),
    'sol3_contribution_pct':          (4.0,   8.0, 12.0),
    'coordination_discount_pct':      (5.0,  15.0, 25.0),
    'sol1_cost_M':                    (0.4,   0.65, 0.9),   # REVISED
    'sol2_cost_M':                    (0.3,   0.55, 0.8),   # REVISED
    'sol3_cost_M':                    (3.0,   6.5,  10.0),  # REVISED
    'litigation_benefit_y1_M':        (5.0,  12.0, 20.0),
}

rng = np.random.default_rng(RANDOM_SEED)

def run_iteration():
    p = {name: rng.triangular(low, mode, high)
         for name, (low, mode, high) in PARAMS.items()}
    
    raw_sum = p['sol1_contribution_pct'] + p['sol2_contribution_pct'] + p['sol3_contribution_pct']
    coord = p['coordination_discount_pct'] / 100
    combined_red = raw_sum * (1 - coord)
    
    annual_revenue_preserved = (combined_red / 50.0) * p['annual_loss_no_intervention_M']
    annual_cost = p['sol1_cost_M'] + p['sol2_cost_M'] + p['sol3_cost_M']
    
    cash_flows = []
    for year in range(1, HORIZON_YEARS + 1):
        cf = annual_revenue_preserved - annual_cost
        if year == 1:
            cf += p['litigation_benefit_y1_M']
        cash_flows.append(cf)
    
    npv = sum(cf / ((1 + DISCOUNT_RATE) ** y) for y, cf in enumerate(cash_flows, 1))
    
    return {**p, 'combined_reduction_pct': combined_red, 'annual_cost_M': annual_cost,
            'npv_M': npv, 'meets_target': combined_red >= 50.0, 'positive_npv': npv > 0}

results = pd.DataFrame([run_iteration() for _ in range(N_ITERATIONS)])

print("="*70)
print("REVISED MONTE CARLO RESULTS (sourced cost ranges)")
print("="*70)
print()
print(f"NPV ($M, 5yr horizon at 12% discount):")
for q, lab in [(0.10, 'P10'), (0.50, 'Median'), (0.90, 'P90')]:
    print(f"  {lab:7s}: ${results['npv_M'].quantile(q):6.1f}M")
print(f"  Mean:   ${results['npv_M'].mean():6.1f}M")
print()
print(f"Volatility reduction (%):")
for q, lab in [(0.10, 'P10'), (0.50, 'Median'), (0.90, 'P90')]:
    print(f"  {lab:7s}: {results['combined_reduction_pct'].quantile(q):5.1f}%")
print(f"  Mean:   {results['combined_reduction_pct'].mean():5.1f}%")
print()
print(f"Probabilities:")
print(f"  P(NPV > 0):                 {results['positive_npv'].mean()*100:5.1f}%")
print(f"  P(NPV > $50M):              {(results['npv_M']>50).mean()*100:5.1f}%")
print(f"  P(NPV > $100M):             {(results['npv_M']>100).mean()*100:5.1f}%")
print(f"  P(meets 50% target):        {results['meets_target'].mean()*100:5.1f}%")
print()
print(f"Annual cost distribution:")
print(f"  Mean: ${results['annual_cost_M'].mean():.1f}M")
print(f"  P10-P90: ${results['annual_cost_M'].quantile(0.1):.1f}M-${results['annual_cost_M'].quantile(0.9):.1f}M")
