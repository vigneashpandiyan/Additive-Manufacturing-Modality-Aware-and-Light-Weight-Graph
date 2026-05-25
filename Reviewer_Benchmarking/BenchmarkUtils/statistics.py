# -*- coding: utf-8 -*-
"""
Statistical testing and Confidence Interval utilities.
"""

import numpy as np
import scipy.stats as stats


def compute_95_ci(values):
    """
    Compute the 95% confidence interval for a list/array of values using the t-distribution.
    """
    n = len(values)
    if n < 2:
        return 0.0
    sem = stats.sem(values)
    if sem == 0:
        return 0.0
    ci = sem * stats.t.ppf((1 + 0.95) / 2.0, n - 1)
    return ci


def run_statistical_test(proposed_values, baseline_values):
    """
    Perform a paired t-test comparing the Proposed model scores to a baseline model's scores
    across identical seed runs.
    Returns: t-statistic, p-value
    """
    if len(proposed_values) != len(baseline_values) or len(proposed_values) < 2:
        return 0.0, 1.0
    
    if np.array_equal(proposed_values, baseline_values):
        return 0.0, 1.0
        
    t_stat, p_val = stats.ttest_rel(proposed_values, baseline_values)
    
    if np.isnan(t_stat) or np.isnan(p_val):
        return 0.0, 1.0
        
    return t_stat, p_val
