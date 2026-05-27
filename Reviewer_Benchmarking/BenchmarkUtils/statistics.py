# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Performing Wilcoxon rank-sum or t-tests between the proposed framework and baseline models.
- Computing 95% confidence intervals (CI) for experimental results.

Note: Any reuse of this code should be authorized by the code author.
"""

import numpy as np
import scipy.stats as stats


def compute_95_ci(values):
    """
    Description:
        Computes the analytical 95% confidence interval for a list/array of metrics using standard error of the mean (SEM) and Student's t-distribution critical values.
    Purpose:
        To calculate experimental margin of error for standardizing statistical significance tables.
    Input Types:
        - values (list or numpy.ndarray): Numerical observations across different random seed trials.
    Output Types:
        - ci (float): Margin value representing the 95% confidence interval.
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
    Description:
        Performs a paired sample Student's t-test comparing the metric vectors of the proposed Shapelet-GAT against a given baseline model.
    Purpose:
        To establish whether proposed framework performance gains are statistically significant rather than artifactual.
    Input Types:
        - proposed_values (list or numpy.ndarray): Performance metrics of the proposed Shapelet-GAT.
        - baseline_values (list or numpy.ndarray): Performance metrics of the baseline model.
    Output Types:
        - tuple: (t_stat, p_val)
            - t_stat (float): Calculated test statistic.
            - p_val (float): Calculated significance probability p-value.
    """
    if len(proposed_values) != len(baseline_values) or len(proposed_values) < 2:
        return 0.0, 1.0
    
    if np.array_equal(proposed_values, baseline_values):
        return 0.0, 1.0
    
    t_stat, p_val = stats.ttest_rel(proposed_values, baseline_values)
    
    if np.isnan(t_stat) or np.isnan(p_val):
        return 0.0, 1.0
        
    return t_stat, p_val
