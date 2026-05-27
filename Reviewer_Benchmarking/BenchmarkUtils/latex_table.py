# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Formatting accuracy, parameter counts, memory, and statistical p-values into clean LaTeX tables ready for the manuscript.

Note: Any reuse of this code should be authorized by the code author.
"""

import os
import numpy as np


def generate_latex_table(aggregated_df, p_values_dict, save_folder):
    """
    Description:
        Formats accuracy, precision, recall, f1, roc-auc, parameters, and paired statistical significance p-value markers across all evaluated architectures into a publication-ready LaTeX tabular format. Emphasizes best performing indices using LaTeX bold tags.
    Purpose:
        To auto-generate the central performance baseline comparison LaTeX block directly for direct manuscript inclusion.
    Input Types:
        - aggregated_df (pandas.DataFrame): Aggregated performance and resource metrics of the benchmarked models.
        - p_values_dict (dict): Dictionary mapping model names to calculated significance test p-values.
        - save_folder (str): Directory where the compiled LaTeX table code (.tex) is saved.
    Output Types:
        - latex_code (str): Entire generated LaTeX string code.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Comparative benchmarking of the proposed Modality-Aware Shapelet--GAT against state-of-the-art baselines across multiple seeds. Metrics report Mean $\pm$ 95\% Confidence Interval. Statistical significance compared to the proposed model is indicated by $^{*}$ ($p < 0.05$) or $^{**}$ ($p < 0.01$).}")
    latex_lines.append(r"\label{tab:reviewer_benchmarking}")
    latex_lines.append(r"\begin{tabular}{lcccccc}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\textbf{Model} & \textbf{Accuracy (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1-Score (\%)} & \textbf{ROC-AUC} & \textbf{Parameters (K)} \\")
    latex_lines.append(r"\hline")
    
    models = aggregated_df["Model"].tolist()
    
    acc_vals = aggregated_df["Accuracy_mean"].tolist()
    prec_vals = aggregated_df["Precision_mean"].tolist()
    rec_vals = aggregated_df["Recall_mean"].tolist()
    f1_vals = aggregated_df["F1_mean"].tolist()
    auc_vals = aggregated_df["ROC-AUC_mean"].tolist()
    
    best_acc_idx = np.argmax(acc_vals)
    best_prec_idx = np.argmax(prec_vals)
    best_rec_idx = np.argmax(rec_vals)
    best_f1_idx = np.argmax(f1_vals)
    best_auc_idx = np.argmax(auc_vals)
    
    for idx, row in aggregated_df.iterrows():
        model_name = row["Model"]
        
        param_count_k = row["Parameters"] / 1000.0 if row["Parameters"] != "N/A" else "N/A"
        param_str = f"{param_count_k:.1f}" if isinstance(param_count_k, float) else "N/A"
        
        acc_mean, acc_ci = row["Accuracy_mean"] * 100, row["Accuracy_ci"] * 100
        prec_mean, prec_ci = row["Precision_mean"] * 100, row["Precision_ci"] * 100
        rec_mean, rec_ci = row["Recall_mean"] * 100, row["Recall_ci"] * 100
        f1_mean, f1_ci = row["F1_mean"] * 100, row["F1_ci"] * 100
        auc_mean, auc_ci = row["ROC-AUC_mean"], row["ROC-AUC_ci"]
        
        acc_str = f"{acc_mean:.2f} \\pm {acc_ci:.2f}"
        prec_str = f"{prec_mean:.2f} \\pm {prec_ci:.2f}"
        rec_str = f"{rec_mean:.2f} \\pm {rec_ci:.2f}"
        f1_str = f"{f1_mean:.2f} \\pm {f1_ci:.2f}"
        auc_str = f"{auc_mean:.3f} \\pm {auc_ci:.3f}"
        
        if idx == best_acc_idx: acc_str = f"\\mathbf{{{acc_str}}}"
        if idx == best_prec_idx: prec_str = f"\\mathbf{{{prec_str}}}"
        if idx == best_rec_idx: rec_str = f"\\mathbf{{{rec_str}}}"
        if idx == best_f1_idx: f1_str = f"\\mathbf{{{f1_str}}}"
        if idx == best_auc_idx: auc_str = f"\\mathbf{{{auc_str}}}"
        
        sig_marker = ""
        if model_name in p_values_dict:
            p_val = p_values_dict[model_name]
            if p_val < 0.01:
                sig_marker = r"$^{**}$"
            elif p_val < 0.05:
                sig_marker = r"$^{*}$"
                
        clean_model_name = model_name.replace("_", "--").replace("-", "--")
        if "Shapelet" in clean_model_name:
            clean_model_name = r"\textbf{" + clean_model_name + " (Proposed)}"
            
        line = f"{clean_model_name}{sig_marker} & {acc_str} & {prec_str} & {rec_str} & {f1_str} & {auc_str} & {param_str} \\\\"
        latex_lines.append(line)
        
    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table*}")
    
    latex_code = "\n".join(latex_lines)
    
    with open(os.path.join(save_folder, "manuscript_table.tex"), "w") as f:
        f.write(latex_code)
        
    print("[UTILS] LaTeX table generated successfully in Figures/Comparison/!")
    return latex_code
