# -*- coding: utf-8 -*-
"""
Manuscript: "Learning Composition-Sensitive Signatures in Multi-Material PBF-LB: A Lightweight, Modality-Aware, ExplainableGraph-Attention Sensor Fusion Framework for In-Situ Monitoring of Graded 316L–CuCrZr Alloys"
Author: vpsora
Contact: vigneashwara.solairajapandiyan@utu.fi, vigneashpandiyan@gmail.com
Date: May 2026
Time: 14:04:18

Implementation Includes:
- Wrapper importing the proposed Shapelet–GAT model (GNNWithAttention) from the original codebase.

Note: Any reuse of this code should be authorized by the code author.
"""

import os
import sys

# Resolve paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # Network/
BENCHMARK_DIR = os.path.dirname(CURRENT_DIR)             # Reviewer_Benchmarking/
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)            # Project Root
SENSOR_FUSION_DIR = os.path.join(PROJECT_ROOT, "Sensor fusion")
if SENSOR_FUSION_DIR not in sys.path:
    sys.path.insert(0, SENSOR_FUSION_DIR)

# Re-export original model
from network.Network import GNNWithAttention
