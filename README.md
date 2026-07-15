## A lightweight, modality-aware, and explainable Shapelet–GAT framework for graded 316L–CuCrZr alloys

This repository provides the implementation, benchmarking scripts, and visualization utilities associated with the article:

> **Learning composition-sensitive signatures in multi-material PBF-LB: a lightweight, modality-aware, explainable graph-attention sensor fusion framework for in-situ monitoring of graded 316L–CuCrZr alloys**

**Journal:** *Progress in Additive Manufacturing*  
**DOI:** [10.1007/s40964-026-01854-x](https://doi.org/10.1007/s40964-026-01854-x)

---

## Scientific motivation

Process monitoring in multi-material laser powder bed fusion (PBF-LB) is inherently more complex than in single-alloy processing because the melt-pool response evolves with local composition. In the graded 316L–CuCrZr system, changes in CuCrZr content influence optical absorptivity and reflectivity near the processing-laser wavelength, thermal conductivity, melting and solidification behavior, melt-pool geometry and stability, and phase formation and cracking susceptibility. Consequently, the measured process emissions vary across the compositional gradient, requiring monitoring strategies capable of resolving both material-dependent and process-dependent changes.

![Experimental Setup](Data/Figure%201.jpg)

As the CuCrZr fraction increases, the melt pool becomes shallower and the process transitions from deeper keyhole or transition-mode behavior toward conduction-mode melting. Cu-rich regions also exhibit higher reflectivity near the laser wavelength. These changes influence the measured process emissions in different ways:

- **Acoustic emission (AE)** reflects melt-pool dynamics, mechanical transients, process instabilities, and changes in acoustic energy distribution.
- **Back-reflected optical emission (OE)** is sensitive to changes in laser–material coupling, absorptivity, reflectivity, and process-zone intensity.

No single sensing modality captures all composition-dependent effects consistently across the full gradient. The framework therefore treats multimodal sensing as a physically necessary source of complementary information rather than as a simple increase in input dimensionality.

![Code Pipeline](Data/Figure%203.jpg)
---

## Main contribution

The repository implements a **Shapelet–Graph Attention Network (Shapelet–GAT)** that learns composition-sensitive signatures directly from synchronized AE and OE time series.

The method combines:

1. **Learnable channel-wise shapelets** for extracting localized and class-discriminative temporal motifs.
2. **Temporal graph construction** in which overlapping signal segments form graph nodes.
3. **Graph attention** for learning the relative importance of temporal segments and their interactions.
   
   <img src="Data/Figure%202.jpg" width="60%" alt="Graph Connection" />

4. **Modality-aware interpretation** through channel saliency, shapelet activation, and node-wise attribution.
5. **Compact parameterization** with only **3,753 trainable parameters**, reported as approximately **3.8k parameters**.

The shapelets, graph-attention layers, and classifier are optimized jointly in a single end-to-end learning process.

---

## 📁 Repository Directory Structure

```directory
├── Features Visualization/      # Frequency-domain spectral features & signal diagnostics
├── Model Benchmarking/          # Comprehensive baseline models & LaTeX helper tools
├── Sensor fusion/               # Core multi-modal GAT-Shapelet framework code
├── Unimodal/                    # Single-modality baselines (Optical or Acoustic alone)
└── Data/                        # [Input Target] Holds raw .npy signal spaces and labels
```

---

## ⚙️ Folder Contents & Components

| Folder / Component | Objective | Key Files / Subdirectories |
| :--- | :--- | :--- |
| **🧠 [Sensor fusion/](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion)** | Core multi-modal sensor fusion pipeline orchestrating the GAT-Shapelet framework. | <ul><li>[Config.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion/Config.py)</li><li>[Main.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion/Main.py)</li><li>[Main_Shapelet_Optimization.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion/Main_Shapelet_Optimization.py)</li><li>[dataloader/data_loader.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion/dataloader/data_loader.py)</li><li>[network/Network.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion/network/Network.py)</li><li>[trainer/Trainer.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion/trainer/Trainer.py)</li><li>[utils/Inference.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion/utils/Inference.py) / [Utils.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion/utils/Utils.py)</li><li>[visualization/Graph.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion/visualization/Graph.py) / [Visualization.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Sensor%20fusion/visualization/Visualization.py)</li></ul> |
| **🎛️ [Model Benchmarking/](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Model%20Benchmarking)** | Evaluation suite comparing the GAT-Shapelet against seven sequence, graph, and ML baselines. | <ul><li>[Run_benchmarks.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Model%20Benchmarking/Run_benchmarks.py)</li><li>[Dataloader/data_loader.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Model%20Benchmarking/Dataloader/data_loader.py) / [features.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Model%20Benchmarking/Dataloader/features.py)</li><li>[Network/](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Model%20Benchmarking/Network) (CNN, LSTM, TCN, GAT, GCN)</li><li>[Utils/](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Model%20Benchmarking/Utils) (Metrics & LaTeX reporting)</li><li>[Visualization/](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Model%20Benchmarking/Visualization) (Plots & matrices)</li></ul> |
| **📻 [Unimodal/](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Unimodal)** | Single-modality baselines (Optical or Acoustic alone) to quantify the benefits of sensor fusion. | <ul><li>[Shaplet_Unimodal.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Unimodal/Shaplet_Unimodal.py)</li><li>[trainer.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Unimodal/trainer.py)</li><li>[utils.py](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Unimodal/utils.py)</li></ul> |
| **📦 [Data/](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Data)** | Repository input storage for datasets and layout illustrations. | <ul><li>[File Location.md](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Data/File%20Location.md)</li><li>[Figure 1.jpg](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Data/Figure%201.jpg) / [2.jpg](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Data/Figure%202.jpg) / [3.jpg](file:///c:/Cloud/Github/Additive-Manufacturing-Modality-Aware-and-Light-Weight-Graph/Data/Figure%203.jpg)</li></ul> |


---

## 🚀 Execution & Verification

### 📋 Setup & Prerequisites
```bash
pip install numpy pandas matplotlib networkx scikit-learn torch-geometric thop
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/pub/whl/cu121
```

### 🏃 Running Pipelines
* **Core Pipeline:** `cd "Sensor fusion" && python Main.py` (Runs the primary GAT-Shapelet model)
* **Parameter Sweep:** `cd "Sensor fusion" && python Main_Shapelet_Optimization.py` (Sweeps shapelet count $k \in [2, ..., 16]$)
* **Model Benchmarking:** `cd "Model Benchmarking" && python Run_benchmarks.py` (Compares against 7 baseline architectures)

---

## 🔒 Citation & Copyright

If you use this work, please cite:

**Title:** Learning composition-sensitive signatures in multi-material PBF-LB: a lightweight, modality-aware, explainable graph-attention sensor fusion framework for in-situ monitoring of graded 316L–CuCrZr alloys
**DOI:** [10.1007/s40964-026-01854-x](https://doi.org/10.1007/s40964-026-01854-x)
**Article Type:** Research
**Journal:** Progress in Additive Manufacturing

**Authors:**
Vigneashwara Pandiyan<sup>a,*</sup>, Antonios Baganis<sup>b,c</sup>, Antti Salminen<sup>a</sup>, Christian Leinenbach<sup>b,c</sup>

*   <sup>a</sup> Digital Manufacturing and Surface Engineering (DMS), Department of Mechanical and Materials Engineering, University of Turku, FI-20014, Turun yliopisto, Finland.
*   <sup>b</sup> Laboratory for Advanced Materials Processing (LAMP), Swiss Federal Laboratories for Materials Science and Technology (Empa)-CH-3602 Thun, Switzerland.
*   <sup>c</sup> Laboratory for Photonic Materials and Characterization, Ecole Polytechnique Fédérale de Lausanne, 1015 Lausanne, Switzerland.

### Citation Format (BibTeX)
```bibtex
@article{pandiyan2026learning,
  title={Learning composition-sensitive signatures in multi-material PBF-LB: a lightweight, modality-aware, explainable graph-attention sensor fusion framework for in-situ monitoring of graded 316L--CuCrZr alloys},
  author={Pandiyan, Vigneashwara and Baganis, Antonios and Salminen, Antti and Leinenbach, Christian},
  journal={Progress in Additive Manufacturing},
  year={2026},
  doi={10.1007/s40964-026-01854-x},
  publisher={Springer}
}
```

Any reuse of this code should be authorized by the code author:
**vpsora** (vigneashwara.solairajapandiyan@utu.fi | vigneashpandiyan@gmail.com).

