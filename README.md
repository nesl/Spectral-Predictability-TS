# Spectral Predictability as a Fast Reliability Indicator for Time Series Forecasting Model Selection

Oliver Wang, Pengrui Quan, Kang Yang, Mani Srivastava  
AAAI AIforTS Workshop, 2026

This repository contains all code used in the paper. The structure centers on two main components:  
1. A modified TimeLLM pipeline for forecasting experiments and Omega-conditioned performance visualization  
2. A GiftEval-based pipeline for large-scale analysis of model behavior versus spectral predictability

This dataset is released under the BSD 3-Clause License. See the LICENSE file for details.

---

## Repository Overview

    .
    ├── TimeLLM/                     Modified clone of the official TimeLLM repo
    │   ├── datasets/                Standard time series datasets (user must supply)
    │   ├── scripts/                 Experiment scripts (use testSpectralAll.sh)
    │   ├── results/                 Numerical outputs from all runs
    │   └── results_automate/        Automated postprocessing and plotting
    │       └── graph_spectral.py
    │
    └── gift_eval/
        ├── git_repo/                Clone of the GiftEval repo (user must supply)
        ├── series/                  Arrow-formatted datasets (user must supply)
        ├── merge_gift_results.py
        ├── compute_metrics_fast.py
        └── visualize_modeltype_effects.py

---

## 1. TimeLLM Pipeline

### Datasets
Download and place the standard time series datasets from the Time Series Library into `TimeLLM/datasets/`.  
The directory names must match what the TimeLLM scripts expect.

### Running Experiments

From within `TimeLLM/` run:

    bash scripts/testSpectralAll.sh

Edit arguments inside the script as needed.

### Generating Omega Figures

After all runs complete in the `results/` directory:

    cd TimeLLM/results_automate
    python graph_spectral.py

Figures and processed summaries appear under:

    TimeLLM/results_automate/out/Omega/base/
    TimeLLM/results_automate/out/Omega/mse/

---

## 2. GiftEval Pipeline

### Setup

Inside `gift_eval/`:

1. Clone the GiftEval repository into `git_repo/`
2. Populate a `series/` repository with Arrow datasets from the GiftEval repo.

### Merge Results

    python merge_gift_results.py

Produces:

    merged_gift_results.csv

### Compute Predictability Metrics

    python compute_metrics_fast.py

Produces:

    metrics_summary_wide.csv

Ensure both CSVs have matching dataset and model identifiers before visualization.

### Visualize Model-Type Effects

    python visualize_modeltype_effects.py

Outputs correlation tables and figures under:

    gift_eval/corr_out/
    gift_eval/corr_out/figures/

---

## 3. Reproducing All Figures

1. Run the full TimeLLM experiment pipeline  
2. Run the full GiftEval processing pipeline  
3. Collect figures from:

    TimeLLM/results_automate/out/Omega/
    gift_eval/corr_out/figures/

These reproduce the plots used in the paper.

---

## Citation

    @inproceedings{wang2026spectralpredictability,
      title={Spectral Predictability as a Fast Reliability Indicator for Time Series Forecasting Model Selection},
      author={Wang, Oliver and Quan, Pengrui and Yang, Kang and Srivastava, Mani},
      booktitle={AAAI Workshop on AI for Time Series (AIforTS)},
      year={2026}
    }

---

## License

This dataset is released under the BSD 3-Clause License. See the LICENSE file for details.
