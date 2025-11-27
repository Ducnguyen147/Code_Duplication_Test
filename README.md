# Code Duplication Test

This repository accompanies the master thesis *“A Comparative Study of Multi-Language Code-Clone Detectors and a Hybrid Transformer-Based Semantic Approach”*. It provides a **synthetic benchmark suite** of Python, Java, JavaScript and C++ files along with all the scripts used to run the experiments and reproduce the results discussed in the thesis. The goal is to make the experiments **transparent and reproducible** by publishing the dataset, tool wrappers and analysis scripts openly.

## Contents

The repository is organised as follows:

| Directory or file | Purpose |
| --- | --- |
| `dataset/` | Contains the four-language **artificial “mix” benchmark suite**. Each subdirectory (`python/`, `java/`, `javascript/`, `cpp/`) holds synthetic files used to evaluate clone detectors. |
| `sweep/` | Parameter-sweep scripts for each detector. These scripts iterate over tool-specific parameters (e.g. Dolos k/w values, thresholds) and write out metrics tables. |
| `tools_execution/` | Command-line wrappers and helper scripts for running third-party tools (Dolos, JPlag, PMD-CPD, jscpd) and parsing their outputs. They normalise the outputs into a unified CSV format for evaluation. |
| `tables/` | Generated CSV tables containing precision/recall/F1 scores for different tools, languages and thresholds. These tables underpin the figures and tables in the thesis. |
| `visualizations/` | Figures and plots used in the Thesis. |