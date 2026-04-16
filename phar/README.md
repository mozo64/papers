# PHAR: Post-hoc Attribution Rules

This directory contains the code necessary to reproduce the experiments from our manuscript on extracting and fusing semi-factual intervals for Time Series Classification (https://arxiv.org/abs/2508.01687).

## Repository Structure
* `model_server.py` & `model_manager.py` - Flask-based API for handling predictions during the perturbation steps.
* `UCR-explainers-lime-shap-anchor.ipynb` - Generation of baseline attributions (Note: pre-computed artifacts can be downloaded from our Zenodo repository https://zenodo.org/records/18892817).
* `UCR-num2rules-optuna.ipynb` - The core PHAR extraction pipeline translating attributions into intervals using Optuna.
* `UCR-fusions-results.ipynb` - Execution of the formal fusion strategies (Lasso, Weighted, etc.).
* `results/` - Contains the final `.csv` files used for the Friedman and Nemenyi statistical tests in the paper.

## Requirements
* Python 3.8+
* TensorFlow 2.x
* Optuna
* scikit-posthocs