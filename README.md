# Code and data for "[From Tools to Social Partners A Meta-Analysis of Generative AI’s Superiority via CASA and SDT]"

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14986348.svg)](https://doi.org/10.5281/zenodo.18993414)

This repository contains the dataset and automated Python workflows to replicate the meta-analysis presented in the article "[From Tools to Social Partners A Meta-Analysis of Generative AI’s Superiority via CASA and SDT]" submitted to *Humanities and Social Sciences Communications*.

## Repository Structure
- `meta_analysis_results_new_english.csv`: Cleaned and standardized dataset for the meta-analysis.
- `01_effect_size_workflow.py`: Effect size calculation and data QA protocol.
- `02_core_model_fitting.py`: Two-level/Three-level random-effects model fitting.
- `03_moderator_analysis.py`: Automated systematic moderator analysis.
- `04_publication_bias.py`: Funnel plots, Egger's regression, and Trim-and-Fill analysis.
- `05_descriptive_analysis.py`: Study characteristics and descriptive statistics.

## Requirements
Python 3.8+ is required. To install the necessary dependencies, please run:
`pip install pandas numpy scipy matplotlib seaborn pingouin pymare statsmodels scikit-learn`
