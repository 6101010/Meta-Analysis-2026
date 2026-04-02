# From Tools to Social Partners: A Meta-Analysis of Generative AI’s Superiority via CASA and SDT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18993414.svg)](https://zenodo.org/records/18993414)

## Project Overview
This repository contains the dataset, search protocols, and analytical code for the systematic review and meta-analysis titled: **"From Tools to Social Partners: A Meta-Analysis of Generative AI’s Superiority via CASA and SDT"**. 

This study evaluates the effectiveness of AI-based educational interventions on academic achievement, exploring the psychological mechanisms driving these effects by integrating the Computers as Social Actors (CASA) paradigm with Self-Determination Theory (SDT).

## Repository Structure

The repository is organized into data/documentation and reproducible analysis scripts.

### Data & Documentation
* `meta_analysis_results_new_english.csv`: The primary extracted dataset including study characteristics and statistical variables.
* `Search Strategies for Meta-Analysis`: Detailed Boolean search queries applied across Web of Science and Scopus.
* `Inclusion and Exclusion Criteria`: The predefined criteria used for screening studies.
* `PRISMA.jpg`: The PRISMA 2020 flow diagram illustrating the study selection process.
* `PRISMA_2020_checklist.pdf`: The completed PRISMA 2020 reporting checklist indicating where each item is addressed.

### Analysis Scripts (Python)
The quantitative synthesis was conducted using Python. The scripts are numbered in the order of the analytical workflow:
* `01_effect_size_workflow.py`: Calculates standardized effect sizes (Hedges' g) with small-sample bias adjustments.
* `02_core_model_fitting.py`: Fits the multilevel random-effects models (two-level and three-level) to synthesize the overall effect and estimate variance components.
* `03_moderator_analysis.py`: Performs meta-regression and subgroup analyses for key moderators (e.g., educational stage, AI type, intervention duration, sample size).
* `04_publication_bias.py`: Implements publication bias diagnostics, including funnel plot generation, Egger's regression test (with clustered standard error adjustments), and trim-and-fill analysis.
* `05_descriptive_analysis.py`: Generates descriptive statistics and summaries of the included studies.

## How to Use
To replicate the findings:
1. Clone this repository to your local machine.
2. Ensure you have the required Python libraries installed (e.g., `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib`).
3. Run the scripts sequentially from `01` to `05`. Ensure the working directory is set to the folder containing `meta_analysis_results_new_english.csv`.

## Data Availability
All materials, datasets, and supporting documentation underpinning this research are publicly available to support the transparent evaluation of the findings. A static release of this repository is archived on Zenodo at: [https://zenodo.org/records/18993414](https://zenodo.org/records/18993414).

## License
This project is open-sourced under the MIT License. You are free to use, modify, and distribute the code and data, provided that appropriate credit is given to the original study.
