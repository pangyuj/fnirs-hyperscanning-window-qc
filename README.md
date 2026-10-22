# Window-wise Quality Control for fNIRS Hyperscanning

Code for the analyses presented in:

**Feasibility of Multimodal fNIRS Hyperscanning and Eye-Tracking During Peer Feedback on Clinical Reasoning**

P. Joo, V. Popov, X. Hu, T. Danciu, C. D. DaSilva, V. Ramaswamy, B. Veremis, and A. F. DaSilva

## Overview

This repository provides the MATLAB code of the analysis pipeline and the validation simulations of our study:
-   **/pipeline**: Contains the window-wise quality control, activation, task-versus-rest connectivity and group statistics, run on dummy hyperscanning recordings.
-   **/simulations**: Contains the validation simulations of window-wise quality control.

## How to Run

1. **Analysis Pipeline:**
- Add the [NIRS Brain AnalyzIR Toolbox](https://github.com/huppertt/nirs-toolbox), [QT-NIRS](https://github.com/lpollonini/qt-nirs) and [FieldTrip](https://www.fieldtriptoolbox.org) to the MATLAB path
- Navigate to `pipeline/` and run `run_pipeline.m` to generate dummy recordings of four dyads, run the pipeline and save Figures 1C, 2 and 3 in `pipeline/results/`
- The dummy recordings only demonstrate the code. To analyze other recordings, replace them in `run_pipeline.m`.

2. **Validation Simulations:**
- Navigate to `simulations/` and run `simulate_dilution_threshold.m` and `simulate_false_effect.m` to reproduce Supplementary Figures 1 and 2
- The `paper_template.mat` file contains the timing and channels of the task recording used in the paper, without fNIRS signals.

## Contact

For questions about the code, please contact Pangyu Joo at pangyuj@umich.edu.
