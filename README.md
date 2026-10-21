# Leakage-Aware Fault Detection on GPVS-Faults

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23537370.svg)](https://doi.org/10.5281/zenodo.23537370)

Published work on the GPVS-Faults dataset (a grid-connected PV inverter recorded under seven fault types) often reports near-perfect fault classification. This repo checks whether those scores come from the faults or from something easier. In short, the random-split protocol mostly teaches a model to recognize which recording a window came from. When the model has to work on a different operating mode, macro-F1 falls to between 0.15 and 0.52.

## Data
GPVS-Faults by Bakdi, Guichi, Mekhilef and Bounoua (2020), [doi:10.17632/n76t439f65.1](https://doi.org/10.17632/n76t439f65.1), CC BY 4.0. It has 16 runs (fault F0 to F7, each in MPPT and IPPT mode) with 13 signals sampled at 10 kHz. The data are not included here. Download `CSV_Files.zip` from Mendeley Data and unzip it to `data/raw/gpvs_csv/CSV_Files/`.

## Pipeline
| Script | Step |
|---|---|
| `00_explore.py`, `01_overview_plots.py` | Check sampling rate, run lengths and signals |
| `02_features.py` | Cut each run into 20 ms windows (one 50 Hz cycle) and compute 52 statistical and 5 physical features |
| `03_onset.py` | Estimate when each fault starts, since the dataset does not publish onsets |
| `04_experiments.py` | Experiments E1 to E4 below, five seeds, five model families |
| `05_figures.py`, `06_tables.py` | Figures and the tables in `results/tables.md` |
| `07_sensitivity.py` | Sensitivity to onset error, guard band and baseline length |

## What the experiments show (macro-F1, mean over 5 seeds)
- **E1, common protocol** (every window labeled with its run's fault, random split): random forest 0.999 to 1.000.
- **E2, shortcut probe:** the same protocol using only windows recorded *before* any fault began still scores 1.000. The model separates runs that have no fault in them yet, so it is recognizing the recording rather than the fault.
- **E3a, onset-based labels, random split:** LightGBM 0.999, random forest 0.991, logistic regression 0.812.
- **E3b, cross-mode** (train on MPPT runs, test on IPPT runs, and the reverse): 0.15 to 0.52 across models. Features referenced to each run's own fault-free baseline help most, with LightGBM at 0.524 for IPPT to MPPT. Faults F1 and F3 are almost never recognized across modes.
- **E4, unsupervised detection** (Hotelling T² on baseline-referenced features): a sliding-baseline detector flags all 14 fault runs, most within 0.4 s of onset. A static threshold misses 4.

Full tables, per-class recall and bootstrap confidence intervals are in `results/tables.md`. Figures are in `figures/`.

## Reproduce
```bash
pip install -r requirements.txt
python code/02_features.py && python code/03_onset.py && python code/04_experiments.py
python code/05_figures.py && python code/06_tables.py && python code/07_sensitivity.py
```

## Limits
- One laboratory test rig. The results say nothing yet about field inverters.
- Fault onsets are estimated, not ground truth (see `07_sensitivity.py`). F4 has no locatable onset and is left out of E3.
- The F2L run oscillates strongly from the start of the recording, which inflates its pre-onset false alarms.
- Public data only. No employer data, equipment or procedures were used.

Author: Khandoker Samiul Hoque (ORCID [0009-0008-0756-2395](https://orcid.org/0009-0008-0756-2395)). Code license: MIT. Manuscript in preparation.

## Archived release
Version 1.0.0 is archived on Zenodo: [doi:10.5281/zenodo.23537371](https://doi.org/10.5281/zenodo.23537371). To cite the latest version instead, use [doi:10.5281/zenodo.23537370](https://doi.org/10.5281/zenodo.23537370).
