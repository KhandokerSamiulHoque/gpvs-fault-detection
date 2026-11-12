# Leakage-Aware Fault Detection on GPVS-Faults

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23537370.svg)](https://doi.org/10.5281/zenodo.23537370)

Published work on the GPVS-Faults dataset (a grid-connected PV inverter recorded under seven fault types) often reports near-perfect fault classification. This repo checks where those scores come from. A random forest separates the runs at macro-F1 1.000 using only windows recorded before the estimated fault onset, so the model is recognizing the recording. Subtracting each run's own fault-free baseline helps across operating modes, but the shortcut stays: the same test still scores 0.897 (MPPT) and 0.961 (IPPT).

## Data
GPVS-Faults by Bakdi, Guichi, Mekhilef and Bounoua (2020), [doi:10.17632/n76t439f65.1](https://doi.org/10.17632/n76t439f65.1), CC BY 4.0. It has 16 runs (fault F0 to F7, each in MPPT and IPPT mode) with 13 signals sampled at 10 kHz. The data are not included here. Download `CSV_Files.zip` from Mendeley Data and unzip it to `data/raw/gpvs_csv/CSV_Files/`.

## Pipeline
| Script | Step |
|---|---|
| `00_explore.py`, `01_overview_plots.py` | Check sampling rate, run lengths and signals |
| `02_features.py` | Cut each run into 20 ms windows (one 50 Hz cycle), compute 52 statistical and 5 physical features |
| `03_onset.py` | Estimate when each fault starts, since the dataset does not publish onsets |
| `04_experiments.py` | Experiments E1 to E4, five seeds, five model families |
| `11_cnn.py` | 1D CNN on raw 20 ms waveforms for E2 and E3b |
| `12_detection_heldout.py` | E4 detection with thresholds set on the fault-free run from 4 to 9 s and checked from 9 s to the end |
| `07_sensitivity.py` | Sensitivity to onset error, guard band and baseline length |
| `13_checks.py`, `14_checks_summary.py` | Label shuffle, trivial baselines, raw versus referenced features, referencing on the shortcut test |
| `05_figures.py`, `06_tables.py` | Figures and `results/tables.md` |

The added checks were planned in [docs/checks_plan_2026-11-03.md](docs/checks_plan_2026-11-03.md) before they ran, but after the main results were known. The main analysis was not preregistered.

## Results
Cross-mode scores below are macro-F1 over the six classes present in both modes (normal, F1, F2, F5, F6, F7). F3 has no usable MPPT windows, so the default seven-class score is not on one scale across directions. Those scores remain in `results/experiment_results.csv`.

| Test | Result |
|---|---|
| E1, every window labeled with its run's fault, random split | Random forest 0.999 to 1.000 |
| E2, pre-onset windows only, labeled by run | Random forest 1.000, CNN 0.969 to 0.998 |
| E2 with baseline-referenced features, time-blocked split, from 3 s on | Random forest 0.897 (MPPT), 0.961 (IPPT). Floor from trivial baselines 0.142 to 0.146 |
| E3b, train MPPT, test IPPT | LightGBM referenced 0.564, CNN referenced 0.486 |
| E3b, train IPPT, test MPPT | LightGBM referenced 0.524, CNN referenced 0.717 |
| E4, sliding-baseline detector | 12 of 12 locatable faults, delay up to 0.88 s, no false alarms in the held-out fault-free span |
| E4, static detector | Held-out false alarm rate 97.3% (MPPT), 79.9% (IPPT), so its crossings don't count as detections |

Referencing beat raw features for all four tree-ensemble comparisons (Holm p at most 0.0032). This test covers variation between windows only, because each class comes from one experiment per mode. Neither LightGBM nor the CNN wins in both directions.

Full tables are in `results/tables.md`, per-run check outputs in `results_checks/`, figures in `figures/`.

## Reproduce
```bash
pip install -r requirements.txt
python code/02_features.py && python code/03_onset.py && python code/04_experiments.py
python code/12_detection_heldout.py && python code/07_sensitivity.py
python code/11_cnn.py && python code/11_cnn.py 0 1 2 3 4 5 6 7 8 9
python code/13_checks.py && python code/14_checks_summary.py
python code/05_figures.py && python code/06_tables.py
```
On one CPU with no GPU, `13_checks.py` took 22 minutes and the two `11_cnn.py` runs took 26 minutes. `python code/13_checks.py --smoke` runs a short version.

### Check the 0.897 and 0.961 figures
These come from check H4 and need version 1.1.0 or later. Version 1.0.0 does not have `13_checks.py`.
```bash
python code/02_features.py && python code/03_onset.py && python code/04_experiments.py
python code/13_checks.py && python code/14_checks_summary.py
```
In the H4 table that `14_checks_summary.py` prints, the `H4_E2_blocked` rows for `rf_delta` give 0.897 (MPPT) and 0.961 (IPPT), each the mean of five seeds. Everything above, plus `11_cnn.py`, was rerun on 12 November 2026 from a clean copy of this repo. All 260 check runs, the summary files and the CNN results matched the stored files exactly.

## Limits
- One laboratory test rig and one dataset. The results say nothing yet about field inverters.
- Fault onsets are estimated, not ground truth. Bakdi et al. state faults start around 7 to 8 s, and 9 of the 12 locatable onsets here are later. Windows called fault-free may include faulty data in 8 runs. This has not been tested.
- F4 has no locatable onset and is left out of E3.
- Each mode has one fault-free run, and F2L alarms through its whole pre-onset part.
- Which features carry the shortcut left after referencing has not been tested.
- Raszka and Grycz independently reached the same core shortcut result in a [Zenodo deposit](https://doi.org/10.5281/zenodo.23646443) dated 25 October 2026.
- Public data only. No employer data, equipment or procedures were used.

Author: Khandoker Samiul Hoque (ORCID [0009-0008-0756-2395](https://orcid.org/0009-0008-0756-2395)). Code license: MIT. Manuscript in preparation.

## Versions
| Version | Date | Contents |
|---|---|---|
| 1.1.0 | 12 Nov 2026 | Current analysis: scripts 11 to 14, held-out detection, the added checks and the results above |
| 1.0.0 | 21 Oct 2026 | First release: scripts 00 to 07 only. Its README gives an older detector result, and it cannot reproduce the CNN, the held-out detection or the H4 figures |

Each version is archived on Zenodo. Version 1.0.0 is [doi:10.5281/zenodo.23537371](https://doi.org/10.5281/zenodo.23537371). To cite the latest version, use the concept DOI [10.5281/zenodo.23537370](https://doi.org/10.5281/zenodo.23537370), which always points to the newest release. See [CHANGELOG.md](CHANGELOG.md) for details.
