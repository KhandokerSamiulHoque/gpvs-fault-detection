---
type: analysis-plan
title: "Paper B added checks, analysis plan"
description: Plan for the robustness checks added to Paper B, frozen before any of them ran.
status: stable
---
# Analysis plan for added checks. Frozen on 2026-11-03, before these checks ran.

This covers only the checks below. The main Paper B analysis was settled earlier with results visible and is not
preregistered. Window labels, onsets, splits, guard band, models and settings are the ones in `04_experiments.py` and
`11_cnn.py`, unchanged.

| Item | Value |
|---|---|
| Primary metric | Macro-F1 |
| Data | GPVS-Faults CSV release, `paper_b/results/windows.csv`, `onsets.csv` |
| Alpha / correction | 0.05 / Holm within each family below |
| Seeds | 0 to 4 for classical checks; 0 to 9 for the CNN (seeds 0 to 2 already exist and are kept) |
| Minimum meaningful effect | 0.05 macro-F1 |

## Hypotheses and what would falsify them
- **H1, sanity.** With training labels shuffled, tested on true labels, macro-F1 falls to about chance. RF on E2 (random
  and time-blocked) and LightGBM on E3b (raw and delta, both directions). Falsified if any mean is above 0.25.
- **H2, trivial baseline.** `DummyClassifier` (most frequent, stratified) on the same E2 and E3b splits gives the measured
  floor. Falsified if LightGBM delta in E3b (0.483 and 0.524) is not at least 0.10 above the better dummy in that direction.
- **H3, raw versus delta.** Delta beats raw for LightGBM and RF in both directions. Test: paired block bootstrap on the
  stored seed-0 E3b test predictions, 1 s blocks across runs, 5000 resamples, same blocks for both feature sets, fixed
  7-class macro-F1. Report difference, 95% CI and two-sided p for all 10 model-direction pairs, Holm over the 10.
  Falsified if any of the four tree-ensemble pairs has a Holm p of 0.05 or more.
- **H4, delta on the shortcut.** On E2 windows from 3 s onward, RF with delta features scores at least 0.05 below RF with
  the same raw statistics, under the time-blocked split. Falsified in a mode if the drop is under 0.05. Report the delta
  score against the measured floor either way, since a drop can still leave the shortcut in place.
- **H5, CNN.** With 10 seeds: (a) delta beats raw in both directions (paired Wilcoxon over seeds, Holm over 2); (b) CNN
  delta versus LightGBM delta at 0.483 and 0.524 (one-sample Wilcoxon, two-sided, Holm over 2). (a) is falsified if
  either adjusted p is 0.05 or more. (b) has no predicted direction and is reported as found.

## Known limits, stated in advance
- Each test class comes from one experiment. The bootstrap covers variation between windows, not between experiments.
- H3 uses seed 0 only because those are the stored predictions. LightGBM has SD 0.000 over seeds; RF does not.
- None of this adds a second dataset, so the generalization check stays open.

## Deviations after freezing (dated, with reason)
- **2026-11-03, before any full run; only a one-seed smoke test had run.** The smoke test showed cross-mode macro-F1 is
  not on one scale. F3 has 45 usable windows in IPPT and 0 in MPPT (F3M ends 0.5 s after onset and the guard band drops
  them). scikit-learn's default macro-F1 averages over classes in the labels or predictions, so MPPT to IPPT averages
  7 classes with F3 untrainable, and IPPT to MPPT averages 6, or 7 if a model predicts F3. Changes:
  (1) added a secondary metric, macro-F1 over the six classes present in both modes (normal, F1, F2, F5, F6, F7);
  (2) H3 runs on both the frozen 7-class score and the six-class score;
  (3) added H6: refit all five E3b models over five seeds and log both scores, which must also reproduce the published
  default values;
  (4) H5 reruns all 10 CNN seeds, because the stored seeds 0 to 2 lack the six-class score, and must reproduce the stored
  values. H5(b) compares on the six-class score, since the default score is not comparable between models.
