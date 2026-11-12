# Changelog

## 1.1.0 (12 November 2026)
Added
- `11_cnn.py`: 1D CNN on raw 20 ms waveforms for E2 and E3b, 3 seeds and a 10-seed run.
- `12_detection_heldout.py`: E4 thresholds set on the fault-free run from 4 to 9 s and checked from 9 s to the end. The first version's output is kept as `results/detection_results_v1.csv`.
- `13_checks.py`, `14_checks_summary.py` and `results_checks/`: label shuffle, trivial baselines, raw versus referenced features (paired bootstrap), referencing on the shortcut test (H4), and all five E3b models refit over five seeds.
- `docs/checks_plan_2026-11-03.md`: the plan for these checks, written before they ran.
- `tools/run_logger.py`: writes `metrics.json`, `config.json` and `env.txt` for each check run.

Changed
- Cross-mode scores are now reported on the six classes present in both modes. F3 has no usable MPPT windows, so the seven-class score was not on one scale. The seven-class scores stay in `results/experiment_results.csv`.
- E4: the sliding detector finds 12 of 12 locatable faults with no held-out false alarms. Version 1.0.0 said it "flags all 14 fault runs" and that a static threshold "misses 4". The static detector's held-out false alarm rate is 97.3% (MPPT) and 79.9% (IPPT), so its crossings no longer count as detections.
- README rewritten around these results, with the limits stated.

## 1.0.0 (21 October 2026)
First public release: scripts 00 to 07, experiments E1 to E4, figures and tables.
