# CSPG scar and axon-crossing model

Code and result files for *What Sets a Scar-Density Threshold for Axon Crossing?
Implementation and Calibration Sensitivity in a Computational Model of the CSPG
Scar after Spinal Cord Injury* (J. D. Curry, draft, November 2026).

[![DOI](https://zenodo.org/badge/1108262007.svg)](https://doi.org/10.5281/zenodo.17794992)


# Requirements

The model has two parts:

- **`scar_model.py`**: chondroitin sulfate proteoglycan (CSPG) accumulation in a
  circular lesion on a 2-D grid. It is a pointwise ODE: CSPG does not diffuse.
  Therapy is a first-order degradation term with a time window and a spatial
  scope: the lesion only, or the whole grid as a negative control.
- **`axon_model.py`**: a caudally biased random walk with a cost budget, run on
  the day-28 CSPG field. It is vectorized over axons and seeded. Crossing is
  split into crossing through the lesion core and crossing around the scar zone.

## Main findings

- An initial implementation showed a sharp threshold near 3x baseline CSPG. That
  threshold came mainly from therapy that also cleared healthy tissue, combined
  with a growth budget too small to cross an uninjured cord. No crossing axon
  went through the lesion core. Where the step fell also depended on CSPG
  differences of less than 0.001 in healthy tissue, left by an auxiliary MMP
  field (`audit/audit_mmp.py`).
- After the budget is calibrated on an uninjured cord and therapy is confined to
  the lesion, crossing through the lesion declines gradually with CSPG. There is
  no step at 3x.
- The scar level at which through-core crossing halves (S50) depends on the
  assumed budget, on lesion size and on the distance axons must grow. Once
  matched to the simulated baseline, it can be predicted from path costs.
- The calibrated scar time course is not identifiable: a bounded refit with very
  different phase amplitudes fits the stylized target better. The same
  fractional cut in a production phase has a several-fold different effect
  under the two parameter sets, though the ranking of the arms tested is the
  same.

See `CHANGES.md` for exactly what changed from the initial implementation, and
`audit/` for the diagnosis.

## Reproduce

Requires Python 3.10+ with `numpy`, `scipy` and `matplotlib` (`pip install -r requirements.txt`).
Run from this directory:

```bash
python calibrate_scar.py          # scar calibration and identifiability -> results/scar_calibration.json  (~5 s)
python run_experiments.py         # experiments E1-E6 -> results/e*.json                                 (~10 min on 2 cores)
python make_figures.py            # figures/fig1-fig6 (.pdf and .png)
python summarize_results.py > results/summary.txt   # every number cited in the paper
```

Experiments can be run individually, e.g. `python run_experiments.py e2 e5`.
E6 reads `results/scar_calibration.json`, so run `calibrate_scar.py` first.

| Experiment | What it does |
|---|---|
| E1 `e1_uninjured_calibration` | Crossing in an uninjured cord vs budget, with and without the initial 200-step cap |
| E2 `e2_crossing_vs_scar` | Crossing vs lesion CSPG; scar lowered by lesion-scoped therapy or by reduced production |
| E3 `e3_budget_sensitivity` | S50 vs budget on idealized scar fields, simulated and predicted from path costs |
| E4 `e4_lesion_size` | S50 vs lesion size, with start/target rows that move with the lesion and with them fixed |
| E5 `e5_therapy_scope` | Lesion-scoped vs whole-grid therapy, at initial and calibrated settings |
| E6 `e6_treatments` | Intervention arms with identical axon parameters, under two scar parameter sets (production changes are fractional) |

Every random draw uses a `numpy` Generator seeded from (base seed, experiment,
condition, replicate). Results are reproducible on the same numpy version;
results produced with a different numpy version may differ in the last digit.

## Files

```
scar_model.py           CSPG accumulation model
axon_model.py           cost-budget axon growth model
calibrate_scar.py       calibration, extended grid, bounded least-squares refit
run_experiments.py      E1-E6
make_figures.py         all figures, from results/*.json only
summarize_results.py    every number in the manuscript, from results/*.json
results/                JSON outputs and summary.txt
figures/                fig1-fig6 (.pdf, .png)
audit/                  the initial implementation (unmodified) and the scripts that diagnosed it
CHANGES.md              what changed from the initial implementation, and why
```

## License

MIT (see `LICENSE`).
