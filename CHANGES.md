# Changes from the initial implementation (November 2026)

The initial implementation (December 2025) is kept unmodified in
`audit/initial_implementation/`. This file records what changed and why. Each
change to the model is tested in the manuscript or in `audit/`.

## Removed

- **`sci_scar_simulator_v2.py`, `test_zinc20_compounds.py`, `zinc20_data_converter.py`**
  (TIMP2/MMP extension, TIMP2 genotype and ZINC20 compound testing). This
  material is out of scope for this paper. Two problems were also found:
  - As released, `sci_scar_simulator_v2.py` became numerically unstable and
    produced NaN values within a 28-day run, for the control, A;A and chABC
    conditions alike. The likely cause is the explicit MMP diffusion step,
    whose D*dt = 0.4 exceeds the 0.25 stability limit of that scheme.
  - The V2 equations in the December 2025 draft (MMP *driving* CSPG
    production) do not match this code (MMP *degrading* CSPG), and the draft's
    parameter table does not match the code's defaults. The code that
    produced the draft's V2 numbers is not in this repository.
- **Auxiliary MMP and generic TIMP fields in the V1 model.** MMP was set only
  at injury (0.5 in the lesion core), diffused and decayed within hours; TIMP
  acted only on MMP. Removing both fields changed the calibrated core trajectory
  by at most 0.003. It did change axon crossing in the initial threshold sweep:
  MMP left CSPG in healthy tissue very slightly nonuniform (0.1238-0.1240 at
  gamma = 0.003), and because the direction rule responds to the sign of a CSPG
  difference, crossing at that dose went from 2.9% with MMP to 22.9% without
  (`audit/audit_mmp.py`). This moves the initial step but does not remove it.

## Scar model (`sci_scar_simulator.py` -> `scar_model.py`)

| Initial | Now | Why |
|---|---|---|
| Therapy term `gamma * S` applied to every grid point | Therapy applies to the scar zone; whole-grid kept as an explicit negative control (`scope="global"`) | Whole-grid therapy drove healthy-tissue CSPG to the 0.1 floor. That, not lesion density, produced the apparent 3x threshold |
| Self-limitation `delta * S^2` applied everywhere | Applied only in the scar zone | Tissue outside the lesion of an injured, untreated cord drifted from 1.0 to 0.86 over 28 days |
| `add_chABC_treatment` existed but was unused; "early" and "late" chABC were continuous doses from day 0 | Each therapy has an explicit time window | The labels described timing the code did not implement |
| Production zone from 5 iterations of binary dilation (diamond-shaped) | Euclidean disk of radius core + 5 | Simpler to describe; core dynamics are identical |
| Described as reaction-diffusion | Described as a pointwise ODE | S never diffused |

The calibrated parameter values are unchanged. `calibrate_scar.py` adds an
extended grid and a bounded least-squares refit (0.05-20x each reference value),
which show that the parameters are not identifiable from the stylized target.
The calibration itself runs on a 32x32 grid (dt 1 h, core radius 5), as in the
initial implementation; the fit is the same to three decimals on the 64x64 grid.

## Axon model (`axon_growth_simulator.py` -> `axon_model.py`)

| Initial | Now | Why |
|---|---|---|
| Budget 25 and a 200-step cap | Budget 200 (calibrated: 93.8% crossing of an uninjured cord), cap 2,000 (never binds) | With the initial settings, 0.0% of axons crossed an uninjured cord. The 200-step cap alone held uninjured crossing to about 50% |
| Axon parameters varied by condition (chABC arms: budget 28; combined arm: budget 30, growth rate 0.96, inhibition 0.25; TGF-beta and neuroprotective arms: 26; the threshold sweep switched parameters at scar targets 3.5 and 2.5) | Identical for all conditions | Confounded treatment comparisons. Holding them fixed did not remove the threshold (see `audit/`) |
| Global `np.random`, no seeds | Explicit, seeded `numpy` Generators | Reproducibility |
| Per-axon Python loop | Vectorized over axons; validated against the initial code (crossing rates agree within sampling error) | Speed (about 40x) |
| Crossing rate only | Crossing split into through-core, through-penumbra and around-the-zone; moves inside the scar zone recorded | All crossing in the initial threshold result was around the lesion |

The walk rules themselves (extension probability, direction weights, cost) are unchanged.

## Analysis and figures

- `threshold_analysis.py` plotted panels A and B of the threshold figure from
  hard-coded scar values with treatment names attached, including an
  "Aggressive combined" arm that was never simulated. All figures are now built
  by `make_figures.py` from saved result files only.
- The threshold location is now defined directly (S50: the scar level at
  which through-core crossing halves relative to the lesion at baseline),
  with bootstrap intervals. A path-cost calculation predicts it.
- Experiments E1-E6 (`run_experiments.py`) replace the earlier scripts.
- E4 compares lesion sizes twice: with start and target rows that move with
  the lesion (so the whole path lengthens) and with them fixed.
- In E6, production-targeting arms are multipliers on the parameter set in use,
  so each arm makes the same relative change under the reference and
  least-squares parameters.
