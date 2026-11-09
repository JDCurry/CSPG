# Audit of the initial implementation

`initial_implementation/` holds the December 2025 V1 code, unmodified:
`sci_scar_simulator.py`, `axon_growth_simulator.py` and `threshold_analysis.py`.
Their `__main__` blocks write to hard-coded paths from the original environment;
import them as modules (as the scripts below do) rather than running them directly.

| Script | What it shows |
|---|---|
| `audit_threshold.py` | Reruns the initial threshold sweep with fixed seeds. The step from about 3% to about 27% crossing reproduces with or without the condition-dependent axon parameters. In every condition, none of the crossing axons passes through the lesion core. |
| `audit_mmp.py` | Removes the initial code's MMP/TIMP fields. The calibrated core trajectory changes by at most 0.003, but crossing in the threshold sweep at gamma = 0.003 (core 3.52x) goes from 2.9% to 22.9%: MMP leaves CSPG differences of less than 0.001 in healthy tissue, and the direction rule responds to their sign. |
| `audit_controls.py` | Under the initial settings, confining therapy to the lesion gives 0% crossing at every scar level (down to a lesion at baseline). An uninjured cord at CSPG 1.0 or 0.86 also gives 0%. With no lesion, crossing is 1.1% when CSPG is 0.4 everywhere, 31.7% at 0.2 and 78.9% at 0.1. |

Outputs are written to `results/` (`sweep_results.json`, `control_results.json`, `mmp_results.json`). Both scripts use `numpy.random.seed`, as the
initial code does.

The same findings are reproduced with the rewritten code in experiment E5
(`run_experiments.py e5`), which runs the rewritten models at the initial
settings (budget 25, 200-step cap) with whole-grid and lesion-scoped therapy.
