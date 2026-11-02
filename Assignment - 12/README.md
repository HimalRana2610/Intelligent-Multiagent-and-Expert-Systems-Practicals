# Experiment 12 — Financial Advisor with Scoring Rules and Hard Constraints

Run the complete experiment:

```powershell
python main.py
```

Run deterministic validation checks:

```powershell
python main.py --test
```

## Aim

This experiment separates a risk-score layer from a hard-constraint layer.
Age, horizon, income stability, dependants, and loss tolerance select a
standard equity/bonds/cash allocation. High-interest debt, an insufficient
emergency reserve, and a horizon of three years or less then override that
allocation when necessary. Every recommendation includes an explanation trace.

Each recommendation is evaluated with 2,000 deterministic Monte Carlo return
paths. The output reports the median final value, fifth percentile, and
probability of finishing below the amount invested, both with and without
constraints.

The exercises compare constrained and unconstrained advice over a client
population and examine the discontinuity caused by score bands versus smooth
interpolation between band centres. Only the Python standard library is
required.
