# Experiment 11 — Medical Diagnosis with Certainty Factors

Run the complete experiment:

```powershell
python main.py
```

Run deterministic validation checks:

```powershell
python main.py --test
```

## Aim

This experiment implements a medical expert system with four candidate
diagnoses. Rules use certainty factors, findings may be positive (reported
present) or negative (reported absent), and a conjunction is evaluated as the
minimum certainty of its findings. Contributions from rules supporting the
same diagnosis are combined using the signed certainty-factor rule and the
results are ranked as a differential diagnosis.

The consultation driver stops when the leading diagnosis reaches `0.65`.
It compares a fixed intake order, an order based on rule frequency, and a
goal-directed order that prioritises findings used by the current leading
diagnosis.

The exercises perturb every rule certainty factor by ±0.05, ±0.10, and ±0.20
over a deterministic patient population, then measure leading-diagnosis
changes, certainty changes, and lost threshold conclusions. They also report
the mean and maximum number of questions for every consultation strategy.

Only the Python standard library is required.
