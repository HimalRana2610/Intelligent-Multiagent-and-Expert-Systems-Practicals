# Experiment 10 — Ethical Considerations

Run the complete audit:

```powershell
python main.py
```

Run deterministic validation:

```powershell
python main.py --test
```

The program generates reproducible applicant populations and audits a
decision-tree rule base induced from historical decisions. It reports:

- base rate, selection rate, true-positive rate, false-positive rate, and
  precision for both groups, including group gaps;
- agreement with historical decisions and accuracy against deserved outcomes;
- equivalence-class privacy statistics before and after generalisation;
- mean and maximum explanation length plus a complete applicant trace;
- mitigation comparisons for removing protected/proxy features and training on
  deserved outcomes;
- the accuracy, privacy, and explanation effects of four generalisation levels.

Only the Python standard library is required.
