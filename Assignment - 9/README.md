# Experiment 9 — Knowledge Acquisition

Run the complete experiment with:

```powershell
python main.py
```

Run deterministic validation checks with:

```powershell
python main.py --test
```

## Aim

This experiment audits a rule base written after interviewing an expert and
induces a second rule base from solved cases using ID3-style decision-tree
learning. It reports uncovered cases, contradictory rule firings, first-match
accuracy, the induced tree, production rules read from its leaves, and
agreement between the two rule bases.

The program also repeats the induction with 0%, 5%, 10%, and 20% corrupted
labels, comparing ordinary induction with a minimum split size of four, and
averages results over case-base sizes 8, 16, 24, 40, and 80.

The implementation uses only the Python standard library and fixed seeds, so
the complete output and self-tests are reproducible.
