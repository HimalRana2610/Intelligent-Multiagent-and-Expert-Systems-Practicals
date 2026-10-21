# Experiment 8 — Q-learning

Run the complete experiment with:

```powershell
python main.py
```

Run deterministic validation checks with:

```powershell
python main.py --test
```

## Aim

This experiment trains a tabular Q-learning agent in a 5x5 grid world and
then trains two independent, stateless Q-learners in a cooperative game.
The implementation is deterministic for each seed and uses only the Python
standard library.

## Model and assumptions

The grid starts at `(0, 0)`, has a goal at `(4, 4)` worth `+10`, a pit at
`(2, 2)` worth `-10`, and every other transition costs `-1`. Moves outside
the grid leave the agent in place. The reported run uses
`alpha=0.5`, `gamma=0.9`, `epsilon=0.1`, 500 episodes, and a 100-step
episode limit. The first optimal episode is the first episode after which
the greedy policy reaches the goal in the shortest possible eight moves.

The multi-agent game has actions 0, 1, and 2. Matching either optimal action
(0 or 2) pays 10, matching the safe action (1) pays 2, mismatching the two
optimal actions pays `k`, and all other combinations pay 0. Both learners
observe the common payoff but update only the value of their own selected
action.

## Exercises

The program prints the policy, path length, first optimal episode, early and
late mean returns, the 8-setting sweep over epsilon and alpha (10 seeds per
setting), the zero-exploration `Q=-20` comparison, the payoff matrix, and the
50-seed game sweep for `k=0, -20, -50, -100`. The game sweep is repeated
with both the ordinary averaging update and a best-payoff-ever update.

## Observations

Exploration can keep episodic returns below the greedy policy's performance
even after the policy has become optimal. Independent learners create a
non-stationary environment for one another: an action's observed reward
depends on the other learner's current exploration. This can lead to the
safe equilibrium `(1, 1)` or a mismatch rather than the high-payoff
coordinated equilibria `(0, 0)` and `(2, 2)`.
