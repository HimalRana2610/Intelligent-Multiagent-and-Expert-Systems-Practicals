"""Experiment 8: single-agent and independent multi-agent Q-learning.

Run ``python main.py`` for the experiment, exercises, and observations.
Run ``python main.py --test`` for deterministic validation checks.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, List, Sequence, Tuple


Action = str
State = Tuple[int, int]
ACTIONS: Tuple[Action, ...] = ("up", "right", "down", "left")
DELTAS: Dict[Action, Tuple[int, int]] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
GRID_SIZE = 5
START: State = (0, 0)
GOAL: State = (4, 4)
PIT: State = (2, 2)


@dataclass(frozen=True)
class SingleRun:
    q_values: Dict[State, Dict[Action, float]]
    returns: Tuple[float, ...]
    greedy_path: Tuple[State, ...]
    path_length: int | None
    first_optimal_episode: int | None


@dataclass(frozen=True)
class SweepRow:
    epsilon: float
    alpha: float
    optimal_runs: int
    median_first_optimal: float | None
    mean_last_fifty: float


@dataclass(frozen=True)
class GameRun:
    values_a: Tuple[float, float, float]
    values_b: Tuple[float, float, float]
    actions: Tuple[int, int]
    outcome: str


def move(state: State, action: Action) -> State:
    row, column = state
    delta_row, delta_column = DELTAS[action]
    return (
        max(0, min(GRID_SIZE - 1, row + delta_row)),
        max(0, min(GRID_SIZE - 1, column + delta_column)),
    )


def transition(state: State, action: Action) -> Tuple[State, int, bool]:
    next_state = move(state, action)
    if next_state == GOAL:
        return next_state, 10, True
    if next_state == PIT:
        return next_state, -10, True
    return next_state, -1, False


def greedy_action(values: Dict[Action, float]) -> Action:
    return max(ACTIONS, key=lambda action: (values[action], -ACTIONS.index(action)))


def choose_action(values: Dict[Action, float], epsilon: float, rng: random.Random) -> Action:
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    return greedy_action(values)


def greedy_path(q_values: Dict[State, Dict[Action, float]],
                max_steps: int = GRID_SIZE * GRID_SIZE) -> Tuple[State, ...]:
    state = START
    path = [state]
    for _ in range(max_steps):
        if state in (GOAL, PIT):
            break
        state = move(state, greedy_action(q_values[state]))
        path.append(state)
        if state in (GOAL, PIT):
            break
    return tuple(path)


def is_optimal_policy(q_values: Dict[State, Dict[Action, float]]) -> bool:
    path = greedy_path(q_values, max_steps=8)
    return path[-1] == GOAL and len(path) - 1 == 8


def train_single(
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 0,
    initial_value: float = 0.0,
    max_steps: int = 100,
) -> SingleRun:
    """Train tabular Q-learning in the fixed 5x5 grid world."""
    if episodes < 1 or not 0 < alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError("episodes must be positive, alpha in (0, 1], gamma in [0, 1]")
    rng = random.Random(seed)
    q_values = {
        (row, column): {action: initial_value for action in ACTIONS}
        for row in range(GRID_SIZE)
        for column in range(GRID_SIZE)
    }
    returns: List[float] = []
    first_optimal: int | None = None
    for episode in range(1, episodes + 1):
        state = START
        episode_return = 0.0
        for _ in range(max_steps):
            action = choose_action(q_values[state], epsilon, rng)
            next_state, reward, terminal = transition(state, action)
            target = reward
            if not terminal:
                target += gamma * max(q_values[next_state].values())
            q_values[state][action] += alpha * (target - q_values[state][action])
            episode_return += reward
            state = next_state
            if terminal:
                break
        returns.append(episode_return)
        if first_optimal is None and is_optimal_policy(q_values):
            first_optimal = episode
    path = greedy_path(q_values)
    return SingleRun(
        q_values, tuple(returns), path,
        len(path) - 1 if path[-1] == GOAL else None,
        first_optimal,
    )


def payoff(action_a: int, action_b: int, penalty: int) -> int:
    if action_a == action_b and action_a in (0, 2):
        return 10
    if action_a == action_b == 1:
        return 2
    if {action_a, action_b} == {0, 2}:
        return penalty
    return 0


def train_game(
    penalty: int = -50,
    rounds: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    seed: int = 0,
    best_ever: bool = False,
) -> GameRun:
    """Train two stateless, independent learners on the same payoff."""
    if penalty > 0 or rounds < 1 or not 0 < alpha <= 1:
        raise ValueError("penalty must be non-positive, rounds positive, alpha in (0, 1]")
    rng = random.Random(seed)
    values = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for _ in range(rounds):
        actions = []
        for agent_values in values:
            if rng.random() < epsilon:
                actions.append(rng.randrange(3))
            else:
                actions.append(max(range(3), key=lambda index: (agent_values[index], -index)))
        reward = payoff(actions[0], actions[1], penalty)
        for index, action in enumerate(actions):
            if best_ever:
                values[index][action] = max(values[index][action], reward)
            else:
                values[index][action] += alpha * (reward - values[index][action])
    actions = tuple(
        max(range(3), key=lambda index: (agent_values[index], -index))
        for agent_values in values
    )
    if actions[0] == actions[1] and actions[0] in (0, 2):
        outcome = "optimal joint action"
    elif actions == (1, 1):
        outcome = "safe one"
    else:
        outcome = "mismatch"
    return GameRun(tuple(values[0]), tuple(values[1]), actions, outcome)


def run_sweep() -> Tuple[SweepRow, ...]:
    rows: List[SweepRow] = []
    for epsilon in (0.0, 0.05, 0.1, 0.3):
        for alpha in (0.1, 0.5):
            runs = [
                train_single(alpha=alpha, epsilon=epsilon, seed=1000 + seed)
                for seed in range(10)
            ]
            first = [run.first_optimal_episode for run in runs
                     if run.first_optimal_episode is not None]
            rows.append(SweepRow(
                epsilon, alpha, len(first),
                float(median(first)) if first else None,
                mean(mean(run.returns[-50:]) for run in runs),
            ))
    return tuple(rows)


def print_policy(run: SingleRun) -> None:
    arrows = {"up": "^", "right": ">", "down": "v", "left": "<"}
    print("Greedy policy (^ up, > right, v down, < left; G goal, X pit):")
    for row in range(GRID_SIZE):
        cells = []
        for column in range(GRID_SIZE):
            state = (row, column)
            if state == GOAL:
                cells.append("G")
            elif state == PIT:
                cells.append("X")
            else:
                cells.append(arrows[greedy_action(run.q_values[state])])
        print(" ".join(cells))


def print_payoff_matrix(penalty: int) -> None:
    print("Payoff matrix (rows=A, columns=B; actions 0, 1, 2):")
    for action_a in range(3):
        print("  " + " ".join(f"{payoff(action_a, action_b, penalty):>4}"
                              for action_b in range(3)))


def self_test() -> None:
    run = train_single(seed=7)
    assert run.path_length == 8
    assert run.greedy_path[-1] == GOAL
    assert len(run.returns) == 500
    assert train_single(episodes=2, epsilon=0, seed=1).first_optimal_episode is None
    assert payoff(0, 0, -50) == 10
    assert payoff(1, 1, -50) == 2
    assert payoff(0, 2, -50) == -50
    assert train_game(seed=3, best_ever=True).outcome in {
        "optimal joint action", "safe one", "mismatch",
    }
    assert len(run_sweep()) == 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="run validation checks only")
    args = parser.parse_args()
    self_test()
    if args.test:
        print("self-test passed")
        return

    run = train_single()
    print("Experiment 8: Q-learning in a 5x5 grid world")
    print(f"Parameters: alpha=0.5, gamma=0.9, epsilon=0.1, episodes=500, seed=0")
    print_policy(run)
    print(f"Greedy path: {run.greedy_path}")
    print(f"Greedy path length: {run.path_length}")
    print(f"First optimal-policy episode: {run.first_optimal_episode or 'never'}")
    print(f"Mean return, first 50 episodes: {mean(run.returns[:50]):.2f}")
    print(f"Mean return, last 50 episodes: {mean(run.returns[-50:]):.2f}")

    print("\nExercise 1: alpha/epsilon sweep (10 seeds per setting)")
    print("epsilon alpha optimal/10 median-first-optimal mean-last-50")
    for row in run_sweep():
        median_text = f"{row.median_first_optimal:.1f}" if row.median_first_optimal else "-"
        print(f"{row.epsilon:>7.2f} {row.alpha:>5.1f} "
              f"{row.optimal_runs:>10} {median_text:>19} {row.mean_last_fifty:>14.2f}")
    negative_initial = [
        train_single(epsilon=0, initial_value=-20, seed=2000 + seed)
        for seed in range(10)
    ]
    print("epsilon=0, initial Q=-20: "
          f"{sum(run.path_length == 8 for run in negative_initial)}/10 optimal, "
          f"mean last 50={mean(mean(run.returns[-50:]) for run in negative_initial):.2f}")

    print("\nExercise 2: independent learners")
    penalty = -50
    print_payoff_matrix(penalty)
    for best_ever in (False, True):
        label = "averaging update" if not best_ever else "best-payoff-ever update"
        print(f"{label}:")
        for current_penalty in (0, -20, -50, -100):
            results = [train_game(current_penalty, seed=3000 + seed,
                                  best_ever=best_ever) for seed in range(50)]
            counts = {outcome: sum(result.outcome == outcome for result in results)
                      for outcome in ("optimal joint action", "safe one", "mismatch")}
            print(f"  k={current_penalty:>4}: optimal={counts['optimal joint action']:>2}, "
                  f"safe={counts['safe one']:>2}, mismatch={counts['mismatch']:>2}")
    example = train_game(seed=0)
    print(f"Example final estimates: A={example.values_a}, B={example.values_b}; "
          f"greedy actions={example.actions} ({example.outcome})")
    print("\nObservations:")
    print("- A greedy policy can become shortest-path optimal while exploratory returns remain lower.")
    print("- At the zero-initialised start, equally good first moves share the highest value.")
    print("- The other learner's exploration makes the reward for a locally good action non-stationary.")
    print("- The final pair is an equilibrium only when it is a matching optimal choice or (1, 1).")
    print("\nValidation: self-test passed")


if __name__ == "__main__":
    main()