"""Experiment 12: a financial advisor with scoring rules and hard constraints.

Run ``python main.py`` for the complete experiment.
Run ``python main.py --test`` for deterministic validation checks.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple


ASSETS = ("equity", "bonds", "cash")
Allocation = Tuple[float, float, float]
BANDS = (
    (0, 3, "conservative", (0.20, 0.55, 0.25)),
    (4, 6, "balanced", (0.50, 0.40, 0.10)),
    (7, 9, "growth", (0.70, 0.25, 0.05)),
    (10, 12, "aggressive", (0.85, 0.10, 0.05)),
)
RETURNS = {
    "equity": (0.075, 0.18),
    "bonds": (0.035, 0.06),
    "cash": (0.020, 0.01),
}


@dataclass(frozen=True)
class Client:
    identifier: str
    age: int
    horizon: int
    income_stability: str
    dependants: int
    loss_tolerance: str
    high_interest_debt: bool
    emergency_months: int
    investment: float = 100000.0


@dataclass(frozen=True)
class Advice:
    score: int
    band: str
    unconstrained: Allocation
    allocation: Allocation
    trace: Tuple[str, ...]


@dataclass(frozen=True)
class Outcome:
    median_value: float
    fifth_percentile: float
    probability_of_loss: float


def score(client: Client) -> int:
    points = 0
    points += 3 if client.age < 35 else 2 if client.age < 55 else 0
    points += 3 if client.horizon >= 15 else 2 if client.horizon >= 7 else 0
    points += 2 if client.income_stability == "high" else 1 if client.income_stability == "medium" else 0
    points += 1 if client.dependants == 0 else 0
    points += 3 if client.loss_tolerance == "high" else 1 if client.loss_tolerance == "medium" else 0
    return points


def band_for(value: int) -> Tuple[str, Allocation]:
    for lower, upper, name, allocation in BANDS:
        if lower <= value <= upper:
            return name, allocation
    raise ValueError(f"score outside supported range: {value}")


def apply_constraints(client: Client, allocation: Allocation) -> Tuple[Allocation, Tuple[str, ...]]:
    equity, bonds, cash = allocation
    trace: List[str] = []
    if client.high_interest_debt:
        trace.append("C1: high-interest debt -> hold 100% cash until repaid")
        return (0.0, 0.0, 1.0), tuple(trace)
    if client.emergency_months < 6:
        target_cash = 0.20
        if cash < target_cash:
            shift = target_cash - cash
            equity = max(0.0, equity - shift)
            cash = target_cash
            trace.append("C2: emergency reserve below 6 months -> raise cash to 20%")
    if client.horizon <= 3 and equity > 0.0:
        trace.append("C3: money needed within 3 years -> cap equity at 0%")
        equity = 0.0
    total = equity + bonds + cash
    return (equity / total, bonds / total, cash / total), tuple(trace)


def advise(client: Client) -> Advice:
    value = score(client)
    band, unconstrained = band_for(value)
    allocation, constraints = apply_constraints(client, unconstrained)
    trace = (f"S1: score {value} -> {band} band ({format_allocation(unconstrained)})",) + constraints
    if not constraints:
        trace += ("No hard constraint fired.",)
    return Advice(value, band, unconstrained, allocation, trace)


def format_allocation(allocation: Allocation) -> str:
    return " / ".join(f"{asset} {value:.0%}" for asset, value in zip(ASSETS, allocation))


def simulate(allocation: Allocation, horizon: int, paths: int = 2000, seed: int = 1) -> Outcome:
    if paths < 1000:
        raise ValueError("the simulation requires at least 1,000 paths")
    rng = random.Random(seed)
    values = []
    for _ in range(paths):
        value = 100000.0
        for _ in range(horizon):
            annual = sum(
                weight * rng.gauss(*RETURNS[asset])
                for asset, weight in zip(ASSETS, allocation)
            )
            value *= 1.0 + annual
        values.append(value)
    values.sort()
    fifth = values[max(0, int(0.05 * len(values)) - 1)]
    return Outcome(median(values), fifth, sum(value < 100000.0 for value in values) / len(values))


def clients() -> Tuple[Client, ...]:
    return (
        Client("C1 young stable investor", 29, 25, "high", 0, "high", False, 9),
        Client("C2 family near-term need", 48, 3, "medium", 2, "medium", False, 8),
        Client("C3 debt emergency", 41, 12, "high", 1, "high", True, 2),
    )


def population(size: int, seed: int) -> Tuple[Client, ...]:
    rng = random.Random(seed)
    return tuple(
        Client(
            f"P{index + 1:03d}",
            rng.randint(25, 65),
            rng.randint(1, 25),
            rng.choice(("low", "medium", "high")),
            rng.randint(0, 3),
            rng.choice(("low", "medium", "high")),
            rng.random() < 0.18,
            rng.randint(0, 12),
        )
        for index in range(size)
    )


def percentile_change(advice: Advice, client: Client, seed: int) -> Tuple[Outcome, Outcome]:
    return (
        simulate(advice.allocation, client.horizon, seed=seed),
        simulate(advice.unconstrained, client.horizon, seed=seed),
    )


def smooth_allocation(value: int) -> Allocation:
    centres = ((1.5, BANDS[0][3]), (5.0, BANDS[1][3]), (8.0, BANDS[2][3]), (11.0, BANDS[3][3]))
    if value <= centres[0][0]:
        return centres[0][1]
    if value >= centres[-1][0]:
        return centres[-1][1]
    for (left_score, left), (right_score, right) in zip(centres, centres[1:]):
        if left_score <= value <= right_score:
            fraction = (value - left_score) / (right_score - left_score)
            return tuple(a + fraction * (b - a) for a, b in zip(left, right))  # type: ignore[return-value]
    raise AssertionError("score interpolation failed")


def main() -> None:
    selected = clients()
    print("=" * 78)
    print("Experiment 12: Financial advisor with scoring rules and constraints")
    print("=" * 78)
    print("\nClient recommendations and explanation traces")
    for client in selected:
        advice = advise(client)
        print(f"\n{client.identifier}: score={advice.score}, band={advice.band}")
        print(f"  unconstrained: {format_allocation(advice.unconstrained)}")
        print(f"  recommended:   {format_allocation(advice.allocation)}")
        for item in advice.trace:
            print(f"  {item}")

    print("\nScenario evaluation (2,000 paths)")
    print("client                         recommendation                 median       p5   loss    unconstrained p5")
    for index, client in enumerate(selected):
        advice = advise(client)
        recommended, unconstrained = percentile_change(advice, client, 900 + index)
        print(f"{client.identifier:<30} {format_allocation(advice.allocation):<34}"
              f" ${recommended.median_value:>9,.0f} ${recommended.fifth_percentile:>9,.0f}"
              f" {recommended.probability_of_loss:>5.1%} ${unconstrained.fifth_percentile:>9,.0f}")

    people = population(120, 120)
    print("\nExercise 1: population and constraint effects")
    changes = sum(advise(person).allocation != advise(person).unconstrained for person in people)
    print(f"Recommendations changed by constraints: {changes}/{len(people)} ({changes / len(people):.1%})")
    for label, predicate in (
        ("C1 high-interest debt", lambda person: person.high_interest_debt),
        ("C2 emergency reserve", lambda person: not person.high_interest_debt and person.emergency_months < 6),
        ("C3 short horizon", lambda person: not person.high_interest_debt and person.horizon <= 3),
    ):
        affected = [(person, advise(person)) for person in people if predicate(person)]
        if affected:
            constrained = mean(
                percentile_change(advice, person, 2000 + index)[0].fifth_percentile
                for index, (person, advice) in enumerate(affected)
            )
            unconstrained = mean(
                percentile_change(advice, person, 2000 + index)[1].fifth_percentile
                for index, (person, advice) in enumerate(affected)
            )
            print(f"  {label:<26} affected={len(affected):>3} mean p5: "
                  f"constrained ${constrained:,.0f}, unconstrained ${unconstrained:,.0f}")

    print("\nExercise 2: band boundaries and smooth interpolation")
    boundary = 6
    print(f"At scores {boundary} and {boundary + 1}: band equity "
          f"{band_for(boundary)[1][0]:.0%} -> {band_for(boundary + 1)[1][0]:.0%}; "
          f"single-point jump={band_for(boundary + 1)[1][0] - band_for(boundary)[1][0]:.0%}")
    print(f"Smooth version at scores {boundary} and {boundary + 1}: "
          f"{smooth_allocation(boundary)[0]:.1%} -> {smooth_allocation(boundary + 1)[0]:.1%}")
    boundary_client = Client("boundary client", 40, 10, "medium", 1, "medium", False, 8)
    stepped = simulate(band_for(boundary)[1], 10, seed=77)
    smooth = simulate(smooth_allocation(boundary), 10, seed=77)
    print(f"Ten-year score-{boundary} comparison: stepped p5 ${stepped.fifth_percentile:,.0f}, "
          f"smooth p5 ${smooth.fifth_percentile:,.0f}")
    print("\nObservations")
    print("- Constraints act as safety gates; a high risk score cannot buy away debt or liquidity needs.")
    print("- The fifth percentile exposes a bad outcome that the median hides.")
    print("- The trace explains both the score-based recommendation and each override, including cash holdings.")
    print("- Return simulation does not model debt interest, taxes, fees, inflation, or the real value of repaying debt.")


def run_tests() -> None:
    client = clients()[1]
    advice = advise(client)
    assert sum(advice.allocation) == 1.0
    assert advice.allocation[0] == 0.0
    assert any("C3" in item for item in advice.trace)
    debt_advice = advise(clients()[2])
    assert debt_advice.allocation == (0.0, 0.0, 1.0)
    assert len(simulate(advice.allocation, 3, paths=1000, seed=1).__dict__) == 3
    assert smooth_allocation(6)[0] > band_for(3)[1][0]
    assert smooth_allocation(6)[0] < band_for(7)[1][0]
    assert len(population(20, 1)) == 20
    print("All Assignment 12 tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else main()