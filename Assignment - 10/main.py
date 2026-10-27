"""Experiment 10: ethical auditing of a decision system.

Run ``python main.py`` for the complete audit.
Run ``python main.py --test`` for deterministic validation checks.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


FEATURES = ("group", "income", "experience", "degree", "region")
QIS = ("income", "experience", "degree", "region")
GROUPS = ("A", "B")


@dataclass(frozen=True)
class Applicant:
    identifier: str
    group: str
    income: int
    experience: int
    degree: str
    region: str
    deserved: bool
    historical: bool

    def values(self, features: Sequence[str]) -> Tuple[object, ...]:
        return tuple(getattr(self, name) for name in features)


@dataclass(frozen=True)
class Tree:
    feature: str | None = None
    branches: Tuple[Tuple[object, "Tree"], ...] = ()
    decision: bool | None = None
    fallback: bool | None = None

    def predict(self, applicant: Applicant) -> bool:
        if self.decision is not None:
            return self.decision
        value = getattr(applicant, self.feature)  # type: ignore[arg-type]
        for branch_value, child in self.branches:
            if value == branch_value:
                return child.predict(applicant)
        if self.fallback is not None:
            return self.fallback
        raise ValueError(f"tree has no branch for {self.feature}={value!r}")

    def path(self, applicant: Applicant) -> Tuple[Tuple[str, object], ...]:
        if self.decision is not None:
            return ()
        value = getattr(applicant, self.feature)  # type: ignore[arg-type]
        for branch_value, child in self.branches:
            if value == branch_value:
                return ((self.feature, value),) + child.path(applicant)  # type: ignore[return-value]
        if self.fallback is not None:
            return ((self.feature, value), ("fallback", self.fallback))  # type: ignore[return-value]
        raise ValueError("tree has no matching branch")

    def leaves(self) -> int:
        return 1 if self.decision is not None else sum(child.leaves() for _, child in self.branches)


def make_population(size: int, seed: int, biased: bool = True) -> Tuple[Applicant, ...]:
    rng = random.Random(seed)
    applicants: List[Applicant] = []
    for index in range(size):
        group = "B" if rng.random() < 0.45 else "A"
        income = rng.choice((30, 40, 50, 60, 70, 80))
        experience = rng.choice((1, 3, 5, 7, 9))
        degree = rng.choice(("none", "bachelor", "master"))
        region = rng.choice(("north", "south", "east", "west"))
        score = (income >= 60) + (experience >= 5) + (degree != "none")
        deserved = score >= 2
        historical = deserved
        if biased and group == "B" and deserved and rng.random() < 0.28:
            historical = False
        if biased and group == "B" and not deserved and rng.random() < 0.05:
            historical = True
        applicants.append(Applicant(f"P{index + 1:03d}", group, income, experience,
                                    degree, region, deserved, historical))
    return tuple(applicants)


def entropy(rows: Sequence[Tuple[Tuple[object, ...], bool]]) -> float:
    counts = Counter(label for _, label in rows)
    return -sum((n / len(rows)) * math.log2(n / len(rows)) for n in counts.values()) if rows else 0.0


def gain(rows: Sequence[Tuple[Tuple[object, ...], bool]], index: int) -> float:
    remainder = 0.0
    for value in {values[index] for values, _ in rows}:
        subset = [row for row in rows if row[0][index] == value]
        remainder += len(subset) / len(rows) * entropy(subset)
    return entropy(rows) - remainder


def majority(rows: Sequence[Tuple[Tuple[object, ...], bool]]) -> bool:
    return sum(label for _, label in rows) * 2 >= len(rows)


def induce(rows: Sequence[Tuple[Tuple[object, ...], bool]], features: Sequence[str],
           minimum: int = 1) -> Tree:
    labels = {label for _, label in rows}
    if len(labels) == 1 or len(rows) < minimum or not features:
        return Tree(decision=majority(rows))
    best = max(range(len(features)), key=lambda i: (gain(rows, i), -i))
    if gain(rows, best) <= 1e-12:
        return Tree(decision=majority(rows))
    feature = features[best]
    remaining = features[:best] + features[best + 1:]
    values = sorted({values[best] for values, _ in rows}, key=str)
    branches = tuple(
        (value, induce([(values_row, label) for values_row, label in rows if values_row[best] == value],
                       remaining, minimum))
        for value in values
    )
    return Tree(feature=feature, branches=branches, fallback=majority(rows))


def train(applicants: Sequence[Applicant], features: Sequence[str],
          label: str = "historical") -> Tree:
    return induce([(applicant.values(features), getattr(applicant, label)) for applicant in applicants], features)


def rates(tree: Tree, applicants: Sequence[Applicant]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for group in GROUPS:
        rows = [a for a in applicants if a.group == group]
        predicted = [tree.predict(a) for a in rows]
        tp = sum(p and a.deserved for p, a in zip(predicted, rows))
        fp = sum(p and not a.deserved for p, a in zip(predicted, rows))
        positives = sum(a.deserved for a in rows)
        negatives = len(rows) - positives
        selected = sum(predicted)
        result[group] = {
            "base": positives / len(rows),
            "selection": selected / len(rows),
            "tpr": tp / positives if positives else 0.0,
            "fpr": fp / negatives if negatives else 0.0,
            "precision": tp / selected if selected else 0.0,
        }
    return result


def gaps(metric: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    return {name: abs(metric["A"][name] - metric["B"][name]) for name in metric["A"]}


def accuracy(tree: Tree, applicants: Iterable[Applicant], label: str) -> float:
    rows = tuple(applicants)
    return sum(tree.predict(a) == getattr(a, label) for a in rows) / len(rows)


def equivalence(applicants: Sequence[Applicant], attributes: Sequence[str]) -> Dict[Tuple[object, ...], int]:
    return dict(Counter(a.values(attributes) for a in applicants))


def privacy_stats(applicants: Sequence[Applicant], attributes: Sequence[str]) -> Dict[str, float]:
    classes = equivalence(applicants, attributes)
    counts = tuple(classes.values())
    return {
        "k": min(counts),
        "unique": sum(n == 1 for n in counts),
        "under5": sum(n for n in counts if n < 5) / len(applicants),
        "classes": len(classes),
    }


def generalised(a: Applicant, level: int) -> Tuple[object, ...]:
    if level == 0:
        return a.values(QIS)
    if level == 1:
        return (a.income // 20 * 20, a.experience // 4 * 4, a.degree, a.region)
    if level == 2:
        return (a.income // 30 * 30, a.experience // 5 * 5, a.degree, "*")
    return (a.income // 40 * 40, a.experience // 10 * 10, "*", "*")


def print_tree(tree: Tree, indent: str = "") -> None:
    if tree.decision is not None:
        print(f"{indent}=> {'accept' if tree.decision else 'reject'}")
        return
    print(f"{indent}{tree.feature}?")
    for value, child in tree.branches:
        print(f"{indent}  [{value}]")
        print_tree(child, indent + "    ")


def main() -> None:
    train_set = make_population(240, 10)
    test_set = make_population(160, 20)
    features = FEATURES
    tree = train(train_set, features)
    metric = rates(tree, test_set)
    print("=" * 78)
    print("Experiment 10: Ethical audit of a decision system")
    print("=" * 78)
    print("\nHistorical-decision tree")
    print_tree(tree)
    print(f"Leaves: {tree.leaves()}")
    print("\nFairness audit against deserved outcomes")
    print("group  base rate  selection  true-positive  false-positive  precision")
    for group in GROUPS:
        row = metric[group]
        print(f"  {group}    {row['base']:>7.1%}    {row['selection']:>7.1%}"
              f"      {row['tpr']:>7.1%}       {row['fpr']:>7.1%}    {row['precision']:>7.1%}")
    print("gap   " + "  ".join(f"{name}={value:.1%}" for name, value in gaps(metric).items()))
    print(f"Agreement with historical decisions: {accuracy(tree, test_set, 'historical'):.1%}")
    print(f"Accuracy against deserved outcomes: {accuracy(tree, test_set, 'deserved'):.1%}")

    print("\nPrivacy audit")
    print("level  quasi-identifiers                 k   unique  records in classes <5")
    for level in range(4):
        counts = dict(Counter(generalised(a, level) for a in train_set))
        values = tuple(counts.values())
        print(f"  {level}    {('raw' if level == 0 else 'generalised ' + str(level)): <30}"
              f" {min(values):>3}   {sum(v == 1 for v in values):>6}        "
              f"{sum(v for v in values if v < 5) / len(train_set):>7.1%}")

    lengths = [len(tree.path(a)) for a in test_set]
    named = test_set[0]
    print("\nTransparency audit")
    print(f"Mean explanation conditions: {sum(lengths) / len(lengths):.2f}")
    print(f"Maximum explanation conditions: {max(lengths)}")
    print(f"Explanation for {named.identifier}:")
    for feature, value in tree.path(named):
        print(f"  {feature} = {value}")
    print(f"  decision = {'accept' if tree.predict(named) else 'reject'}")

    print("\nExercise 1: mitigation comparison")
    print("configuration                         accuracy  selection gap  TPR gap  precision gap")
    configurations = (
        ("unmodified", FEATURES, "historical"),
        ("remove group", tuple(f for f in FEATURES if f != "group"), "historical"),
        ("remove group+region", tuple(f for f in FEATURES if f not in ("group", "region")), "historical"),
        ("train on deserved (oracle)", FEATURES, "deserved"),
    )
    for name, selected, label in configurations:
        model = train(train_set, selected, label)
        values = rates(model, test_set)
        print(f"{name:<35} {accuracy(model, test_set, 'deserved'):>7.1%}"
              f"     {gaps(values)['selection']:>7.1%}  {gaps(values)['tpr']:>7.1%}"
              f"       {gaps(values)['precision']:>7.1%}")

    print("\nExercise 2: privacy generalisation and model effects")
    print("level  smallest class  unique records  historical agreement  deserved accuracy  mean path")
    for level in range(4):
        key = lambda a, level=level: generalised(a, level)
        # A tree trained on generalised quasi-identifiers demonstrates the
        # accuracy/explanation cost of publishing less precise data.
        generalized_train = tuple(
            Applicant(a.identifier, a.group, *key(a)[:2],
                      str(key(a)[2]), str(key(a)[3]), a.deserved, a.historical)
            for a in train_set
        )
        model = train(generalized_train, ("group", "income", "experience", "degree", "region"))
        generalized_test = tuple(
            Applicant(a.identifier, a.group, *key(a)[:2],
                      str(key(a)[2]), str(key(a)[3]), a.deserved, a.historical)
            for a in test_set
        )
        stats = privacy_stats(train_set, QIS) if level == 0 else privacy_stats(
            tuple(Applicant(a.identifier, a.group, *key(a)[:2], str(key(a)[2]), str(key(a)[3]),
                            a.deserved, a.historical) for a in train_set), QIS)
        lengths2 = [len(model.path(a)) for a in generalized_test]
        print(f"  {level}          {stats['k']:>3}             {stats['unique']:>4}"
              f"             {accuracy(model, generalized_test, 'historical'):>7.1%}"
              f"             {accuracy(model, generalized_test, 'deserved'):>7.1%}"
              f"        {sum(lengths2) / len(lengths2):>5.2f}")

    print("\nObservations")
    print("- Selection, true-positive, and precision gaps answer different fairness questions; equalising one does not guarantee the others.")
    print("- Removing a protected field may not remove proxy effects from correlated attributes such as region or income.")
    print("- Singleton equivalence classes expose records to linkage and re-identification from another dataset.")
    print("- Accuracy alone omits disparate outcomes, privacy risk, and the trace needed for accountability.")


def run_tests() -> None:
    train_set, test_set = make_population(240, 10), make_population(160, 20)
    model = train(train_set, FEATURES)
    metric = rates(model, test_set)
    assert len(train_set) == 240 and len(test_set) == 160
    assert set(metric) == set(GROUPS)
    assert all(0.0 <= value <= 1.0 for row in metric.values() for value in row.values())
    assert len(model.path(test_set[0])) >= 1
    assert model.leaves() >= 2
    assert sum(equivalence(train_set, QIS).values()) == len(train_set)
    assert privacy_stats(train_set, QIS)["k"] >= 1
    assert 0.0 <= accuracy(model, test_set, "deserved") <= 1.0
    print("All Assignment 10 tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else main()