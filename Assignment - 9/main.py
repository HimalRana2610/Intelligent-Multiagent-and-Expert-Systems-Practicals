"""Experiment 9: auditing elicited rules and inducing rules with a decision tree.

Run ``python main.py`` for the complete experiment and exercises.
Run ``python main.py --test`` for deterministic validation checks.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


ATTRIBUTES = ("has_hair", "gives_milk", "has_feathers", "flies", "swims")
VALUES = (False, True)
LABELS = ("bird", "carnivore", "fish", "mammal", "penguin", "unknown")
Case = Tuple[bool, ...]


@dataclass(frozen=True)
class Rule:
    identifier: str
    conditions: Tuple[Tuple[str, bool], ...]
    conclusion: str

    def matches(self, case: Mapping[str, bool]) -> bool:
        return all(case[name] == value for name, value in self.conditions)

    def __str__(self) -> str:
        tests = " AND ".join(f"{name}={value}" for name, value in self.conditions)
        return f"{self.identifier}: IF {tests} THEN {self.conclusion}"


@dataclass(frozen=True)
class Tree:
    attribute: str | None = None
    children: Tuple[Tuple[bool, "Tree"], ...] = ()
    label: str | None = None

    def predict(self, case: Mapping[str, bool] | Case) -> str:
        if self.label is not None:
            return self.label
        facts = as_dict(case) if isinstance(case, tuple) else case
        for value, child in self.children:
            if facts[self.attribute] == value:  # type: ignore[index]
                return child.predict(case)
        raise ValueError("tree has no branch for case value")

    def leaves(self) -> int:
        return 1 if self.label is not None else sum(child.leaves() for _, child in self.children)


def all_cases() -> Tuple[Case, ...]:
    return tuple(itertools.product(VALUES, repeat=len(ATTRIBUTES)))


def as_dict(case: Case) -> Dict[str, bool]:
    return dict(zip(ATTRIBUTES, case))


def target_label(case: Case) -> str:
    facts = as_dict(case)
    if facts["has_feathers"] and facts["swims"]:
        return "penguin"
    if facts["has_feathers"] and facts["flies"]:
        return "bird"
    if facts["has_hair"] and facts["gives_milk"] and facts["flies"]:
        return "carnivore"
    if facts["has_hair"] and facts["gives_milk"]:
        return "mammal"
    if facts["swims"]:
        return "fish"
    return "unknown"


def elicited_rules() -> Tuple[Rule, ...]:
    return (
        Rule("R1", (("has_feathers", True), ("flies", True)), "bird"),
        Rule("R2", (("has_feathers", True), ("swims", True)), "penguin"),
        Rule("R3", (("has_feathers", True),), "bird"),
        Rule("R4", (("has_hair", True), ("gives_milk", True)), "mammal"),
        Rule("R5", (("has_hair", True), ("gives_milk", True), ("flies", True)), "carnivore"),
        Rule("R6", (("swims", True),), "fish"),
        Rule("R7", (("has_hair", True), ("flies", True)), "carnivore"),
    )


def fired_rules(rules: Sequence[Rule], case: Case) -> Tuple[Rule, ...]:
    facts = as_dict(case)
    return tuple(rule for rule in rules if rule.matches(facts))


def audit(rules: Sequence[Rule]) -> Tuple[List[Case], List[Tuple[Case, Tuple[Rule, ...]]], float]:
    uncovered: List[Case] = []
    conflicts: List[Tuple[Case, Tuple[Rule, ...]]] = []
    correct = 0
    for case in all_cases():
        fired = fired_rules(rules, case)
        if not fired:
            uncovered.append(case)
        if len({rule.conclusion for rule in fired}) > 1:
            conflicts.append((case, fired))
        if fired and fired[0].conclusion == target_label(case):
            correct += 1
    return uncovered, conflicts, correct / len(all_cases())


def entropy(cases: Sequence[Tuple[Case, str]]) -> float:
    counts: Dict[str, int] = {}
    for _, label in cases:
        counts[label] = counts.get(label, 0) + 1
    total = len(cases)
    return -sum((n / total) * math.log2(n / total) for n in counts.values()) if total else 0.0


def information_gain(cases: Sequence[Tuple[Case, str]], index: int) -> float:
    remainder = 0.0
    for value in VALUES:
        subset = [item for item in cases if item[0][index] == value]
        if subset:
            remainder += len(subset) / len(cases) * entropy(subset)
    return entropy(cases) - remainder


def majority(cases: Sequence[Tuple[Case, str]]) -> str:
    counts: Dict[str, int] = {}
    for _, label in cases:
        counts[label] = counts.get(label, 0) + 1
    return min(counts, key=lambda label: (-counts[label], LABELS.index(label)))


def induce_tree(
    examples: Sequence[Tuple[Case, str]],
    available: Sequence[int] | None = None,
    min_split: int = 1,
) -> Tree:
    if not examples:
        raise ValueError("cannot induce a tree from no examples")
    labels = {label for _, label in examples}
    available = tuple(range(len(ATTRIBUTES))) if available is None else tuple(available)
    if len(labels) == 1 or not available or len(examples) < min_split:
        return Tree(label=majority(examples))
    best = max(available, key=lambda index: (information_gain(examples, index), -index))
    if information_gain(examples, best) <= 1e-12:
        return Tree(label=majority(examples))
    remaining = tuple(index for index in available if index != best)
    children = []
    for value in VALUES:
        subset = [item for item in examples if item[0][best] == value]
        if subset:
            children.append((value, induce_tree(subset, remaining, min_split)))
        else:
            children.append((value, Tree(label=majority(examples))))
    return Tree(ATTRIBUTES[best], tuple(children))


def tree_rules(tree: Tree) -> Tuple[Rule, ...]:
    counter = 0

    def visit(node: Tree, prefix: Tuple[Tuple[str, bool], ...]) -> Tuple[Rule, ...]:
        nonlocal counter
        if node.label is not None:
            counter += 1
            return (Rule(f"T{counter}", prefix, node.label),)
        rules: List[Rule] = []
        for value, child in node.children:
            rules.extend(visit(child, prefix + ((node.attribute, value),)))
        return tuple(rules)

    return visit(tree, ())


def format_case(case: Case) -> str:
    return "(" + ", ".join(f"{name}={value}" for name, value in zip(ATTRIBUTES, case)) + ")"


def accuracy(predict, cases: Iterable[Case]) -> float:
    cases = tuple(cases)
    return sum(predict(case) == target_label(case) for case in cases) / len(cases)


def sample_examples(size: int, seed: int, corruption: float = 0.0) -> Tuple[Tuple[Case, str], ...]:
    rng = random.Random(seed)
    cases = all_cases()
    examples = []
    for _ in range(size):
        case = rng.choice(cases)
        label = target_label(case)
        if rng.random() < corruption:
            label = rng.choice(tuple(other for other in LABELS if other != label))
        examples.append((case, label))
    return tuple(examples)


def print_tree(tree: Tree, indent: str = "") -> None:
    if tree.label is not None:
        print(f"{indent}=> {tree.label}")
        return
    print(f"{indent}{tree.attribute}?")
    for value, child in tree.children:
        print(f"{indent}  [{value}]")
        print_tree(child, indent + "    ")


def main() -> None:
    rules = elicited_rules()
    cases = all_cases()
    uncovered, conflicts, elicited_accuracy = audit(rules)
    examples = tuple((case, target_label(case)) for case in cases)
    tree = induce_tree(examples)
    induced = tree_rules(tree)

    print("=" * 78)
    print("Experiment 9: Knowledge acquisition")
    print("=" * 78)
    print("\nElicited rule base")
    for rule in rules:
        print(" ", rule)
    print(f"\nComplete case space: {len(cases)} cases")
    print(f"Uncovered cases ({len(uncovered)}):")
    for case in uncovered:
        print(" ", format_case(case))
    print(f"\nConflicting cases ({len(conflicts)}):")
    for case, fired in conflicts:
        print(" ", format_case(case), "->", ", ".join(f"{r.identifier}:{r.conclusion}" for r in fired))
    print(f"\nElicited first-match accuracy: {elicited_accuracy:.1%}")

    print("\nInduced decision tree")
    print_tree(tree)
    print("\nRules read from induced tree")
    for rule in induced:
        print(" ", rule)
    agrees = sum(
        any(rule.conditions == elicited.conditions and rule.conclusion == elicited.conclusion
            for rule in induced)
        for elicited in rules
    )
    print(f"\nInduced leaves/rules: {tree.leaves()}/{len(induced)}")
    print(f"Induced accuracy over complete space: {accuracy(tree.predict, cases):.1%}")
    print(f"Elicited rules exactly confirmed by induced rules: {agrees}/{len(rules)}")

    print("\nExercise 1: label corruption (mean over 20 case bases)")
    print("  corruption   leaves (plain/min-4)   accuracy (plain/min-4)")
    for level in (0.0, 0.05, 0.10, 0.20):
        plain, pruned, plain_acc, pruned_acc = [], [], [], []
        for seed in range(20):
            sample = sample_examples(40, 1000 + seed, level)
            a, b = induce_tree(sample), induce_tree(sample, min_split=4)
            plain.append(a.leaves()); pruned.append(b.leaves())
            plain_acc.append(accuracy(a.predict, cases)); pruned_acc.append(accuracy(b.predict, cases))
        print(f"  {level:>9.0%}   {sum(plain)/20:>6.1f}/{sum(pruned)/20:<6.1f}"
              f"       {sum(plain_acc)/20:>6.1%}/{sum(pruned_acc)/20:<6.1%}")

    print("\nExercise 2: case-base size (mean over 20 draws)")
    print("  cases   leaves   accuracy   elicited rules confirmed")
    overtakes = None
    for size in (8, 16, 24, 40, 80):
        leaves, accuracies, confirmations = [], [], []
        for seed in range(20):
            sample = sample_examples(size, 2000 + seed)
            induced_sample = induce_tree(sample)
            leaves.append(induced_sample.leaves())
            accuracies.append(accuracy(induced_sample.predict, cases))
            learned = tree_rules(induced_sample)
            confirmations.append(sum(any(r.conditions == e.conditions and r.conclusion == e.conclusion
                                         for r in learned) for e in rules))
        mean_accuracy = sum(accuracies) / len(accuracies)
        print(f"  {size:>5}   {sum(leaves)/20:>6.1f}   {mean_accuracy:>8.1%}   {sum(confirmations)/20:>8.1f}/{len(rules)}")
        if overtakes is None and mean_accuracy > elicited_accuracy:
            overtakes = size
    print(f"Induction first overtakes the elicited base at: {overtakes or 'no tested size'} cases")
    print("\nObservations")
    print("- An uncovered case silently produces no answer; a conflicting case can produce a wrong answer.")
    print("- First-match ordering hides conflicts operationally but does not remove contradictory knowledge.")
    print("- The induced tree is complete for the examples and its agreement with elicited rules is a separate measure.")
    print("- The expert still supplies the vocabulary, solved cases, labels, and judgement about noisy data.")


def run_tests() -> None:
    cases = all_cases()
    rules = elicited_rules()
    uncovered, conflicts, first_accuracy = audit(rules)
    assert len(cases) == 32
    assert uncovered and conflicts
    assert 0.0 <= first_accuracy <= 1.0
    complete = tuple((case, target_label(case)) for case in cases)
    tree = induce_tree(complete)
    assert tree.leaves() > 1
    assert accuracy(tree.predict, cases) == 1.0
    assert len(tree_rules(tree)) == tree.leaves()
    assert entropy(complete) > 0
    assert information_gain(complete, 0) >= 0
    small = sample_examples(8, 99)
    assert induce_tree(small, min_split=4).leaves() <= induce_tree(small).leaves()
    print("All Assignment 9 tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="run deterministic validation checks")
    args = parser.parse_args()
    run_tests() if args.test else main()