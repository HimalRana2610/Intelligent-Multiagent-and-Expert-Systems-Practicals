"""Experiment 11: a medical diagnosis expert system with certainty factors.

Run ``python main.py`` for the complete experiment.
Run ``python main.py --test`` for deterministic validation checks.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, Mapping, Sequence, Tuple


DIAGNOSES = ("influenza", "migraine", "pneumonia", "gastroenteritis")
FINDINGS = (
    "fever",
    "headache",
    "cough",
    "shortness_of_breath",
    "nausea",
    "vomiting",
    "diarrhoea",
    "light_sensitivity",
    "body_aches",
    "chest_pain",
)
THRESHOLD = 0.65
Evidence = Mapping[str, float]


@dataclass(frozen=True)
class Rule:
    identifier: str
    diagnosis: str
    conditions: Tuple[Tuple[str, bool], ...]
    certainty: float


@dataclass(frozen=True)
class DiagnosisResult:
    diagnosis: str
    certainty: float
    contributors: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class Consultation:
    strategy: str
    asked: Tuple[str, ...]
    result: Tuple[DiagnosisResult, ...]
    conclusion: str | None


def combine_cf(first: float, second: float) -> float:
    """Combine two certainty factors using the standard signed CF rule."""
    if not -1 <= first <= 1 or not -1 <= second <= 1:
        raise ValueError("certainty factors must be between -1 and 1")
    if first >= 0 and second >= 0:
        value = first + second - first * second
    elif first <= 0 and second <= 0:
        value = first + second + first * second
    else:
        denominator = 1 - min(abs(first), abs(second))
        value = (first + second) / denominator if denominator else 0.0
    return max(-1.0, min(1.0, value))


def conjunction_strength(rule: Rule, evidence: Evidence) -> float:
    """Return the weakest reported certainty for a rule's required findings."""
    strengths = []
    for finding, required in rule.conditions:
        reported = evidence.get(finding, 0.0)
        strengths.append(reported if required else -reported)
    return min(strengths)


def rule_contribution(rule: Rule, evidence: Evidence) -> float:
    return rule.certainty * conjunction_strength(rule, evidence)


def evaluate(rules: Sequence[Rule], evidence: Evidence) -> Tuple[DiagnosisResult, ...]:
    grouped: Dict[str, Tuple[float, Tuple[Tuple[str, float], ...]]] = {}
    for rule in rules:
        contribution = rule_contribution(rule, evidence)
        if contribution == 0:
            continue
        if rule.diagnosis not in grouped:
            grouped[rule.diagnosis] = (0.0, ())
        current, contributors = grouped[rule.diagnosis]
        grouped[rule.diagnosis] = (
            combine_cf(current, contribution),
            contributors + ((rule.identifier, contribution),),
        )
    results = [
        DiagnosisResult(diagnosis, certainty, contributors)
        for diagnosis, (certainty, contributors) in grouped.items()
    ]
    results.extend(
        DiagnosisResult(diagnosis, 0.0, ())
        for diagnosis in DIAGNOSES
        if diagnosis not in grouped
    )
    return tuple(sorted(results, key=lambda result: (-result.certainty, DIAGNOSES.index(result.diagnosis))))


def rule_base() -> Tuple[Rule, ...]:
    return (
        Rule("R1", "influenza", (("fever", True), ("body_aches", True)), 0.80),
        Rule("R2", "influenza", (("cough", True), ("fever", True), ("headache", True)), 0.72),
        Rule("R3", "influenza", (("nausea", False), ("body_aches", True)), 0.45),
        Rule("R4", "migraine", (("headache", True), ("light_sensitivity", True)), 0.88),
        Rule("R5", "migraine", (("nausea", True), ("fever", False)), 0.62),
        Rule("R6", "pneumonia", (("cough", True), ("shortness_of_breath", True)), 0.90),
        Rule("R7", "pneumonia", (("fever", True), ("chest_pain", True)), 0.70),
        Rule("R8", "gastroenteritis", (("vomiting", True), ("diarrhoea", True)), 0.86),
        Rule("R9", "gastroenteritis", (("nausea", True), ("diarrhoea", True), ("fever", False)), 0.60),
    )


def patient() -> Dict[str, float]:
    return {
        "fever": 0.90,
        "headache": 0.75,
        "cough": 0.80,
        "shortness_of_breath": -0.65,
        "nausea": 0.35,
        "vomiting": 0.20,
        "diarrhoea": -0.70,
        "light_sensitivity": 0.80,
        "body_aches": 0.85,
        "chest_pain": -0.60,
    }


def frequency_order(rules: Sequence[Rule]) -> Tuple[str, ...]:
    counts = Counter(finding for rule in rules for finding, _ in rule.conditions)
    return tuple(sorted(FINDINGS, key=lambda finding: (-counts[finding], FINDINGS.index(finding))))


def goal_order(rules: Sequence[Rule], leading: str | None, remaining: Iterable[str]) -> Tuple[str, ...]:
    remaining_set = set(remaining)
    if leading is None:
        return frequency_order(rules)
    useful = {
        finding
        for rule in rules
        if rule.diagnosis == leading
        for finding, _ in rule.conditions
    }
    return tuple(finding for finding in frequency_order(rules) if finding in useful and finding in remaining_set)


def consult(
    rules: Sequence[Rule],
    complete_evidence: Evidence,
    strategy: str,
    threshold: float = THRESHOLD,
) -> Consultation:
    if strategy == "fixed":
        order = FINDINGS
    elif strategy == "frequency":
        order = frequency_order(rules)
    elif strategy == "goal-directed":
        order = ()
    else:
        raise ValueError(f"unknown consultation strategy: {strategy}")

    known: Dict[str, float] = {}
    asked = []
    remaining = set(FINDINGS)
    while remaining:
        current = evaluate(rules, known)
        leading = current[0].diagnosis if current[0].certainty > 0 else None
        if strategy == "goal-directed":
            choices = goal_order(rules, leading, remaining) or frequency_order(rules)
        else:
            choices = order
        finding = next(candidate for candidate in choices if candidate in remaining)
        remaining.remove(finding)
        asked.append(finding)
        known[finding] = complete_evidence[finding]
        current = evaluate(rules, known)
        if current[0].certainty >= threshold:
            break
    result = evaluate(rules, known)
    conclusion = result[0].diagnosis if result[0].certainty >= threshold else None
    return Consultation(strategy, tuple(asked), result, conclusion)


def perturb(rules: Sequence[Rule], amount: float, seed: int) -> Tuple[Rule, ...]:
    rng = random.Random(seed)
    return tuple(
        Rule(rule.identifier, rule.diagnosis, rule.conditions,
             max(-1.0, min(1.0, rule.certainty + rng.uniform(-amount, amount))))
        for rule in rules
    )


def population(size: int, seed: int) -> Tuple[Dict[str, float], ...]:
    rng = random.Random(seed)
    patients = []
    for _ in range(size):
        patients.append({finding: round(rng.uniform(-1.0, 1.0), 3) for finding in FINDINGS})
    return tuple(patients)


def print_result(result: DiagnosisResult) -> None:
    contributors = ", ".join(f"{identifier}={value:+.3f}" for identifier, value in result.contributors)
    print(f"  {result.diagnosis:<16} CF={result.certainty:+.3f}  [{contributors or 'none'}]")


def main() -> None:
    rules = rule_base()
    findings = patient()
    ranked = evaluate(rules, findings)
    print("=" * 78)
    print("Experiment 11: Medical diagnosis with certainty factors")
    print("=" * 78)
    print("\nRule base")
    for rule in rules:
        conditions = " AND ".join(f"{finding}={'present' if required else 'absent'}"
                                  for finding, required in rule.conditions)
        print(f"  {rule.identifier}: IF {conditions} THEN {rule.diagnosis} (CF={rule.certainty:+.2f})")
    print("\nPatient findings")
    for finding in FINDINGS:
        print(f"  {finding:<22} {findings[finding]:+.2f}")
    print("\nRanked differential")
    for result in ranked:
        print_result(result)
    print(f"  threshold = {THRESHOLD:.2f}")

    print("\nConsultation strategies")
    for strategy in ("fixed", "frequency", "goal-directed"):
        consultation = consult(rules, findings, strategy)
        conclusion = consultation.conclusion or "no diagnosis passed threshold"
        print(f"  {strategy:<14} asked {len(consultation.asked)}: {', '.join(consultation.asked)}")
        print(f"  {'':14} conclusion: {conclusion}")

    print("\nExercise 1: rule-certainty perturbation")
    patients = population(120, 110)
    baseline_rows = [evaluate(rules, item)[0] for item in patients]
    baseline_consultations = [consult(rules, item, "goal-directed") for item in patients]
    for amount in (0.05, 0.10, 0.20):
        changed = no_longer_passed = 0
        changes = []
        altered = perturb(rules, amount, int(amount * 1000))
        for before, original_consultation, item in zip(
            baseline_rows, baseline_consultations, patients
        ):
            after = evaluate(altered, item)[0]
            changed += after.diagnosis != before.diagnosis
            after_consultation = consult(altered, item, "goal-directed")
            no_longer_passed += (
                original_consultation.conclusion is not None
                and after_consultation.conclusion is None
            )
            original_diagnosis_cf = next(
                result.certainty
                for result in evaluate(altered, item)
                if result.diagnosis == before.diagnosis
            )
            changes.append(original_diagnosis_cf - before.certainty)
        print(f"  +/-{amount:.2f}: leading changed {changed / len(patients):.1%}, "
              f"mean leading-CF change {mean(changes):+.3f}, "
              f"threshold lost {no_longer_passed / len(patients):.1%}")

    print("\nExercise 2: consultation cost over population")
    for strategy in ("fixed", "frequency", "goal-directed"):
        consultations = [consult(rules, item, strategy) for item in patients]
        conclusions = {item.conclusion for item in consultations}
        print(f"  {strategy:<14} mean questions={mean(len(item.asked) for item in consultations):.2f}, "
              f"maximum={max(len(item.asked) for item in consultations)}, "
              f"conclusions={len(conclusions)}")
    print("\nObservations")
    print("- Combining independent positive rules increases confidence beyond either rule alone.")
    print("- A conjunction is limited by its weakest reported finding, so uncertain evidence remains weak.")
    print("- A denied finding (negative CF) is evidence against presence; zero means unknown.")
    print("- If no diagnosis reaches the threshold, the system should report uncertainty and seek clinical review.")
    print(f"- For the demonstrated patient, goal-directed asks fewer questions while preserving: "
          f"{ranked[0].diagnosis}.")


def run_tests() -> None:
    rules = rule_base()
    findings = patient()
    assert abs(combine_cf(0.6, 0.5) - 0.8) < 1e-12
    assert abs(combine_cf(0.5, 0.6) - combine_cf(0.6, 0.5)) < 1e-12
    assert conjunction_strength(rules[0], {"fever": 0.9, "body_aches": 0.2}) == 0.2
    assert conjunction_strength(rules[2], {"nausea": 0.8, "body_aches": 0.7}) == -0.8
    ranked = evaluate(rules, findings)
    assert len(ranked) == len(DIAGNOSES)
    assert ranked[0].certainty >= ranked[-1].certainty
    consultations = [consult(rules, findings, strategy) for strategy in ("fixed", "frequency", "goal-directed")]
    assert all(consultation.asked for consultation in consultations)
    assert len(set(consultation.conclusion for consultation in consultations)) == 1
    assert len(perturb(rules, 0.1, 1)) == len(rules)
    assert len(population(10, 1)) == 10
    print("All Assignment 11 tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    run_tests() if args.test else main()