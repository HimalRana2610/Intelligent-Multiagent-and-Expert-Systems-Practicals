import random
import time
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Rule:
    id: str
    conditions: Tuple[str, ...]
    conclusion: str

    def __str__(self):
        return f"{self.id}: IF {' AND '.join(self.conditions)} THEN {self.conclusion}"

RULES = [
    Rule("R1", ("has_hair",), "mammal"),
    Rule("R2", ("gives_milk",), "mammal"),
    Rule("R3", ("has_feathers",), "bird"),
    Rule("R4", ("flies", "lays_eggs"), "bird"),
    Rule("R5", ("mammal", "eats_meat"), "carnivore"),
    Rule("R6", ("mammal", "pointed_teeth", "claws"), "carnivore"),
    Rule("R7", ("mammal", "hoofs"), "ungulate"),
    Rule("R8", ("carnivore", "tawny_colour", "dark_spots"), "cheetah"),
    Rule("R9", ("carnivore", "tawny_colour", "black_stripes"), "tiger"),
    Rule("R10", ("ungulate", "long_neck", "long_legs", "dark_spots"), "giraffe"),
    Rule("R11", ("ungulate", "black_stripes"), "zebra"),
    Rule("R12", ("bird", "swims", "black_and_white"), "penguin"),
    Rule("R13", ("bird", "good_flyer"), "albatross"),
]


# ---------------------------------------------------------------------------
# The match phase
# ---------------------------------------------------------------------------

def holds(condition, wm):
    """True if a single condition is satisfied by working memory.

    Exercise 2: a condition written 'NOT fact' holds when fact is absent from
    WM (negation as failure / closed-world assumption).
    """
    if condition.startswith("NOT "):
        return condition[4:].strip() not in wm
    return condition in wm


def is_applicable(rule, wm):
    """A rule is applicable when every condition holds and its conclusion is
    not already believed (refractoriness)."""
    return all(holds(c, wm) for c in rule.conditions) and rule.conclusion not in wm


def match(rules, wm):
    """The conflict set: every rule applicable to the current WM."""
    return [rule for rule in rules if is_applicable(rule, wm)]


def show(conflict_set):
    if not conflict_set:
        return "(empty)"
    return ", ".join(rule.id for rule in conflict_set)


# ---------------------------------------------------------------------------
# Recognise-act cycle (match -> select -> act), included to show what the
# conflict set is *for*. Selection here is trivially "first rule in order".
# ---------------------------------------------------------------------------

def forward_chain(rules, wm, refractory=True, max_cycles=20, verbose=True):
    wm = set(wm)
    for cycle in range(1, max_cycles + 1):
        if refractory:
            conflict_set = match(rules, wm)
        else:
            # No refractoriness: conclusion already in WM no longer blocks the rule.
            conflict_set = [r for r in rules if all(holds(c, wm) for c in r.conditions)]

        if not conflict_set:
            if verbose:
                print(f"  cycle {cycle}: conflict set empty -> halt")
            return wm, cycle - 1

        chosen = conflict_set[0]  # selection strategy: textual order
        if verbose:
            print(
                f"  cycle {cycle}: conflict set = [{show(conflict_set)}]"
                f" -> fire {chosen.id}, assert '{chosen.conclusion}'"
            )
        wm.add(chosen.conclusion)

    if verbose:
        print(f"  stopped after {max_cycles} cycles (did not halt)")
    return wm, max_cycles


# ---------------------------------------------------------------------------
# Exercise 1: add facts one at a time and watch the conflict set change
# ---------------------------------------------------------------------------

def trace_additions(rules, facts, wm=None):
    wm = set() if wm is None else set(wm)
    previous = {rule.id for rule in match(rules, wm)}
    print(f"  WM = {{}}{'':<28} conflict set = [{show(match(rules, wm))}]")

    for fact in facts:
        wm.add(fact)
        current_rules = match(rules, wm)
        current = {rule.id for rule in current_rules}
        gained = sorted(current - previous)
        lost = sorted(previous - current)
        delta = ""
        if gained:
            delta += " +" + ",".join(gained)
        if lost:
            delta += " -" + ",".join(lost)
        print(
            f"  + {fact:<30} conflict set = [{show(current_rules)}]"
            f"{'  |' + delta if delta else ''}"
        )
        previous = current
    return wm


# ---------------------------------------------------------------------------
# Exercise 2: rules that use negated conditions
# ---------------------------------------------------------------------------

RULES_NEG = RULES + [
    Rule("R14", ("mammal", "NOT eats_meat", "NOT pointed_teeth"), "herbivore"),
    Rule("R15", ("bird", "NOT flies"), "flightless_bird"),
    Rule("R16", ("flightless_bird", "long_neck", "long_legs"), "ostrich"),
]


# ---------------------------------------------------------------------------
# Exercise 3: how match time scales with the size of the rule base
# ---------------------------------------------------------------------------

def generate_rules(n_rules, n_facts=200, conds_per_rule=3, seed=7):
    rng = random.Random(seed)
    return [
        Rule(
            f"G{i}",
            tuple(f"f{j}" for j in rng.sample(range(n_facts), conds_per_rule)),
            f"c{i}",
        )
        for i in range(n_rules)
    ]


def benchmark(sizes=(10, 100, 1000, 5000), n_facts=200, conds_per_rule=3):
    wm = {f"f{i}" for i in range(0, n_facts, 2)}  # half the facts are believed
    print(f"  WM holds {len(wm)} facts; {conds_per_rule} conditions per rule")
    print(f"  {'rules':>7}  {'match (ms)':>11}  {'per rule (us)':>14}  {'|conflict set|':>15}")

    for n in sizes:
        rules = generate_rules(n, n_facts, conds_per_rule)
        repeats = max(1, 200000 // (n * conds_per_rule))

        start = time.perf_counter()
        for _ in range(repeats):
            conflict_set = match(rules, wm)
        elapsed = (time.perf_counter() - start) / repeats

        print(
            f"  {n:>7}  {elapsed * 1e3:>11.3f}  {elapsed / n * 1e6:>14.3f}"
            f"  {len(conflict_set):>15}"
        )


def benchmark_cycle(sizes=(100, 400, 1600), n_facts=200, conds_per_rule=3):
    """One match is linear in the rule count, but a full run re-matches every
    rule on every cycle -- so the total work is (cycles x rules)."""
    wm = {f"f{i}" for i in range(0, n_facts, 2)}
    print(f"  {'rules':>7}  {'cycles':>7}  {'total (ms)':>11}  {'rule tests':>11}")

    for n in sizes:
        rules = generate_rules(n, n_facts, conds_per_rule)
        start = time.perf_counter()
        _, cycles = forward_chain(rules, wm, max_cycles=10**6, verbose=False)
        elapsed = time.perf_counter() - start
        print(f"  {n:>7}  {cycles:>7}  {elapsed * 1e3:>11.3f}  {cycles * n:>11,}")


# ---------------------------------------------------------------------------

def demo():
    print("=" * 70)
    print("Rule base")
    print("=" * 70)
    for rule in RULES:
        print(f"  {rule}")

    print("\n" + "=" * 70)
    print("Match phase: conflict set for a fixed working memory")
    print("=" * 70)
    wm = {"has_hair", "eats_meat", "tawny_colour", "dark_spots", "mammal"}
    print(f"  WM = {sorted(wm)}")
    print(f"  conflict set = [{show(match(RULES, wm))}]")
    print("  R1 is NOT in the set: its conclusion 'mammal' is already in WM (refractoriness).")

    print("\n" + "=" * 70)
    print("Exercise 1: adding facts one at a time")
    print("=" * 70)
    trace_additions(
        RULES,
        [
            "has_hair",
            "mammal",       # conclusion asserted -> R1 leaves the set, R5/R6/R7 become reachable
            "eats_meat",
            "carnivore",    # -> R5 leaves the set, R8/R9 become reachable
            "tawny_colour",
            "dark_spots",
            "cheetah",      # -> R8 leaves the set, nothing left to derive
        ],
    )
    print("  Grows when a fact completes a rule's IF part;")
    print("  shrinks when a fact matches a rule's conclusion (refractoriness).")

    print("\n" + "=" * 70)
    print("Recognise-act cycle using the conflict set")
    print("=" * 70)
    facts = {"has_hair", "eats_meat", "tawny_colour", "dark_spots"}
    print(f"  WM = {sorted(facts)}")
    final, cycles = forward_chain(RULES, facts)
    print(f"  derived after {cycles} cycles: {sorted(final - facts)}")

    print("\n  Without refractoriness (same WM, R1 keeps re-firing):")
    forward_chain(RULES, facts, refractory=False, max_cycles=4)

    print("\n" + "=" * 70)
    print("Exercise 2: negated conditions ('NOT fact')")
    print("=" * 70)
    for rule in RULES_NEG[len(RULES):]:
        print(f"  {rule}")

    print("\n  Herbivore case -- WM never mentions eats_meat, so 'NOT eats_meat' holds:")
    wm = {"gives_milk", "mammal", "hoofs", "black_stripes"}
    print(f"  WM = {sorted(wm)}")
    print(f"  conflict set = [{show(match(RULES_NEG, wm))}]")

    print("\n  Now assert eats_meat -- a *new* fact makes R14 drop out (non-monotonic):")
    wm.add("eats_meat")
    print(f"  WM = {sorted(wm)}")
    print(f"  conflict set = [{show(match(RULES_NEG, wm))}]")

    print("\n  Flightless bird chain:")
    trace_additions(RULES_NEG, ["has_feathers", "bird", "flightless_bird", "long_neck", "long_legs"])

    print("\n" + "=" * 70)
    print("Exercise 3: match time vs. rule-base size (time.perf_counter)")
    print("=" * 70)
    benchmark()
    print("\n  Cost per rule is roughly constant -> one match is O(R x C).")

    print("\n  Cost of a whole run (match repeated every cycle):")
    benchmark_cycle()
    print("\n  Rule tests grow as cycles x rules, and each WM change re-tests every")
    print("  rule from scratch even though almost nothing changed. RETE removes that")
    print("  waste: it compiles the conditions into a network, shares tests common to")
    print("  several rules, stores partial matches, and propagates only the WM delta.")


if __name__ == "__main__":
    demo()
