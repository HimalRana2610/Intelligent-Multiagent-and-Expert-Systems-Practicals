"""
Experiment 4 - Reasoning Techniques
Inference Rules: Modus Ponens and Modus Tollens

Intelligent Multi-Agent and Expert Systems (AI401), Semester VII.
Standard library only.  Run:  python main.py
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

NEG = "¬"  # printed in front of a negated literal


# --------------------------------------------------------------------------
# 1.  Literals and rules  (Algorithm 1, step 1)
# --------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Literal:
    """A symbol together with a polarity, so that p and NOT p are distinct."""

    symbol: str
    negated: bool = False

    def __invert__(self) -> "Literal":
        return Literal(self.symbol, not self.negated)

    def __str__(self) -> str:
        return (NEG if self.negated else "") + self.symbol


def P(symbol: str) -> Literal:
    """The positive literal p."""
    return Literal(symbol, False)


def N(symbol: str) -> Literal:
    """The negative literal NOT p."""
    return Literal(symbol, True)


@dataclass(frozen=True)
class Rule:
    id: str
    antecedents: Tuple[Literal, ...]
    consequent: Literal

    def __str__(self) -> str:
        return f"{self.id}: {' AND '.join(map(str, self.antecedents))} -> {self.consequent}"

    def contrapositive(self) -> str:
        """Algorithm 2, step 1: negate the consequent, negate the antecedent."""
        head = " OR ".join(str(~a) for a in self.antecedents)
        return f"{~self.consequent} -> {head}"


def rule(rid: str, antecedents: Sequence[Literal], consequent: Literal) -> Rule:
    return Rule(rid, tuple(antecedents), consequent)


# --------------------------------------------------------------------------
# 2.  The knowledge base -- a car that will not start
# --------------------------------------------------------------------------

RULES: List[Rule] = [
    rule("R1", [P("battery_charged")], P("headlights_work")),
    rule("R2", [P("battery_charged"), P("starter_ok")], P("engine_cranks")),
    rule("R3", [P("engine_cranks"), P("fuel_ok")], P("engine_starts")),
    rule("R4", [P("engine_starts")], P("alternator_charges")),
    rule("R5", [P("alternator_charges")], P("battery_stays_charged")),
    rule("R6", [P("fuel_pump_ok"), P("fuel_in_tank")], P("fuel_ok")),
    rule("R7", [N("engine_starts")], P("tow_required")),
]

# Positive observations -- drive Algorithm 1.
FACTS_MP = [P("battery_charged"), P("starter_ok"), P("fuel_pump_ok"), P("fuel_in_tank")]

# A negative observation -- drives Algorithm 2.
FACTS_MT = [N("alternator_charges")]


# --------------------------------------------------------------------------
# 3.  Justifications
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    """How one literal entered the fact set."""

    literal: Literal
    kind: str                       # "given" | "MP" | "MT"
    rule_id: Optional[str]
    premises: Tuple[Literal, ...]

    def note(self) -> str:
        if self.kind == "given":
            return "given"
        return f"{self.kind} {self.rule_id} from {', '.join(map(str, self.premises))}"


@dataclass(frozen=True)
class Clause:
    """A pending disjunction of negated antecedents left by Modus Tollens."""

    rule_id: str
    trigger: Literal                # the NOT Q that fired the contrapositive
    literals: Tuple[Literal, ...]
    known_true: Tuple[Literal, ...]

    def __str__(self) -> str:
        if not self.literals:
            return f"[empty clause from {self.rule_id}]"
        return " OR ".join(map(str, self.literals))


# --------------------------------------------------------------------------
# 4.  The inference engine  (Algorithms 1 and 2)
# --------------------------------------------------------------------------

class Engine:
    """Modus Ponens forwards, Modus Tollens on the contrapositive, alternated."""

    def __init__(self, rules: Sequence[Rule], facts: Sequence[Literal]):
        self.rules = list(rules)
        self.facts: Dict[Literal, Step] = {}
        self.pending: List[Clause] = []
        self.empty_clauses: List[Clause] = []
        self.log: List[str] = []
        for f in facts:
            self._add(Step(f, "given", None, ()))

    # -- fact set ----------------------------------------------------------

    def _add(self, step: Step) -> bool:
        if step.literal in self.facts:
            return False
        self.facts[step.literal] = step
        return True

    def derived(self) -> List[Literal]:
        return [l for l, s in self.facts.items() if s.kind != "given"]

    # -- Algorithm 1 -------------------------------------------------------

    def mp_scan(self) -> List[Literal]:
        """Step 2: one complete scan of the rule base under Modus Ponens."""
        added = []
        for r in self.rules:
            if r.consequent in self.facts:
                continue                                # already believed
            if all(a in self.facts for a in r.antecedents):
                if self._add(Step(r.consequent, "MP", r.id, r.antecedents)):
                    added.append(r.consequent)
        return added

    def forward_mp(self) -> List[List[Literal]]:
        """Step 3: repeat the scan until a complete pass adds no new literal."""
        scans = []
        while True:
            added = self.mp_scan()
            scans.append(added)
            if not added:
                return scans

    # -- Algorithm 2 -------------------------------------------------------

    def mt_scan(self) -> List[Literal]:
        """Steps 1-2: match every contrapositive against the fact set.

        The contrapositive of  A1 AND ... AND An -> Q  is
        NOT Q -> NOT A1 OR ... OR NOT An.  Given NOT Q that disjunction holds;
        every antecedent already known true removes one disjunct.  One disjunct
        left is a definite literal (unit resolution); more than one is recorded
        as a pending conclusion; none left is the empty clause, i.e. a
        contradiction.
        """
        added: List[Literal] = []
        self.pending = []
        for r in self.rules:
            if ~r.consequent not in self.facts:
                continue                                # no NOT Q available
            disjuncts = tuple(~a for a in r.antecedents)
            if any(d in self.facts for d in disjuncts):
                continue                                # clause already satisfied
            known_true = tuple(a for a in r.antecedents if a in self.facts)
            open_ = tuple(d for d in disjuncts if ~d not in self.facts)
            premises = (~r.consequent,) + known_true
            if len(open_) == 1:
                if self._add(Step(open_[0], "MT", r.id, premises)):
                    added.append(open_[0])
            elif open_:
                self.pending.append(Clause(r.id, ~r.consequent, open_, known_true))
            else:
                self.empty_clauses.append(Clause(r.id, ~r.consequent, (), known_true))
        return added

    def saturate(self, verbose: bool = False) -> "Engine":
        """Step 3: alternate the two algorithms until neither adds anything."""
        round_no = 0
        while True:
            round_no += 1
            mp_added = [l for scan in self.forward_mp() for l in scan]
            mt_added = self.mt_scan()
            if verbose:
                self.log.append(
                    f"  round {round_no}: MP added {_fmt(mp_added)}   "
                    f"MT added {_fmt(mt_added)}"
                )
            if not mt_added:
                return self

    # -- reporting ---------------------------------------------------------

    def contradictions(self) -> List[str]:
        """Step 4: symbols present in the fact set together with their negation."""
        return sorted({l.symbol for l in self.facts if ~l in self.facts})

    def explain(self, literal: Literal) -> List[str]:
        """The justification chain of a literal, as indented lines."""
        lines: List[str] = []
        expanded = set()

        def walk(lit: Literal, depth: int) -> None:
            pad = "    " * depth
            step = self.facts.get(lit)
            if step is None:
                lines.append(f"{pad}{lit}   [not derived]")
                return
            if step.kind == "given":
                lines.append(f"{pad}{lit}   [given]")
                return
            lines.append(f"{pad}{lit}   [{step.kind} {step.rule_id}]")
            if lit in expanded:
                return
            expanded.add(lit)
            for p in step.premises:
                walk(p, depth + 1)

        walk(literal, 0)
        return lines


def _fmt(literals: Sequence[Literal]) -> str:
    return "[" + ", ".join(map(str, literals)) + "]"


def run(rules: Sequence[Rule], facts: Sequence[Literal], verbose: bool = False) -> Engine:
    return Engine(rules, facts).saturate(verbose=verbose)


# --------------------------------------------------------------------------
# 5.  Semantics -- truth-table checking, used to reject the fallacies
# --------------------------------------------------------------------------

def symbols_of(rules: Sequence[Rule], literals: Sequence[Literal] = ()) -> List[str]:
    s = set()
    for r in rules:
        s.update(a.symbol for a in r.antecedents)
        s.add(r.consequent.symbol)
    s.update(l.symbol for l in literals)
    return sorted(s)


def holds(lit: Literal, model: Dict[str, bool]) -> bool:
    return model[lit.symbol] != lit.negated


def rule_holds(r: Rule, model: Dict[str, bool]) -> bool:
    """P -> Q is false only in the row where P is true and Q is false."""
    return (not all(holds(a, model) for a in r.antecedents)) or holds(r.consequent, model)


def counterexample(rules, premises, goal, extra=()) -> Optional[Dict[str, bool]]:
    """Smallest model satisfying rules + premises + extra but falsifying goal.

    None means the goal is entailed: no such row of the truth table exists.
    Models are enumerated by increasing number of true symbols, so the
    counter-example returned is the smallest and hence the most readable one.
    """
    syms = symbols_of(rules, list(premises) + [goal])
    for k in range(len(syms) + 1):
        for trues in itertools.combinations(syms, k):
            model = {s: (s in trues) for s in syms}
            if not all(rule_holds(r, model) for r in rules):
                continue
            if not all(holds(p, model) for p in premises):
                continue
            if not all(c(model) for c in extra):
                continue
            if not holds(goal, model):
                return model
    return None


def entails(rules, premises, goal, extra=()) -> bool:
    return counterexample(rules, premises, goal, extra) is None


def show_model(model: Dict[str, bool], indent: str = "      ") -> List[str]:
    true = [s for s, v in model.items() if v]
    false = [s for s, v in model.items() if not v]
    return [
        f"{indent}true : {', '.join(true) if true else '(none)'}",
        f"{indent}false: {', '.join(false) if false else '(none)'}",
    ]


def report_query(label, pattern, rules, premises, goal) -> None:
    """Algorithm 2, step 4: answer a query, rejecting it if it is unsound."""
    eng = run(rules, premises)
    cx = counterexample(rules, premises, goal)
    print(f"  {label}")
    print(f"    pattern   : {pattern}")
    print(f"    premises  : {', '.join(map(str, premises))}")
    print(f"    query     : is {goal} entailed?")
    if cx is None:
        print(f"    verdict   : ENTAILED (sound) -- no interpretation falsifies it")
        note = eng.facts[goal].note() if goal in eng.facts else "not reached"
        print(f"    engine    : {note}")
    else:
        print(f"    verdict   : NOT ENTAILED -- unsound pattern, rejected")
        print(f"    engine    : {'derived' if goal in eng.facts else 'did not derive it'}")
        print(f"    counter-example (premises true, conclusion false):")
        for line in show_model(cx):
            print(line)
    print()


# --------------------------------------------------------------------------
# 6.  Exercise 1 -- a rule with three antecedent literals
# --------------------------------------------------------------------------

R8 = rule("R8",
          [P("fuel_pump_ok"), P("fuel_in_tank"), P("injectors_clean")],
          P("fuel_delivered"))

RULES_EX1 = RULES + [R8]


def exercise1() -> None:
    print(f"  new rule        {R8}")
    print(f"  contrapositive  {R8.contrapositive()}")
    print()
    print("  The negated consequent is observed, then facts are added one at a time:")
    print()

    header = f"  {'input':<26}{'pending disjunction':<52}{'definite literal'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    observations = [N("fuel_delivered")]
    collapsed_by = None
    for extra in [None, P("fuel_pump_ok"), P("fuel_in_tank")]:
        if extra is not None:
            observations = observations + [extra]
        eng = run(RULES_EX1, observations)
        pending = [c for c in eng.pending if c.rule_id == "R8"]
        definite = [l for l, s in eng.facts.items()
                    if s.kind == "MT" and s.rule_id == "R8"]
        label = f"observe {N('fuel_delivered')}" if extra is None else f"+ {extra}"
        print(f"  {label:<26}"
              f"{(str(pending[0]) if pending else '(none)'):<52}"
              f"{', '.join(map(str, definite)) if definite else '-'}")
        if definite and collapsed_by is None and extra is not None:
            collapsed_by = extra

    print()
    print(f"  At the observation the engine can record only the three-way disjunction.")
    print(f"  It collapses exactly when '{collapsed_by}' is added: at that point all but")
    print(f"  one antecedent of R8 is independently known true, so unit resolution")
    print(f"  leaves the single definite literal {N('injectors_clean')}.")
    print()
    eng = run(RULES_EX1, [N("fuel_delivered"), P("fuel_pump_ok"), P("fuel_in_tank")])
    print("  justification chain:")
    for line in eng.explain(N("injectors_clean")):
        print("    " + line)


# --------------------------------------------------------------------------
# 7.  Exercise 2 -- contradiction detection and minimal repair
# --------------------------------------------------------------------------

# One extra rule whose consequent is a *negative* literal, and one extra fact
# to fire it, so that both sides of the clash are genuinely derived.
R9 = rule("R9", [P("fuse_blown")], N("alternator_charges"))

RULES_EX2 = RULES + [R9]
FACTS_EX2 = FACTS_MP + [P("fuse_blown")]


def minimal_repairs(rules: Sequence[Rule],
                    inputs: Sequence[Literal]) -> List[Tuple[Literal, ...]]:
    """Smallest subsets of the input facts whose removal restores consistency."""
    inputs = list(inputs)
    for size in range(1, len(inputs) + 1):
        repairs = []
        for drop in itertools.combinations(inputs, size):
            kept = [f for f in inputs if f not in drop]
            eng = run(rules, kept)
            if not eng.contradictions() and not eng.empty_clauses:
                repairs.append(drop)
        if repairs:
            return repairs
    return []


def exercise2() -> None:
    print(f"  extra rule:  {R9}   (a rule whose consequent is a negative literal)")
    print("  input facts:", ", ".join(map(str, FACTS_EX2)))
    print("  (the four positive observations, plus an inspection finding: the")
    print("   alternator fuse is blown)")
    print()

    eng = run(RULES_EX2, FACTS_EX2, verbose=True)
    for line in eng.log:
        print(line)
    print()

    clashes = eng.contradictions()
    print(f"  INCONSISTENT knowledge base -- contradiction on: {', '.join(clashes)}")
    print()

    for sym in clashes:
        print(f"  --- the two justification chains that meet at '{sym}' ---")
        for lit in (P(sym), N(sym)):
            for line in eng.explain(lit):
                print("    " + line)
            print()

    if eng.empty_clauses:
        print("  Modus Tollens reports the same clash as an empty clause:")
        for c in eng.empty_clauses:
            ants = ", ".join(map(str, c.known_true))
            print(f"    {c.rule_id}: {c.trigger} holds, but every antecedent ({ants}) is")
            print(f"        known true, so no disjunct of the contrapositive survives.")
        print()

    repairs = minimal_repairs(RULES_EX2, FACTS_EX2)
    print(f"  smallest sets of input facts whose removal restores consistency")
    print(f"  (size {len(repairs[0])}, {len(repairs)} of them):")
    for r in repairs:
        print(f"    remove {{{', '.join(map(str, r))}}}")
    print()
    print("  Every repair cuts one of the two chains: dropping 'fuse_blown' removes")
    print("  the negative branch, and dropping any one of the four positive")
    print("  observations breaks the Modus Ponens chain before it reaches R4.")


# --------------------------------------------------------------------------
# 8.  Sound but not complete -- reasoning by cases
# --------------------------------------------------------------------------

RULES_CASES = [
    rule("D1", [P("stale_fuel")], P("hard_start")),
    rule("D2", [P("dirty_injectors")], P("hard_start")),
]


def cases_demo() -> None:
    for r in RULES_CASES:
        print(f"    {r}")
    print("    known: stale_fuel OR dirty_injectors   (a disjunction, not a literal)")
    print()
    eng = run(RULES_CASES, [])
    either = lambda m: m["stale_fuel"] or m["dirty_injectors"]
    print(f"    the fact set cannot hold the disjunction at all, so the input is []")
    print(f"    engine fact set after saturation        : {_fmt(list(eng.facts))}")
    print(f"    engine derives hard_start               : {P('hard_start') in eng.facts}")
    print(f"    truth table says hard_start is entailed : "
          f"{entails(RULES_CASES, [], P('hard_start'), extra=[either])}")


# --------------------------------------------------------------------------
# 9.  Demo driver
# --------------------------------------------------------------------------

def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def demo() -> None:
    section("Rule base and input facts")
    print("Rules (a list of antecedent literals, one consequent literal):")
    for r in RULES:
        print(f"  {r}")
    print("\nInput facts (positive observations):")
    for f in FACTS_MP:
        print(f"  {f}")

    # ------------------------------------------------------------ Alg. 1
    section("Algorithm 1 - deriving the closure by Modus Ponens")
    eng = Engine(RULES, FACTS_MP)
    scans = eng.forward_mp()
    for i, added in enumerate(scans, 1):
        print(f"  scan {i}: added {_fmt(added)}")
    print(f"\n  Fixed point reached after {len(scans)} scans "
          f"({len(scans) - 1} productive, the last one empty).")

    print("\n  Derived literals, each with its justification:\n")
    print(f"  {'literal':<24}{'rule':<8}{'premises used'}")
    print("  " + "-" * 64)
    for lit in eng.derived():
        s = eng.facts[lit]
        print(f"  {str(lit):<24}{s.rule_id:<8}{', '.join(map(str, s.premises))}")

    print("\n  Full justification chain for the deepest conclusion:\n")
    for line in eng.explain(P("battery_stays_charged")):
        print("    " + line)

    full = run(RULES, FACTS_MP)
    print("\n  Modus Tollens adds nothing here: with only positive observations no")
    print(f"  negated consequent is ever available.  Consistent: "
          f"{not full.contradictions()}")

    print("\n  Soundness check -- every derived literal tested against the truth")
    print(f"  table of the rule base ({2 ** len(symbols_of(RULES))} interpretations):\n")
    for lit in eng.derived():
        print(f"    {str(lit):<24} entailed = {entails(RULES, FACTS_MP, lit)}")

    # ------------------------------------------------------------ Alg. 2
    section("Algorithm 2 - reasoning from negative evidence by Modus Tollens")
    print("Step 1 - the contrapositive of every rule:\n")
    for r in RULES:
        print(f"  {r}")
        print(f"      contrapositive:  {r.contrapositive()}")

    print("\nSteps 2-3 - a negative observation, alternated with Algorithm 1:\n")
    print(f"  observation: {FACTS_MT[0]}   (the ammeter shows no charging current)\n")
    eng = run(RULES, FACTS_MT, verbose=True)
    for line in eng.log:
        print(line)

    print("\n  Derived literals, each with its justification:\n")
    print(f"  {'literal':<24}{'how':<6}{'rule':<8}{'premises used'}")
    print("  " + "-" * 68)
    for lit in eng.derived():
        s = eng.facts[lit]
        print(f"  {str(lit):<24}{s.kind:<6}{s.rule_id:<8}{', '.join(map(str, s.premises))}")

    print(f"\n  {P('tow_required')} is step 3 at work -- a literal derived by Modus")
    print("  Tollens satisfying the antecedent of a rule fired by Modus Ponens:\n")
    for line in eng.explain(P("tow_required")):
        print("    " + line)

    print("\n  Pending conclusions -- the rule matched has more than one antecedent:\n")
    for c in eng.pending:
        print(f"    {c.rule_id}: from {c.trigger} only the disjunction   {c}")
        print("        no disjunct is definite, so nothing enters the fact set")

    print("\n  Adding the single fact 'engine_cranks' collapses R3's disjunction:\n")
    eng2 = run(RULES, FACTS_MT + [P("engine_cranks")])
    for lit in eng2.derived():
        s = eng2.facts[lit]
        if s.kind == "MT":
            print(f"    {str(lit):<24}{s.kind:<6}{s.rule_id:<8}"
                  f"{', '.join(map(str, s.premises))}")
    print("\n  ...and R6 leaves a fresh two-way disjunction in its place:\n")
    for c in eng2.pending:
        print(f"    {c.rule_id}: from {c.trigger} only the disjunction   {c}")

    # ----------------------------------------------------------- queries
    section("Valid patterns and the two fallacies")
    report_query("Modus Ponens (valid)",
                 "P -> Q, P  |-  Q",
                 RULES, FACTS_MP, P("engine_starts"))
    report_query("Modus Tollens (valid)",
                 f"P -> Q, {NEG}Q  |-  {NEG}P",
                 RULES, FACTS_MT, N("engine_starts"))
    report_query("Affirming the consequent (FALLACY)",
                 "P -> Q, Q  |-  P",
                 RULES, [P("headlights_work")], P("battery_charged"))
    report_query("Denying the antecedent (FALLACY)",
                 f"P -> Q, {NEG}P  |-  {NEG}Q",
                 RULES, [N("battery_charged")], N("headlights_work"))

    section("Sound but not complete - the engine cannot reason by cases")
    cases_demo()

    # --------------------------------------------------------- exercises
    section("Exercise 1 - Modus Tollens on a rule with three antecedents")
    exercise1()

    section("Exercise 2 - contradiction detection and minimal repair")
    exercise2()


if __name__ == "__main__":
    demo()