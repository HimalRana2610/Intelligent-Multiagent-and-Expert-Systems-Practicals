"""
Experiment 5 - Uncertainty Reasoning
Inference in Bayesian Networks

Intelligent Multi-Agent and Expert Systems (AI401), Semester VII.
Standard library only.  Run:  python main.py
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

NEG = "¬"           # printed in front of a variable assigned false

T, F = True, False  # so that a conditional probability table reads as a table

Assignment = Dict[str, bool]


# --------------------------------------------------------------------------
# 1.  Nodes and conditional probability tables  (Algorithm 1, step 1)
# --------------------------------------------------------------------------

@dataclass
class Node:
    """One random variable: its parents, and P(variable = true | parents).

    Only the probability of *true* is stored for each row of the table; the
    false entry is its complement.  That is the whole economy of a Bayesian
    network -- a binary node with k parents costs 2**k numbers, not 2**(k+1).
    """

    name: str
    symbol: str                            # one letter, for readable output
    parents: Tuple[str, ...]
    cpt: Dict[Tuple[bool, ...], float]     # parent values -> P(name = true)

    def key(self, a: Assignment) -> Tuple[bool, ...]:
        """The row of the table selected by the parents' values in `a`."""
        return tuple(a[p] for p in self.parents)

    def p(self, value: bool, a: Assignment) -> float:
        """P(name = value | parents), complemented where value is false."""
        p_true = self.cpt[self.key(a)]
        return p_true if value else 1.0 - p_true

    def parent_rows(self) -> Iterator[Tuple[bool, ...]]:
        """Every combination of parent values, in a fixed order."""
        return itertools.product((T, F), repeat=len(self.parents))


def root(name: str, symbol: str, prior: float) -> Node:
    """A node with no parents: one row, the prior."""
    return Node(name, symbol, (), {(): prior})


def child(name: str, symbol: str,
          parents: Sequence[str],
          rows: Dict[Tuple[bool, ...], float]) -> Node:
    return Node(name, symbol, tuple(parents), dict(rows))


# --------------------------------------------------------------------------
# 2.  The network  (Algorithm 1, steps 2-4)
# --------------------------------------------------------------------------

class BayesNet:
    """A directed acyclic graph of nodes, held in topological order."""

    def __init__(self, nodes: Sequence[Node]):
        self.nodes = list(nodes)
        self.by_name = {n.name: n for n in self.nodes}

    # -- structure ---------------------------------------------------------

    @property
    def variables(self) -> List[str]:
        return [n.name for n in self.nodes]

    def node(self, name: str) -> Node:
        return self.by_name[name]

    def extend(self, *nodes: Node) -> "BayesNet":
        """A new network with further nodes appended (Exercise 2)."""
        return BayesNet(self.nodes + list(nodes))

    def children_of(self, name: str) -> List[str]:
        return [n.name for n in self.nodes if name in n.parents]

    def descendants_of(self, name: str) -> List[str]:
        """Transitive closure of the child relation."""
        out, frontier = [], [name]
        while frontier:
            for c in self.children_of(frontier.pop()):
                if c not in out:
                    out.append(c)
                    frontier.append(c)
        return out

    def edges(self) -> List[Tuple[str, str]]:
        return [(p, n.name) for n in self.nodes for p in n.parents]

    # -- display -----------------------------------------------------------

    def lit(self, name: str, value: bool) -> str:
        """`b` for battery_flat = true, `¬b` for battery_flat = false."""
        s = self.by_name[name].symbol.lower()
        return s if value else NEG + s

    def event(self, a: Assignment) -> str:
        return ", ".join(self.lit(v, a[v]) for v in self.variables if v in a)

    def cpt_lines(self, n: Node) -> List[Tuple[str, float]]:
        """The table of one node, as (expression, value) pairs."""
        out = []
        head = n.symbol.lower()
        for row in n.parent_rows():
            given = ", ".join(self.lit(p, v) for p, v in zip(n.parents, row))
            expr = f"P({head} | {given})" if n.parents else f"P({head})"
            out.append((expr, n.cpt[row]))
        return out

    # -- step 2: validation and the parameter count ------------------------

    def parameters(self) -> int:
        """One free number per row of every table: sum of 2**|parents|."""
        return sum(2 ** len(n.parents) for n in self.nodes)

    def joint_parameters(self) -> int:
        """What a full joint over the same variables would cost: 2**n - 1."""
        return 2 ** len(self.nodes) - 1

    def validate(self) -> List[Tuple[str, bool, str]]:
        """Every structural check the specification has to pass."""
        checks: List[Tuple[str, bool, str]] = []
        seen: List[str] = []

        ok, detail = True, "every parent precedes its child"
        for n in self.nodes:
            for p in n.parents:
                if p not in self.by_name:
                    ok, detail = False, f"{n.name} names an unknown parent {p}"
                elif p not in seen:
                    ok, detail = False, f"{n.name} precedes its parent {p}"
            seen.append(n.name)
        checks.append(("topological order, hence acyclic", ok, detail))

        bad = [n.name for n in self.nodes if set(n.cpt) != set(n.parent_rows())]
        checks.append(("one row per combination of parent values", not bad,
                       "; ".join(bad) if bad else
                       f"{self.parameters()} rows over {len(self.nodes)} nodes"))

        bad = [f"{n.name} {row}" for n in self.nodes for row, v in n.cpt.items()
               if not 0.0 <= v <= 1.0]
        checks.append(("every entry in the unit interval", not bad,
                       "; ".join(bad) if bad else "all rows within [0, 1]"))

        saved = self.joint_parameters() - self.parameters()
        checks.append(("parameters against the full joint", True,
                       f"{self.parameters()} network vs {self.joint_parameters()} "
                       f"full joint  ({saved} fewer)"))
        return checks

    # -- step 3: one entry of the full joint -------------------------------

    def joint(self, a: Assignment) -> float:
        """P(x1, ..., xn) = product over nodes of P(xi | parents(xi))."""
        return math.prod(n.p(a[n.name], a) for n in self.nodes)

    def joint_factors(self, a: Assignment) -> List[Tuple[str, float]]:
        """The same product, kept factor by factor so it can be printed."""
        out = []
        for n in self.nodes:
            head = self.lit(n.name, a[n.name])
            given = ", ".join(self.lit(p, a[p]) for p in n.parents)
            expr = f"P({head} | {given})" if n.parents else f"P({head})"
            out.append((expr, n.p(a[n.name], a)))
        return out

    # -- step 4: the sum over all assignments, and marginals ---------------

    def assignments(self, variables: Optional[Sequence[str]] = None,
                    fixed: Optional[Assignment] = None) -> Iterator[Assignment]:
        """Every assignment of `variables`, each extending `fixed`."""
        variables = self.variables if variables is None else list(variables)
        base = dict(fixed or {})
        for combo in itertools.product((T, F), repeat=len(variables)):
            a = dict(base)
            a.update(zip(variables, combo))
            yield a

    def total_probability(self) -> float:
        """Must be 1: the factorisation is a distribution, not just a product."""
        return sum(self.joint(a) for a in self.assignments())

    def marginal(self, var: str) -> Dict[bool, float]:
        """P(var), by summing the joint over every other variable."""
        out = {T: 0.0, F: 0.0}
        for a in self.assignments():
            out[a[var]] += self.joint(a)
        return out


# --------------------------------------------------------------------------
# 3.  The network -- a car that will not start
# --------------------------------------------------------------------------
#
# Two independent causes share one effect, and that effect is watched by two
# separate indicators.  The shape is deliberate: the arrows converging on C
# are what produce explaining away, and the arrows diverging from C give a
# pair of variables dependent while C is hidden and independent once it is
# observed.

DIAGRAM = """
      battery_flat   B --+
                         +--> engine_wont_crank  C --+--> warning_light  W
      starter_faulty S --+                           |
                                                     +--> tow_called     R
"""

BATTERY = root("battery_flat", "B", 0.05)
STARTER = root("starter_faulty", "S", 0.02)

CRANK = child("engine_wont_crank", "C", ["battery_flat", "starter_faulty"], {
    (T, T): 0.99,      # both faults present
    (T, F): 0.95,      # a flat battery alone
    (F, T): 0.90,      # a faulty starter alone
    (F, F): 0.01,      # neither -- some cause outside the model
})

WARNING = child("warning_light", "W", ["engine_wont_crank"], {
    (T,): 0.90,        # the lamp is not perfectly reliable
    (F,): 0.05,
})

TOW = child("tow_called", "R", ["engine_wont_crank"], {
    (T,): 0.70,        # not every stranded driver calls for a tow
    (F,): 0.01,
})

NET = BayesNet([BATTERY, STARTER, CRANK, WARNING, TOW])

# The query used throughout: how likely is a flat battery, given that the
# warning lamp is lit and a tow has been called?
QUERY = "battery_flat"
EVIDENCE = {"warning_light": T, "tow_called": T}


# --------------------------------------------------------------------------
# 4.  Posterior inference by enumeration  (Algorithm 2, steps 1-3)
# --------------------------------------------------------------------------

@dataclass
class Query:
    """The full record of one answered query, so each step can be printed."""

    variable: str
    evidence: Assignment
    hidden: List[str]
    unnormalised: Dict[bool, float]     # step 2 -- joint probabilities
    posterior: Dict[bool, float]        # step 3 -- normalised
    normaliser: float                   # step 3 -- P(evidence)
    terms: int                          # joint entries summed
    multiplications: int                # cost of the sum

    def p_true(self) -> float:
        return self.posterior[T]


def enumeration_ask(net: BayesNet, X: str, evidence: Assignment) -> Query:
    """P(X | e), by summing the joint over every assignment of the hidden set.

    Step 1 collects the hidden variables, step 2 sums the joint once for each
    value of X while holding X and e fixed, step 3 normalises.  The sum of the
    two unnormalised components is P(e) itself, so the normalising constant is
    a quantity of the model, not merely bookkeeping.
    """
    hidden = [v for v in net.variables if v != X and v not in evidence]
    unnorm: Dict[bool, float] = {}
    terms = 0
    for x in (T, F):                                            # step 2
        total = 0.0
        for a in net.assignments(hidden, fixed={**evidence, X: x}):
            total += net.joint(a)
            terms += 1
        unnorm[x] = total
    k = unnorm[T] + unnorm[F]                                   # step 3
    post = {v: unnorm[v] / k for v in (T, F)} if k else {T: 0.0, F: 0.0}
    return Query(X, dict(evidence), hidden, unnorm, post, k,
                 terms, terms * len(net.nodes))


def enumeration_ask_recursive(net: BayesNet, X: str,
                              evidence: Assignment) -> Query:
    """The same posterior, summed depth-first down the topological order.

    Arithmetically identical to `enumeration_ask`, but a factor is evaluated
    once per node of the enumeration *tree* rather than once per complete
    assignment, so common prefixes of the sum are shared.  It is here to make
    the point that the saving is only a constant factor: the tree still has
    2**|hidden| leaves.
    """
    counter = [0]

    def enum_all(variables: Sequence[str], a: Assignment) -> float:
        if not variables:
            return 1.0
        v, rest = variables[0], variables[1:]
        n = net.node(v)
        if v in a:
            counter[0] += 1
            return n.p(a[v], a) * enum_all(rest, a)
        total = 0.0
        for value in (T, F):
            counter[0] += 1
            extended = {**a, v: value}
            total += n.p(value, extended) * enum_all(rest, extended)
        return total

    hidden = [v for v in net.variables if v != X and v not in evidence]
    unnorm = {x: enum_all(net.variables, {**evidence, X: x}) for x in (T, F)}
    k = unnorm[T] + unnorm[F]
    post = {v: unnorm[v] / k for v in (T, F)} if k else {T: 0.0, F: 0.0}
    return Query(X, dict(evidence), hidden, unnorm, post, k,
                 2 ** len(hidden) * 2, counter[0])


def posterior(net: BayesNet, X: str, evidence: Assignment) -> float:
    """Shorthand for P(X = true | e)."""
    return enumeration_ask(net, X, evidence).p_true()


# --------------------------------------------------------------------------
# 5.  Comparing two posteriors  (Algorithm 2, step 4)
# --------------------------------------------------------------------------

TOL = 1e-12


def direction(before: float, after: float, tol: float = 1e-9) -> str:
    if abs(after - before) <= tol:
        return "UNCHANGED"
    return "RISES" if after > before else "FALLS"


def report_query(net: BayesNet, X: str, evidence: Assignment,
                 label: str = "", indent: str = "  ") -> Query:
    """Print one query as the three steps of Algorithm 2."""
    q = enumeration_ask(net, X, evidence)
    ev = net.event(evidence) if evidence else "(no evidence)"
    if label:
        print(f"{indent}{label}")
    print(f"{indent}  query     : P({net.node(X).symbol.lower()} | {ev})")
    print(f"{indent}  hidden    : "
          f"{', '.join(q.hidden) if q.hidden else '(none)'}"
          f"   -> {q.terms // 2} assignments per value of the query variable")
    print(f"{indent}  step 2    : unnormalised "
          f"[{q.unnormalised[T]:.8f}, {q.unnormalised[F]:.8f}]"
          f"   = [P({net.lit(X, T)}, e), P({net.lit(X, F)}, e)]")
    print(f"{indent}  step 3    : P(e) = {q.normaliser:.8f}   "
          f"posterior [{q.posterior[T]:.6f}, {q.posterior[F]:.6f}]")
    print(f"{indent}  answer    : P({net.lit(X, T)} | {ev}) = {q.p_true():.6f}")
    return q


def compare(net: BayesNet, X: str, evidence: Assignment,
            extra: Assignment, note: str, indent: str = "  ") -> Tuple[float, float]:
    """Algorithm 2, step 4: the same query with one more evidence variable."""
    before = enumeration_ask(net, X, evidence)
    after = enumeration_ask(net, X, {**evidence, **extra})
    x = net.lit(X, T)
    labels = [f"P({x} | {net.event(evidence) if evidence else '-'})",
              f"P({x} | {net.event({**evidence, **extra})})"]
    w = max(len(l) for l in labels) + 2
    d = direction(before.p_true(), after.p_true())
    delta = "" if d == "UNCHANGED" else f"  ({after.p_true() - before.p_true():+.6f})"
    print(f"{indent}{labels[0]:<{w}}= {before.p_true():.6f}"
          f"      P(e) = {before.normaliser:.8f}")
    print(f"{indent}{labels[1]:<{w}}= {after.p_true():.6f}"
          f"      P(e) = {after.normaliser:.8f}")
    print(f"{indent}{'':<{w}}  {d}{delta}")
    for line in note.splitlines():
        print(f"{indent}  {line.strip()}")
    return before.p_true(), after.p_true()


# --------------------------------------------------------------------------
# 6.  Conditional independence, checked numerically
# --------------------------------------------------------------------------

def independence_rows(net: BayesNet, x: str, y: str,
                      given: Sequence[str] = ()) -> List[Tuple[Assignment, float, float]]:
    """For every assignment z of `given`: P(x | y, z) against P(x | ¬y, z)."""
    rows = []
    for z in net.assignments(list(given)):
        if enumeration_ask(net, x, z).normaliser <= TOL:
            continue                                   # impossible conditioning
        pt = posterior(net, x, {**z, y: T})
        pf = posterior(net, x, {**z, y: F})
        rows.append((z, pt, pf))
    return rows


def independent(net: BayesNet, x: str, y: str,
                given: Sequence[str] = (), tol: float = 1e-9) -> bool:
    """True when observing y never moves the belief in x, for any z."""
    return all(abs(pt - pf) <= tol for _, pt, pf in independence_rows(net, x, y, given))


def report_independence(net: BayesNet, x: str, y: str,
                        given: Sequence[str] = (), indent: str = "    ") -> bool:
    rows = independence_rows(net, x, y, given)
    ok = independent(net, x, y, given)
    sx, sy = net.node(x).symbol, net.node(y).symbol
    cond = " | " + ", ".join(net.node(g).symbol for g in given) if given else ""
    claim = "INDEPENDENT" if ok else "DEPENDENT"
    print(f"{indent}{sx} {'_||_' if ok else '-/-'} {sy}{cond}"
          f"   {claim}")
    for z, pt, pf in rows:
        zs = net.event(z) if z else "-"
        gap = abs(pt - pf)
        print(f"{indent}    given {zs:<12}"
              f"P({net.lit(x, T)} | {net.lit(y, T)}{', ' + zs if zs != '-' else ''})"
              f" = {pt:.6f}   "
              f"P({net.lit(x, T)} | {net.lit(y, F)}{', ' + zs if zs != '-' else ''})"
              f" = {pf:.6f}   difference {gap:.6f}")
    return ok


# --------------------------------------------------------------------------
# 7.  Exercise 1 -- explaining away
# --------------------------------------------------------------------------

def exercise1() -> None:
    print("  The query variable is a cause of the observed effect; the second")
    print("  cause of that same effect is now added to the evidence.\n")

    compare(NET, QUERY, EVIDENCE, {"starter_faulty": T},
            "  the two indicators are explained by the faulty starter, so the\n"
            "  battery is no longer needed to account for them")
    print()

    print("  The same comparison with the effect itself observed, which isolates")
    print("  the mechanism from the noise in the two indicators:\n")
    compare(NET, QUERY, {"engine_wont_crank": T}, {"starter_faulty": T},
            "  a fall of roughly a factor of twelve")
    print()

    print("  Before anything is observed the two causes are strictly independent;")
    print("  once their common effect is observed they are not:\n")
    report_independence(NET, "battery_flat", "starter_faulty", ())
    print()
    report_independence(NET, "battery_flat", "starter_faulty", ["engine_wont_crank"])
    print()

    pb = NET.marginal("battery_flat")[T]
    pbc = posterior(NET, "battery_flat", {"engine_wont_crank": T})
    pbcs = posterior(NET, "battery_flat", {"engine_wont_crank": T, "starter_faulty": T})
    print(f"  prior                        P(b)          = {pb:.6f}")
    print(f"  raised by the effect         P(b | c)      = {pbc:.6f}")
    print(f"  lowered again by the rival   P(b | c, s)   = {pbcs:.6f}")
    print(f"  still above the prior, since s does not fully account for c:")
    print(f"    P(c | ¬b, s) = {CRANK.cpt[(F, T)]:.2f} < 1, so a flat battery remains")
    print(f"    a slightly better explanation than chance.")


# --------------------------------------------------------------------------
# 8.  Exercise 2 -- one new node with a single parent
# --------------------------------------------------------------------------

# A workshop test of the battery.  Its only parent is battery_flat, so it is
# a second, independent reading of the same underlying fault.
METER = child("meter_reads_low", "M", ["battery_flat"], {
    (T,): 0.95,        # a flat battery nearly always reads low
    (F,): 0.03,        # a healthy one occasionally does
})

NET2 = NET.extend(METER)


def exercise2() -> None:
    print(f"  new node: {METER.name}  ({METER.symbol}), single parent "
          f"{METER.parents[0]}\n")
    for expr, v in NET2.cpt_lines(METER):
        print(f"    {expr:<20} = {v:.4f}")
    print()

    print(f"  parameters : {NET.parameters()} -> {NET2.parameters()}   "
          f"(one row per value of the single parent)")
    print(f"  full joint : {NET.joint_parameters()} -> {NET2.joint_parameters()}   "
          f"(the joint doubles with every variable added)")
    print(f"  the sum of the joint over all "
          f"{2 ** len(NET2.nodes)} assignments is "
          f"{NET2.total_probability():.12f}\n")

    base = posterior(NET, QUERY, EVIDENCE)
    same = posterior(NET2, QUERY, EVIDENCE)
    print(f"  With the new node left unobserved the earlier posterior is")
    print(f"  unchanged: {base:.6f} in the old network, {same:.6f} in the new")
    print(f"  one -- an unobserved leaf sums out of the query entirely.\n")

    print("  The same posterior for both values of the new node:\n")
    for value in (T, F):
        q = enumeration_ask(NET2, QUERY, {**EVIDENCE, "meter_reads_low": value})
        lit = NET2.lit("meter_reads_low", value)
        print(f"    P(b | w, r, {lit:<2}) = {q.p_true():.6f}"
              f"      P(e) = {q.normaliser:.8f}"
              f"      {direction(base, q.p_true())}")
    print()
    print(f"    unobserved, for reference   P(b | w, r) = {base:.6f}")
    print()

    print("  One conditional independence the extended graph asserts:\n")
    report_independence(NET2, "meter_reads_low", "engine_wont_crank",
                        ["battery_flat"])
    print("        M's only parent is B; C is a non-descendant of M, so once B")
    print("        is known, C carries no further information about the meter.")
    print()

    print("  One pair that remains dependent:\n")
    report_independence(NET2, "meter_reads_low", "engine_wont_crank", ())
    print("        With B hidden the path M <- B -> C is open: the meter reading")
    print("        is evidence about the battery, and the battery is a cause of C.")


# --------------------------------------------------------------------------
# 9.  Demo driver
# --------------------------------------------------------------------------

def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def show_network(net: BayesNet) -> None:
    print(f"  {'variable':<20}{'sym':<5}{'parents':<34}{'conditional probability table'}")
    print("  " + "-" * 96)
    for n in net.nodes:
        parents = ", ".join(n.parents) if n.parents else "(none)"
        lines = net.cpt_lines(n)
        for i, (expr, v) in enumerate(lines):
            if i == 0:
                print(f"  {n.name:<20}{n.symbol:<5}{parents:<34}{expr:<22} = {v:.4f}")
            else:
                print(f"  {'':<20}{'':<5}{'':<34}{expr:<22} = {v:.4f}")


def demo() -> None:
    # ------------------------------------------------------------- Alg. 1
    section("Algorithm 1, step 1 - the network as a DAG with a table per node")
    print(DIAGRAM)
    print("  Edges (parent -> child):")
    for p, c in NET.edges():
        print(f"    {p} -> {c}")
    print()
    show_network(NET)
    print()
    print("  Reading of the structure:")
    print("    B and S are root causes, independent of one another a priori.")
    print("    C is their common effect -- the converging arrows.")
    print("    W and R are two noisy indicators of C, and C is their only parent.")

    section("Algorithm 1, step 2 - validating the specification")
    print(f"  {'check':<44}{'':<6}{'detail'}")
    print("  " + "-" * 96)
    for name, ok, detail in NET.validate():
        print(f"  {name:<44}{'OK' if ok else 'FAIL':<6}{detail}")
    print()
    print("  Where the saving comes from, node by node:\n")
    print(f"  {'variable':<20}{'parents':<10}{'rows = 2^parents':<20}")
    print("  " + "-" * 52)
    for n in NET.nodes:
        print(f"  {n.name:<20}{len(n.parents):<10}{2 ** len(n.parents):<20}")
    print("  " + "-" * 52)
    print(f"  {'total':<20}{'':<10}{NET.parameters():<20}")
    print()
    print(f"  A full joint over {len(NET.nodes)} binary variables needs "
          f"2^{len(NET.nodes)} - 1 = {NET.joint_parameters()} numbers.")
    print(f"  The network needs {NET.parameters()}, a saving of "
          f"{NET.joint_parameters() - NET.parameters()}.  The saving is bought by the")
    print("  missing edges: W and R each have one parent instead of four.")

    section("Algorithm 1, step 3 - one entry of the full joint")
    entry = {"battery_flat": T, "starter_faulty": F, "engine_wont_crank": T,
             "warning_light": T, "tow_called": T}
    print(f"  assignment: {NET.event(entry)}")
    print("  (the battery is flat, the starter is sound, the engine will not")
    print("   crank, the lamp is lit and a tow has been called)\n")
    factors = NET.joint_factors(entry)
    width = max(len(e) for e, _ in factors)
    for i, (expr, v) in enumerate(factors):
        print(f"    {'P =' if i == 0 else '':<4}{expr:<{width}}  = {v:.4f}")
    print(f"    {'':<4}{'-' * (width + 11)}")
    print(f"    {'':<4}{'product':<{width}}  = {NET.joint(entry):.10f}")
    print("\n  One multiplication per node, and the row of each table is picked")
    print("  out by the values the assignment gives to that node and its parents.")

    section("Algorithm 1, step 4 - the factorisation is a distribution")
    total = NET.total_probability()
    print(f"  sum of the joint over all 2^{len(NET.nodes)} = "
          f"{2 ** len(NET.nodes)} assignments = {total:.12f}")
    print(f"  equals 1 to within {abs(total - 1.0):.2e}\n")
    print("  Marginals, each obtained by summing the joint over the other four:\n")
    print(f"  {'variable':<22}{'P(true)':<14}{'P(false)'}")
    print("  " + "-" * 50)
    for v in NET.variables:
        m = NET.marginal(v)
        print(f"  {v:<22}{m[T]:<14.6f}{m[F]:.6f}")
    print()
    print("  The roots return their priors exactly, which is the check that the")
    print("  summing-out is correct: P(b) = 0.05 and P(s) = 0.02 as specified.")

    # ------------------------------------------------------------- Alg. 2
    section("Algorithm 2, steps 1-3 - the posterior of the query variable")
    print("  The car will not start.  The warning lamp is lit and a tow has been")
    print("  called.  How likely is a flat battery?\n")
    q = report_query(NET, QUERY, EVIDENCE)
    print()
    print(f"  step 1    : the query variable is {QUERY}, the evidence is")
    print(f"              {NET.event(EVIDENCE)}, and the hidden set is "
          f"{', '.join(q.hidden)}")
    print(f"  step 2    : {q.terms} joint entries summed "
          f"({q.terms // 2} per value of the query variable),")
    print(f"              {q.multiplications} multiplications in all")
    print(f"  step 3    : the normalising constant {q.normaliser:.8f} is P(w, r) --")
    print("              the probability of the evidence itself, not a scaling")
    print("              convenience; it is what a joint query would have returned")
    print()
    print("  The unnormalised components before division, term by term:\n")
    print(f"  {'assignment of the hidden set':<40}{'joint with b':<18}{'joint with ¬b'}")
    print("  " + "-" * 76)
    for h in NET.assignments(q.hidden):
        with_b = NET.joint({**EVIDENCE, **h, QUERY: T})
        without = NET.joint({**EVIDENCE, **h, QUERY: F})
        print(f"  {NET.event(h):<40}{with_b:<18.10f}{without:.10f}")
    print("  " + "-" * 76)
    print(f"  {'sum':<40}{q.unnormalised[T]:<18.10f}{q.unnormalised[F]:.10f}")
    print()
    print("  Cross-check against the depth-first form of the same sum:\n")
    r = enumeration_ask_recursive(NET, QUERY, EVIDENCE)
    print(f"    flat sum over assignments : {q.p_true():.12f}   "
          f"{q.multiplications} multiplications")
    print(f"    depth-first over the DAG  : {r.p_true():.12f}   "
          f"{r.multiplications} multiplications")
    print(f"    agree to within {abs(q.p_true() - r.p_true()):.2e}")
    print("    Sharing common prefixes of the sum saves a constant factor only;")
    print("    the enumeration tree still has one leaf per hidden assignment.")

    section("Algorithm 2, step 4 - the same query with one more observation")
    print("  Three queries, one for each direction the belief can move.\n")

    print("  (a) evidence that supports the query variable\n")
    compare(NET, QUERY, {"warning_light": T}, {"tow_called": T},
            "  a second indicator of the same effect raises the belief in the cause",
            indent="      ")
    print()

    print("  (b) evidence that competes with the query variable\n")
    compare(NET, QUERY, EVIDENCE, {"starter_faulty": T},
            "  a rival cause of the same effect lowers it -- explaining away",
            indent="      ")
    print()

    print("  (c) evidence that is screened off from the query variable\n")
    compare(NET, "tow_called", {"engine_wont_crank": T}, {"battery_flat": T},
            "  no change at all: R's only parent C is already observed, and B is a\n"
            "      non-descendant of R, so C blocks the path B -> C -> R entirely",
            indent="      ")

    section("The independences the graph asserts, verified numerically")
    print("  A variable is independent of its non-descendants given its parents.")
    print("  Each claim below is checked by computing the two posteriors.\n")

    print("  1.  The two root causes, with nothing observed:\n")
    report_independence(NET, "battery_flat", "starter_faulty", ())
    print()
    print("  2.  An indicator and a cause, once the effect between them is known:\n")
    report_independence(NET, "tow_called", "battery_flat", ["engine_wont_crank"])
    print()
    print("  3.  The two indicators, once their shared parent is known:\n")
    report_independence(NET, "tow_called", "warning_light", ["engine_wont_crank"])
    print()
    print("  4.  ...and the same pair with that parent hidden:\n")
    report_independence(NET, "tow_called", "warning_light", ())
    print("        Two effects of one cause are dependent while the cause is")
    print("        hidden -- each is evidence for it -- and independent once it")
    print("        is observed.  Case 3 and case 4 are the same pair of variables.")

    section("Why enumeration is exponential in the hidden variables")
    print("  The same query, answered with successively more of the network")
    print("  hidden.  Each hidden variable doubles the sum.\n")
    print(f"  {'evidence':<26}{'hidden':<8}{'terms':<10}"
          f"{'multiplications':<20}{'P(b | e)'}")
    print("  " + "-" * 78)
    trials = [
        {"warning_light": T, "tow_called": T, "starter_faulty": T,
         "engine_wont_crank": T},
        {"warning_light": T, "tow_called": T, "engine_wont_crank": T},
        {"warning_light": T, "tow_called": T},
        {"warning_light": T},
        {},
    ]
    for e in trials:
        qq = enumeration_ask(NET, QUERY, e)
        label = NET.event(e) if e else "(none)"
        print(f"  {label:<26}{len(qq.hidden):<8}{qq.terms:<10}"
              f"{qq.multiplications:<20}{qq.p_true():.6f}")
    print()
    print("  terms = 2 x 2^|hidden|, and each term is a product over all n nodes,")
    print(f"  so the work is n x 2^(|hidden| + 1) with n = {len(NET.nodes)}.  The last row")
    print("  is the prior: with no evidence the enumeration sums the entire joint")
    print("  and returns P(b) = 0.05, which is the marginal computed in step 4.")

    # --------------------------------------------------------- exercises
    section("Exercise 1 - a second cause of the same observed effect")
    exercise1()

    section("Exercise 2 - extending the network with a new node")
    exercise2()


if __name__ == "__main__":
    demo()
