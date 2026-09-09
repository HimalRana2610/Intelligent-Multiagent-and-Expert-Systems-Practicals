# Experiment 2 — Production Rules and the Match Phase

**Intelligent Multi-Agent and Expert Systems (AI401)** · B.Tech IV (AI), Semester VII
Run with `python app.py` (standard library only).

---

## Aim

To represent domain knowledge as IF–THEN production rules, hold facts in a working memory (WM), and compute the conflict set of applicable rules.

## Theory

A production system has three parts:

| Part | In the code | Role |
|---|---|---|
| Rule base | `RULES` — a list of `Rule` records | Long-term knowledge, IF–THEN productions |
| Working memory | `wm` — a Python `set` of fact strings | Short-term memory: what is believed *right now* |
| Inference engine | `match` / `forward_chain` | Runs the recognise–act cycle |

The engine repeats a **recognise–act cycle**: **match** every rule against WM, **select** one rule from the resulting conflict set, **act** by asserting its conclusion into WM, and repeat until the conflict set is empty.

This experiment isolates the **match** step. A rule is applicable when

1. every condition in its IF part is present in WM, **and**
2. its conclusion is *not* already in WM.

Condition 2 is **refractoriness**: it stops a rule from firing over and over on data it has already accounted for. The set of applicable rules is the **conflict set**; deciding which of them to fire is Experiment 5 (here selection is deliberately trivial — first rule in textual order).

## Algorithm

1. Encode each rule as a record `(id, conditions, conclusion)`.
2. Initialise WM as a set of facts.
3. For each rule, test that all conditions are members of WM and that the conclusion is absent.
4. Collect and display the conflict set.

---

## The code

### Representing a rule

```python
@dataclass(frozen=True)
class Rule:
    id: str
    conditions: Tuple[str, ...]
    conclusion: str
```

`frozen=True` plus a tuple of conditions makes rules immutable and hashable — a rule base is knowledge, not state; the only thing that changes during a run is WM.

The rule base is Winston's classic animal-identification set. It is chosen because the conclusions of the early rules (`mammal`, `carnivore`) are the conditions of later ones, so the rules genuinely chain instead of all matching at once:

```
R1 : IF has_hair                                    THEN mammal
R5 : IF mammal AND eats_meat                        THEN carnivore
R8 : IF carnivore AND tawny_colour AND dark_spots   THEN cheetah
```

### The matcher (step 3 of the algorithm)

```python
def is_applicable(rule, wm):
    return all(holds(c, wm) for c in rule.conditions) and rule.conclusion not in wm

def match(rules, wm):
    return [rule for rule in rules if is_applicable(rule, wm)]
```

`match` is a single linear scan of the rule base — the *naïve* matcher. Its simplicity is the point: Exercise 3 measures exactly what it costs.

Testing `all(...)` short-circuits on the first failing condition, which is why the measured cost per rule drops slightly as the rule base grows (more rules fail early).

### Working memory as a `set`

Membership testing is the inner-loop operation of the whole experiment, so WM is a `set` (average O(1) lookup) rather than a list (O(|WM|)). With a list, the timings in Exercise 3 would grow with the size of WM as well as with the rule count.

---

## Sample output — the match phase

```
WM = ['dark_spots', 'eats_meat', 'has_hair', 'mammal', 'tawny_colour']
conflict set = [R5]
```

Only R5 (`mammal AND eats_meat → carnivore`) is applicable. Note what is *absent*: R1 (`has_hair → mammal`) has all its conditions satisfied, but `mammal` is already in WM, so refractoriness removes it. R8 is not applicable either — `carnivore` has not been derived yet.

### The conflict set in use (recognise–act)

```
cycle 1: conflict set = [R1]  -> fire R1, assert 'mammal'
cycle 2: conflict set = [R5]  -> fire R5, assert 'carnivore'
cycle 3: conflict set = [R8]  -> fire R8, assert 'cheetah'
cycle 4: conflict set empty   -> halt
```

Starting from four observations the system identifies a cheetah in three cycles and halts by itself.

With refractoriness switched off (`forward_chain(..., refractory=False)`), the same run never terminates:

```
cycle 1: conflict set = [R1]      -> fire R1, assert 'mammal'
cycle 2: conflict set = [R1, R5]  -> fire R1, assert 'mammal'
cycle 3: conflict set = [R1, R5]  -> fire R1, assert 'mammal'
stopped after 4 cycles (did not halt)
```

R1 stays in the conflict set forever, is re-selected forever, and re-asserts a fact that is already there. This is the concrete justification for condition 2 of the matcher.

---

## Exercise 1 — adding facts one at a time

`trace_additions()` asserts facts one by one and prints the conflict set plus the delta after each addition:

```
WM = {}              conflict set = [(empty)]
+ has_hair           conflict set = [R1]        | +R1
+ mammal             conflict set = [(empty)]   | -R1
+ eats_meat          conflict set = [R5]        | +R5
+ carnivore          conflict set = [(empty)]   | -R5
+ tawny_colour       conflict set = [(empty)]
+ dark_spots         conflict set = [R8]        | +R8
+ cheetah            conflict set = [(empty)]   | -R8
```

**Observations**

- The conflict set **grows** when a fact completes the last missing condition of some rule (`has_hair` → +R1, `dark_spots` → +R8).
- It **shrinks** when a fact matches a rule's *conclusion* — refractoriness retires that rule (`mammal` → −R1, `cheetah` → −R8).
- Some facts change nothing (`tawny_colour`): R8 also needs `dark_spots`, so no rule flips state.
- The conflict set is small at every step even though the rule base is not. Adding one fact affects only the handful of rules that mention it — the observation that RETE is built on.
- With only positive conditions, matching is **monotonic** in this sense: adding a fact can never falsify a condition. A rule can only leave the conflict set via refractoriness. Exercise 2 breaks that.

## Exercise 2 — negated conditions

The matcher is extended in one place only, by making the per-condition test a function:

```python
def holds(condition, wm):
    if condition.startswith("NOT "):
        return condition[4:].strip() not in wm     # negation as failure
    return condition in wm
```

`is_applicable` then calls `holds(c, wm)` instead of `c in wm`; nothing else changes. Three rules exercise it:

```
R14: IF mammal AND NOT eats_meat AND NOT pointed_teeth  THEN herbivore
R15: IF bird AND NOT flies                              THEN flightless_bird
R16: IF flightless_bird AND long_neck AND long_legs     THEN ostrich
```

`NOT f` succeeds when `f` is **absent from WM** — this is *negation as failure* under the **closed-world assumption**: what is not believed is taken to be false. The system cannot distinguish "known not to eat meat" from "nothing is known about its diet", and that is a real modelling commitment, not an implementation detail.

The consequence is **non-monotonicity** — adding a fact can now *remove* a rule from the conflict set:

```
WM = ['black_stripes', 'gives_milk', 'hoofs', 'mammal']
conflict set = [R7, R14]                       # R14: herbivore

WM = [..., 'eats_meat']                        # one new fact asserted
conflict set = [R5, R7]                        # R14 has gone, R5 appeared
```

A previously derivable conclusion (`herbivore`) becomes underivable when knowledge *increases*. Classical logic cannot do this; production systems can, which is what makes them usable for default reasoning — and what makes truth maintenance necessary if the conclusion had already been asserted and used.

The chained flightless-bird case shows both effects together:

```
+ has_feathers       conflict set = [R3]     | +R3
+ bird               conflict set = [R15]    | +R15 -R3     (R3 retired; R15's 'NOT flies' holds)
+ flightless_bird    conflict set = [(empty)]| -R15
+ long_neck          conflict set = [(empty)]
+ long_legs          conflict set = [R16]    | +R16
```

## Exercise 3 — how match time scales

`generate_rules(n)` builds `n` synthetic rules, each with 3 conditions sampled from a 200-fact vocabulary; WM holds 100 of those facts, so a rule matches with probability ≈ (1/2)³ = 12.5 %. Each measurement is averaged over enough repeats to be stable (`time.perf_counter`).

**One match against the whole rule base**

| rules | match (ms) | per rule (µs) | \|conflict set\| |
|---:|---:|---:|---:|
| 10 | 0.006 | 0.586 | 1 |
| 100 | 0.061 | 0.612 | 10 |
| 1 000 | 0.535 | 0.535 | 116 |
| 5 000 | 2.424 | 0.485 | 627 |

Cost per rule is essentially constant, so a single match is **O(R × C)** in the number of rules R and conditions per rule C. At 1 000 rules one match takes about half a millisecond — that alone looks harmless.

**A whole run, where match is repeated every cycle**

| rules | cycles | total (ms) | rule tests |
|---:|---:|---:|---:|
| 100 | 10 | 0.56 | 1 000 |
| 400 | 48 | 11.0 | 19 200 |
| 1 600 | 187 | 158.5 | 299 200 |

This is the real result. Quadrupling the rule base multiplied the run time by ~14×, not 4×, because the number of cycles grows with the rule base *and* every cycle re-matches every rule. Total work is **cycles × rules**, i.e. **quadratic**.

*(Absolute times are machine-dependent; the growth ratios are the point.)*

### Why RETE exists

The naïve matcher throws away everything it learned on the previous cycle. Firing one rule typically changes **one** fact in a WM of hundreds, yet all R rules are re-tested from scratch — and most of them do not even mention the fact that changed. Two structural sources of waste:

- **Temporal redundancy** — WM barely changes between cycles, but matching restarts each time.
- **Structural similarity** — many rules share conditions (`mammal` appears in R5, R6, R7, R14 above); the naïve matcher tests it once per rule, every cycle.

RETE (Forgy, 1979) attacks both. It compiles the conditions of the rule base once into a dataflow network: alpha nodes test individual conditions and are **shared** by every rule that uses them (structural similarity solved), while beta nodes join partial matches and **store** them in memories that persist across cycles. Only the *delta* — the tokens for facts added to or removed from WM — is propagated through the network, so cost depends on the number of WM changes rather than on the size of the rule base. In exchange for memory (the stored partial matches) and complexity on retraction, matching becomes roughly independent of R, which is what makes rule bases of thousands of productions (OPS5, CLIPS, Drools, Jess) practical.

---

## Viva questions

**1. Describe the recognise–act cycle.**
The inference engine's main loop. **Recognise (match):** test every rule in the rule base against working memory and collect the applicable ones into the conflict set. **Conflict resolution:** select one rule from that set by some strategy — textual order (used here), specificity, recency, priority/salience, or refractoriness. **Act:** execute the chosen rule's right-hand side, which asserts, modifies, or retracts facts in WM. Then repeat. The cycle halts when the conflict set is empty (or a halt action fires). Because WM changed, the next match can yield a different conflict set — this is how forward chaining propagates conclusions, as in `has_hair → mammal → carnivore → cheetah` above.

**2. What is refractoriness and why is it needed?**
Refractoriness stops a rule from firing repeatedly on the same data — here, implemented as the check `rule.conclusion not in wm`. Without it a rule whose conditions stay satisfied stays in the conflict set forever, is re-selected, and re-asserts a fact that is already believed: the system loops without making progress, as the `refractory=False` run demonstrates. It is needed both for **termination** and for **efficiency** — the engine should react to *new* information rather than re-deriving what it already believes. (Production systems such as OPS5 implement it more generally, by refusing to re-fire a rule on the same *instantiation* — the same rule with the same matched facts — which also allows the same rule to fire again on genuinely different data.)

**3. What problem does the RETE matching algorithm solve?**
The cost of the match phase, which dominates a production system's run time (typically ~90 %). Naïve matching is O(R × C) per cycle and O(cycles × R × C) per run — the quadratic growth measured in Exercise 3. RETE exploits temporal redundancy (WM changes little between cycles) and structural similarity (rules share conditions) by compiling the rule base into a network with shared condition tests and persistent partial-match memories, then propagating only WM changes through it. Matching then scales with the number of facts that changed instead of with the size of the rule base, at the cost of extra memory and more expensive retraction.

---

## File map

| Item in `app.py` | Purpose |
|---|---|
| `Rule` | Rule record — id, conditions, conclusion (algorithm step 1) |
| `RULES` | Animal-identification rule base |
| `holds` | Single-condition test, including `NOT` (Exercise 2) |
| `is_applicable` | All conditions hold **and** conclusion absent (step 3) |
| `match` | Computes the conflict set (steps 3–4) |
| `forward_chain` | Recognise–act cycle; `refractory=False` shows the failure mode |
| `trace_additions` | Exercise 1 — incremental facts, conflict-set delta |
| `RULES_NEG` | Exercise 2 — rules with negated conditions |
| `generate_rules`, `benchmark`, `benchmark_cycle` | Exercise 3 — scaling measurements |
| `demo` | Prints all of the above |
