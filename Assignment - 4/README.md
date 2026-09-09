# Experiment 4 — Reasoning Techniques: Modus Ponens and Modus Tollens

**Intelligent Multi-Agent and Expert Systems (AI401)** · B.Tech IV (AI), Semester VII
Run with `python main.py` (standard library only, ~1 s).

---

## Aim

To apply the sound inference rules Modus Ponens and Modus Tollens to a propositional knowledge base of implications, to derive the set of facts entailed by a given set of observations, and to distinguish these valid patterns from the fallacies of affirming the consequent and denying the antecedent.

## Theory

Experiment 1 showed how knowledge is *encoded*; encoding alone derives nothing. An **inference rule** is a syntactic pattern that licenses writing down a new sentence when sentences matching its premises are already present. A rule is **sound** when every interpretation satisfying the premises also satisfies the conclusion, so a sound rule can never turn true knowledge into a false conclusion.

Soundness is a property of the *pattern alone*, not of the domain — which is why an inference engine can be written once and reused over any knowledge base. This program takes that literally: the engine never inspects a symbol name, and the same code runs the car-diagnosis rule base, the three-antecedent rule of Exercise 1 and the inconsistent rule base of Exercise 2 without alteration.

### The two valid patterns

| | premises | conclusion |
|---|---|---|
| **Modus Ponens** | `P → Q`, `P` | `Q` |
| **Modus Tollens** | `P → Q`, `¬Q` | `¬P` |

Both follow from one line of the truth table. `P → Q` is false only in the row where `P` is true and `Q` is false; each premise pair eliminates that row, leaving the conclusion true in every surviving row.

| P | Q | P → Q | |
|---|---|---|---|
| T | T | T | ← the only row Modus Ponens leaves |
| T | F | **F** | eliminated by the implication itself |
| F | T | T | |
| F | F | T | ← the only row Modus Tollens leaves |

Modus Tollens is Modus Ponens applied to the **contrapositive** `¬Q → ¬P`, and that equivalence is what an engine actually implements: it stores each rule once and matches it in either direction.

### The two fallacies

| | premises | invalid conclusion |
|---|---|---|
| **Affirming the consequent** | `P → Q`, `Q` | `P` |
| **Denying the antecedent** | `P → Q`, `¬P` | `¬Q` |

Both fail for the same reason: an implication permits **several distinct antecedents to produce the same consequent**, so the consequent does not identify its cause. `P → Q` constrains what follows from `P`; it says nothing about what else might make `Q` true.

### Conjunctive antecedents weaken Modus Tollens

From `P₁ ∧ P₂ → Q` together with `¬Q`, all that follows is `¬P₁ ∨ ¬P₂` — *at least one* antecedent is false, a **disjunction rather than a definite literal**. A definite literal is obtained only when all but one antecedent is independently known to be true. This is exactly **unit resolution**, and the program implements it as such: the contrapositive is read as the clause `{¬P₁, …, ¬Pₙ}`, and every antecedent known true deletes one disjunct from it.

### Where this sits in the lab

Exhaustive application of Modus Ponens until no new fact appears computes the **deductive closure** of the knowledge base. That fixed-point procedure is the engine of Experiment 5 on forward chaining; the goal-directed use of the same rules read backwards underlies backward chaining.

Both rules are **monotonic** — a derived fact is never withdrawn when new facts arrive. This is the direct opposite of the default reasoning with exceptions of Experiment 1, where asserting `("penguin","cannot","fly")` *removed* a previously derivable conclusion. Here nothing is ever removed, so deriving both `p` and `¬p` cannot be a revisable default: it signals an **inconsistent knowledge base**, which is what Exercise 2 detects and repairs.

---

## Algorithm

### Algorithm 1 — deriving the closure by Modus Ponens

1. Represent every fact as a **literal**, a pair of a symbol and a polarity, so that `p` and `¬p` are distinct entries of the fact set; represent every rule as a list of antecedent literals together with one consequent literal.
2. Scan the rule base once: for each rule whose antecedent literals are all present in the fact set, add the consequent literal and record the rule identifier and the premises used as its **justification**.
3. Repeat step 2 until a complete scan adds no new literal; the fact set has then reached its fixed point and holds the deductive closure.
4. Report each derived literal with the chain of rules that produced it, and report the knowledge base as inconsistent if the fact set contains a symbol together with its negation.

### Algorithm 2 — reasoning from negative evidence by Modus Tollens

1. For every rule, form its **contrapositive** by negating the consequent as the new premise and negating the antecedent as the new conclusion.
2. Given an observed negative literal `¬Q`, select the rules whose consequent is `Q`. If such a rule has a single antecedent `P`, add `¬P`; if it has several, add a definite literal only when all but one antecedent are already known true, otherwise record the **disjunction of negated antecedents as a pending conclusion**.
3. Feed every literal derived in step 2 back into Algorithm 1 — a new negative fact may satisfy the antecedent of a further rule — and **alternate the two algorithms until neither adds anything**.
4. Reject any query matching affirming the consequent or denying the antecedent by reporting that the conclusion is not entailed, and give the interpretation that satisfies the premises while falsifying the conclusion as the **counter-example**.

---

## The knowledge base

A car that will not start. The rules are chosen so that they genuinely chain — the consequents of the early rules are the antecedents of later ones — and so that both a single-antecedent and a multi-antecedent rule sit on the path Modus Tollens has to travel.

```
R1: battery_charged                     -> headlights_work
R2: battery_charged AND starter_ok      -> engine_cranks
R3: engine_cranks AND fuel_ok           -> engine_starts
R4: engine_starts                       -> alternator_charges
R5: alternator_charges                  -> battery_stays_charged
R6: fuel_pump_ok AND fuel_in_tank       -> fuel_ok
R7: ¬engine_starts                      -> tow_required
```

`R7` is the one to notice: its antecedent is a **negative literal**, so it can only ever fire on a fact that Modus Tollens produced. It is what makes step 3 of Algorithm 2 observable.

---

## The code

Everything is in [`main.py`](main.py).

### Step 1 — literals, not symbols

```python
@dataclass(frozen=True, order=True)
class Literal:
    symbol: str
    negated: bool = False

    def __invert__(self):
        return Literal(self.symbol, not self.negated)
```

A fact is a symbol *plus a polarity*, so `p` and `¬p` are two distinct, independently storable entries of the fact set. This is the representational decision the whole experiment rests on. A fact set of bare symbol names could express "believed true" and "not believed", but not "**believed false**" — and without that there is nothing for Modus Tollens to consume and no way for a contradiction to exist at all.

`frozen=True` makes literals hashable, so the fact set is a `dict` keyed by literal: `O(1)` membership, which is the inner-loop operation of both algorithms. `__invert__` gives `~lit`, used constantly — `~r.consequent` *is* the premise Modus Tollens looks for.

### Step 2 — Modus Ponens, one scan

```python
def mp_scan(self):
    added = []
    for r in self.rules:
        if r.consequent in self.facts:
            continue                                # already believed
        if all(a in self.facts for a in r.antecedents):
            if self._add(Step(r.consequent, "MP", r.id, r.antecedents)):
                added.append(r.consequent)
    return added
```

This is Experiment 2's naïve matcher with the act phase inlined: a linear scan of the rule base, `all(...)` short-circuiting on the first missing antecedent. The `r.consequent in self.facts` guard is **refractoriness** — a rule that would only re-assert what is already believed is skipped, which is what makes step 3 terminate.

### Step 3 — the fixed point

```python
def forward_mp(self):
    scans = []
    while True:
        added = self.mp_scan()
        scans.append(added)
        if not added:
            return scans
```

The loop terminates because the fact set only grows and is bounded by `2 × |symbols|`, so a scan that adds nothing can never be followed by one that adds something. **A complete empty scan is the proof of the fixed point** — the last scan in the trace is always the empty one, and it is not wasted work but the termination certificate.

### Justifications (step 4)

Every literal is stored with the record of how it arrived:

```python
@dataclass(frozen=True)
class Step:
    literal: Literal
    kind: str                       # "given" | "MP" | "MT"
    rule_id: Optional[str]
    premises: Tuple[Literal, ...]
```

`explain()` then walks those records recursively to print a full derivation tree. This is the seed of an expert system's **explanation facility** — the *why?* and *how?* questions a user asks of a diagnosis — and Exercise 2 depends on it entirely, since a contradiction is only useful if you can see the two chains that produced it.

### Algorithm 2 — Modus Tollens as unit resolution

The whole of Algorithm 2 steps 1–2 is one loop:

```python
def mt_scan(self):
    added = []
    self.pending = []
    for r in self.rules:
        if ~r.consequent not in self.facts:
            continue                                # no ¬Q available
        disjuncts = tuple(~a for a in r.antecedents)          # step 1
        if any(d in self.facts for d in disjuncts):
            continue                                # clause already satisfied
        known_true = tuple(a for a in r.antecedents if a in self.facts)
        open_ = tuple(d for d in disjuncts if ~d not in self.facts)
        if len(open_) == 1:                                   # definite literal
            self._add(Step(open_[0], "MT", r.id, (~r.consequent,) + known_true))
            added.append(open_[0])
        elif open_:                                           # pending disjunction
            self.pending.append(Clause(r.id, ~r.consequent, open_, known_true))
        else:                                                 # empty clause
            self.empty_clauses.append(Clause(r.id, ~r.consequent, (), known_true))
    return added
```

No contrapositive rule is ever *stored*. `disjuncts` builds it on the fly from the original rule, which is the point the theory makes: the engine keeps each rule once and matches it in either direction.

The four branches are the complete case analysis of unit resolution on the clause `{¬A₁, …, ¬Aₙ}`, and reading them in order is the clearest statement of what Modus Tollens can and cannot do:

| disjuncts left | meaning | action |
|---|---|---|
| some `¬Aᵢ` already believed | the clause is already satisfied | nothing to learn |
| exactly one | Modus Tollens proper — a definite literal | assert it, with justification |
| more than one | only a disjunction is entailed | record as **pending**, assert nothing |
| none — every `Aᵢ` known true | the empty clause | **contradiction** |

`self.pending` is rebuilt on every scan rather than accumulated, so a pending clause is always current: as facts arrive it shrinks, and it disappears the moment it collapses to a definite literal.

### Step 3 — alternating the two algorithms

```python
def saturate(self, verbose=False):
    while True:
        mp_added = [l for scan in self.forward_mp() for l in scan]
        mt_added = self.mt_scan()
        if not mt_added:
            return self
```

Modus Ponens is run to *its own* fixed point, then one Modus Tollens scan; if that scan produced anything, the loop goes round so those new negative literals can fire more rules forwards. Termination has the same argument as before — the fact set only grows.

### The semantic checker — how the fallacies are rejected

The engine could reject a fallacy by simply failing to derive its conclusion, but "I could not derive it" is not "it does not follow". So the program also carries a direct implementation of entailment, enumerating interpretations:

```python
def counterexample(rules, premises, goal, extra=()):
    syms = symbols_of(rules, list(premises) + [goal])
    for k in range(len(syms) + 1):
        for trues in itertools.combinations(syms, k):
            model = {s: (s in trues) for s in syms}
            if not all(rule_holds(r, model) for r in rules):   continue
            if not all(holds(p, model) for p in premises):     continue
            if not all(c(model) for c in extra):               continue
            if not holds(goal, model):
                return model                    # premises true, goal false
    return None                                 # no such row -> entailed
```

`rule_holds` is the truth table of implication verbatim — *false only when the antecedent is true and the consequent false*. Returning `None` means no row of the truth table satisfies the premises while falsifying the goal, which is the definition of entailment.

Models are enumerated **by increasing number of true symbols**, so the counter-example handed back is the smallest one and therefore the one that reads most clearly.

Having semantics as well as syntax buys three things the syntactic engine cannot provide on its own: the fallacies are rejected with a **concrete counter-example** rather than a shrug; every literal Modus Ponens derives is independently **verified sound** against all 2¹¹ interpretations; and the gap between *derivable* and *entailed* becomes measurable, which is what the incompleteness section below reports.

---

## Procedure and Result

`python main.py` prints seven sections; the full console capture is in [`sample_output.txt`](sample_output.txt).

### 1. Algorithm 1 — the closure by Modus Ponens

Input facts: `battery_charged`, `starter_ok`, `fuel_pump_ok`, `fuel_in_tank`.

```
  scan 1: added [headlights_work, engine_cranks, fuel_ok]
  scan 2: added [engine_starts, alternator_charges, battery_stays_charged]
  scan 3: added []

  Fixed point reached after 3 scans (2 productive, the last one empty).
```

Two scans are needed rather than one: `R3` needs `fuel_ok`, which `R6` only produces later in the same scan, so the chain has to propagate on the next pass. The third scan is the fixed-point certificate.

**Derived literals with their justifications (step 4):**

```
  literal                 rule    premises used
  ----------------------------------------------------------------
  headlights_work         R1      battery_charged
  engine_cranks           R2      battery_charged, starter_ok
  fuel_ok                 R6      fuel_pump_ok, fuel_in_tank
  engine_starts           R3      engine_cranks, fuel_ok
  alternator_charges      R4      engine_starts
  battery_stays_charged   R5      alternator_charges
```

**The full chain for the deepest conclusion** — four observations and five rules:

```
    battery_stays_charged   [MP R5]
        alternator_charges   [MP R4]
            engine_starts   [MP R3]
                engine_cranks   [MP R2]
                    battery_charged   [given]
                    starter_ok   [given]
                fuel_ok   [MP R6]
                    fuel_pump_ok   [given]
                    fuel_in_tank   [given]
```

Modus Tollens contributes nothing to this run, and the reason is structural: with only positive observations no negated consequent is ever available, so no contrapositive can fire. **Modus Tollens is not an optional extra that sometimes helps — it is the only rule that can use negative evidence at all.**

**Soundness check.** Every derived literal was tested against all 2¹¹ = 2048 interpretations of the rule base:

```
    headlights_work          entailed = True
    engine_cranks            entailed = True
    fuel_ok                  entailed = True
    engine_starts            entailed = True
    alternator_charges       entailed = True
    battery_stays_charged    entailed = True
```

### 2. Algorithm 2 — reasoning from negative evidence

**Step 1 — contrapositives**, formed from the same stored rules:

```
  R1: battery_charged -> headlights_work
      contrapositive:  ¬headlights_work -> ¬battery_charged
  R3: engine_cranks AND fuel_ok -> engine_starts
      contrapositive:  ¬engine_starts -> ¬engine_cranks OR ¬fuel_ok
  R4: engine_starts -> alternator_charges
      contrapositive:  ¬alternator_charges -> ¬engine_starts
```

Note the shape difference already visible in R3: a conjunctive antecedent contraposes to a **disjunctive** conclusion.

**Steps 2–3 — one negative observation**, `¬alternator_charges` (the ammeter shows no charging current), and nothing else:

```
  round 1: MP added []                MT added [¬engine_starts]
  round 2: MP added [tow_required]    MT added []

  literal                 how   rule    premises used
  --------------------------------------------------------------------
  ¬engine_starts          MT    R4      ¬alternator_charges
  tow_required            MP    R7      ¬engine_starts
```

This is **step 3 of Algorithm 2 in two lines**. Modus Ponens can do nothing in round 1 — there is not a single positive fact to match. Modus Tollens turns the negative observation into `¬engine_starts`, and that literal is exactly what `R7` needs, so the *next* Modus Ponens round fires forwards from a fact that only backward reasoning could have supplied:

```
    tow_required   [MP R7]
        ¬engine_starts   [MT R4]
            ¬alternator_charges   [given]
```

Neither algorithm reaches `tow_required` alone. That is why they must alternate rather than run in sequence.

**The multi-antecedent case.** `¬engine_starts` also matches `R3`, which has two antecedents:

```
    R3: from ¬engine_starts only the disjunction   ¬engine_cranks OR ¬fuel_ok
        no disjunct is definite, so nothing enters the fact set
```

Adding the single fact `engine_cranks` — the mechanic confirms the engine turns over — collapses it:

```
    ¬fuel_ok                MT    R3      ¬engine_starts, engine_cranks
```

and the newly derived `¬fuel_ok` immediately produces a fresh disjunction one level down, from `R6`:

```
    R6: from ¬fuel_ok only the disjunction   ¬fuel_pump_ok OR ¬fuel_in_tank
```

The diagnosis has been narrowed from "something is wrong" to "either the pump has failed or the tank is empty" — a genuinely useful conclusion that is nevertheless not a fact, and the engine is right to refuse to pick one.

### 3. Valid patterns and the two fallacies (Algorithm 2, step 4)

```
  Modus Ponens (valid)
    pattern   : P -> Q, P  |-  Q
    premises  : battery_charged, starter_ok, fuel_pump_ok, fuel_in_tank
    query     : is engine_starts entailed?
    verdict   : ENTAILED (sound) -- no interpretation falsifies it
    engine    : MP R3 from engine_cranks, fuel_ok

  Modus Tollens (valid)
    pattern   : P -> Q, ¬Q  |-  ¬P
    premises  : ¬alternator_charges
    query     : is ¬engine_starts entailed?
    verdict   : ENTAILED (sound) -- no interpretation falsifies it
    engine    : MT R4 from ¬alternator_charges

  Affirming the consequent (FALLACY)
    pattern   : P -> Q, Q  |-  P
    premises  : headlights_work
    query     : is battery_charged entailed?
    verdict   : NOT ENTAILED -- unsound pattern, rejected
    engine    : did not derive it
    counter-example (premises true, conclusion false):
      true : headlights_work, tow_required
      false: alternator_charges, battery_charged, battery_stays_charged,
             engine_cranks, engine_starts, fuel_in_tank, fuel_ok,
             fuel_pump_ok, starter_ok

  Denying the antecedent (FALLACY)
    pattern   : P -> Q, ¬P  |-  ¬Q
    premises  : ¬battery_charged
    query     : is ¬headlights_work entailed?
    verdict   : NOT ENTAILED -- unsound pattern, rejected
    engine    : did not derive it
    counter-example (premises true, conclusion false):
      true : headlights_work, tow_required
      false: alternator_charges, battery_charged, ...
```

Both fallacies are refuted by the **same** interpretation, which is the neatest possible demonstration that they are two readings of one mistake: the world in which the headlights work but the battery is flat. (`tow_required` is true in it only because `engine_starts` is false and `R7` then forces it — the model has to satisfy every rule, not just the one under discussion.)

### 4. Sound but not complete

```
    D1: stale_fuel -> hard_start
    D2: dirty_injectors -> hard_start
    known: stale_fuel OR dirty_injectors   (a disjunction, not a literal)

    engine fact set after saturation        : []
    engine derives hard_start               : False
    truth table says hard_start is entailed : True
```

`hard_start` follows from those premises by cases — it holds on either disjunct — but the engine derives nothing at all, because the fact set can hold only literals and there is nowhere to put the premise in the first place.

---

## Observations

### Why the contrapositive is needed to apply Modus Tollens

Because the engine has exactly one matching operation, and it works forwards. `mp_scan` asks *are all the antecedents present?*; nothing in it can consume a negated **consequent**. `¬Q` matches no antecedent of `P → Q`, so the rule as stored is simply inert under that premise.

The contrapositive `¬Q → ¬P` is the same implication rewritten so that the fact in hand *is* its antecedent. It is not new knowledge — the two are logically equivalent, true in exactly the same rows of the truth table — but it is knowledge in a form the matcher can reach. This is why the engine stores each rule once and forms the contrapositive on the fly (`disjuncts = tuple(~a for a in r.antecedents)`): materialising both directions would double the rule base and, worse, allow the two copies to drift apart when the knowledge base is edited.

Reading it operationally: **Modus Ponens propagates belief down the implication, Modus Tollens propagates disbelief up it.** The car diagnosis needs both directions, because the useful evidence is often the absence of a symptom.

### What can and cannot be concluded when the rule has more than one antecedent

From `A₁ ∧ … ∧ Aₙ → Q` and `¬Q`, what follows is precisely the clause `¬A₁ ∨ … ∨ ¬Aₙ`: **at least one antecedent is false**. It is *not* the case that any particular one is false, and asserting one would be exactly the error the engine exists to prevent.

The run shows both halves. From `¬engine_starts`, `R3` yields only `¬engine_cranks ∨ ¬fuel_ok`, and the engine records it as pending and adds nothing to the fact set. Once `engine_cranks` is independently known true, one disjunct is eliminated and the remaining `¬fuel_ok` becomes definite — derived, justified, and usable by further rules.

Three things are worth drawing out:

- **The general rule is `n − 1`.** A definite literal appears only when *n − 1* antecedents are independently known true. A rule with three antecedents therefore needs two confirmations, which is what Exercise 1 measures.
- **The information is real even when it is not a fact.** "Either the pump failed or the tank is empty" halves the search space and tells a technician what to test next. Discarding it because it is not a literal would throw away most of what negative evidence is worth.
- **This is why definite-clause engines stop here.** Storing the disjunction as a *fact* would make the fact set a set of clauses, the matcher a resolution procedure, and the fixed point potentially exponential. Recording it as a pending conclusion that unit resolution may later collapse is the compromise: full power on the cases that resolve to literals, an honest refusal on the rest.

### The counter-example that shows affirming the consequent to be unsound

`R1: battery_charged → headlights_work`. Observe `headlights_work` and conclude `battery_charged` — the engine rejects it and returns the interpretation

```
      true : headlights_work, tow_required
      false: battery_charged, and every other symbol
```

Check it against the definition of soundness. Every rule is satisfied: `R1`'s antecedent `battery_charged` is false, so the implication holds vacuously — and it holds *whatever* `headlights_work` is doing. The premise `headlights_work` is true. The conclusion `battery_charged` is false. One such row is enough: there is an interpretation in which the premises hold and the conclusion fails, so the conclusion is not entailed and the pattern is unsound.

The intuition the row encodes is that **`P → Q` licenses no traffic from `Q` back to `P`**. It forbids exactly one combination — antecedent true, consequent false — and says nothing whatever about the row where the antecedent is false and the consequent true. Concretely: headlights can be lit by a jump pack, a second battery, or a bench supply. The implication never claimed a charged battery was the *only* route to working headlights, so observing the effect does not identify the cause.

Denying the antecedent is the same error in a mirror: from `¬battery_charged` one cannot conclude `¬headlights_work`, and the run refutes it with the identical interpretation. Both are attempts to read a one-way implication as a biconditional. If `battery_charged ↔ headlights_work` really did hold, both inferences would be valid — which is a useful diagnostic for when the fallacy is tempting: it is tempting exactly when the rule feels like a definition.

### Why the fixed point of Algorithm 1 is sound but not complete for full propositional logic

**Sound** because it applies only Modus Ponens, one truth-preserving step, to premises that are either given or already derived. By induction over the derivation, every literal in the fixed point is true in every interpretation that satisfies the input facts and the rules. The program checks this empirically rather than asserting it: each of the six derived literals is tested against all 2048 interpretations, and all six come back `entailed = True`.

**Not complete** because the engine can express and derive strictly less than propositional logic contains. The fact set holds *literals only* — there is no room for a disjunction, an implication, or any other compound sentence — so anything whose proof must pass through a compound intermediate is unreachable. Concretely, the engine is missing:

- **Reasoning by cases.** The run demonstrates it: from `stale_fuel ∨ dirty_injectors` with `stale_fuel → hard_start` and `dirty_injectors → hard_start`, `hard_start` is entailed (the truth table confirms it) and the engine derives nothing, because the premise cannot even be stated as an input.
- **Proof by contradiction.** Assuming `¬p`, deriving a contradiction and concluding `p` requires hypothetical reasoning — a temporary assumption withdrawn later — and Algorithm 1 has no notion of an assumption, only of facts.
- **Introducing disjunctions or implications.** `p ⊨ p ∨ q` is valid and out of reach, as is the deduction theorem.
- **Anything a rule base cannot phrase.** `RULES` is a set of definite clauses; a general propositional theory can contain clauses with several positive literals, and those have no representation here.

The trade is deliberate and is the standard one in expert systems. Restricting to definite (Horn) clauses makes the closure computable in time linear in the size of the rule base and, crucially, makes every conclusion carry a **single readable justification chain** — the derivation printed above. A complete propositional prover (resolution, DPLL) buys the missing inferences at the price of a possibly exponential search and proofs that are much harder to present to a user as an explanation. For diagnosis, where rules genuinely are of the form *these symptoms imply this fault* and the user needs to be told *why*, completeness is the cheaper thing to give up.

Note also that soundness is what makes the contradiction in Exercise 2 *meaningful*. If the engine could derive unsound conclusions, `p` and `¬p` appearing together would prove nothing about the knowledge base. Because every step is sound, a contradiction in the fixed point is a proof that the input facts and rules cannot all be true.

---

## Exercise 1 — Modus Tollens on a rule with three antecedents

A three-antecedent rule is added:

```
R8: fuel_pump_ok AND fuel_in_tank AND injectors_clean -> fuel_delivered
contrapositive:  ¬fuel_delivered -> ¬fuel_pump_ok OR ¬fuel_in_tank OR ¬injectors_clean
```

`¬fuel_delivered` is observed, and facts are then added one at a time:

```
  input                     pending disjunction                                 definite literal
  ----------------------------------------------------------------------------------------------
  observe ¬fuel_delivered   ¬fuel_pump_ok OR ¬fuel_in_tank OR ¬injectors_clean  -
  + fuel_pump_ok            ¬fuel_in_tank OR ¬injectors_clean                   -
  + fuel_in_tank            (none)                                              ¬injectors_clean
```

### The conclusion the engine can record at the observation

**Only the three-way disjunction** `¬fuel_pump_ok ∨ ¬fuel_in_tank ∨ ¬injectors_clean` — at least one of the three antecedents is false. No literal enters the fact set, and nothing downstream of those three symbols can fire.

### The fact that turns the pending disjunction into a definite literal

**`fuel_in_tank`** — the *second* fact added, not the first and not the last.

Adding `fuel_pump_ok` is not useless: it deletes one disjunct and shortens the clause from three literals to two. But two disjuncts still leave a choice, so the conclusion stays a disjunction. `fuel_in_tank` deletes the second, leaving exactly one, and unit resolution turns it into a definite literal with a full justification:

```
    ¬injectors_clean   [MT R8]
        ¬fuel_delivered   [given]
        fuel_pump_ok   [given]
        fuel_in_tank   [given]
```

The premises recorded are precisely the ones used: the negated consequent plus the *n − 1* antecedents confirmed true.

**What the exercise establishes.** The number of confirmations Modus Tollens needs scales with the arity of the rule: a rule with *n* antecedents needs *n − 1* of them independently established before it yields anything definite from a negated consequent. Conjunctive antecedents are cheap for Modus Ponens — one scan, all present or not — and expensive for Modus Tollens, because contraposition turns a conjunction into a disjunction, and a disjunction has to be whittled down one literal at a time.

There is a modelling consequence. Two rules with two antecedents each will generally yield definite conclusions from negative evidence sooner than one rule with four, even when they encode the same knowledge, because each contraposes to a shorter clause. Rule granularity is not only a matter of readability; it determines how much negative evidence the knowledge base can actually use.

---

## Exercise 2 — Detecting a contradiction and repairing it

One rule is added whose **consequent is a negative literal**, together with one fact to fire it, so that both sides of the clash are genuinely derived rather than one being asserted:

```
R9: fuse_blown -> ¬alternator_charges
input facts: battery_charged, starter_ok, fuel_pump_ok, fuel_in_tank, fuse_blown
```

The story is an ordinary one: the car passes every check that should make it charge, *and* an inspection finds the alternator fuse blown.

```
  round 1: MP added [headlights_work, engine_cranks, fuel_ok, ¬alternator_charges,
                     engine_starts, alternator_charges, battery_stays_charged]

  INCONSISTENT knowledge base -- contradiction on: alternator_charges
```

### The two justification chains that meet

```
    alternator_charges   [MP R4]
        engine_starts   [MP R3]
            engine_cranks   [MP R2]
                battery_charged   [given]
                starter_ok   [given]
            fuel_ok   [MP R6]
                fuel_pump_ok   [given]
                fuel_in_tank   [given]

    ¬alternator_charges   [MP R9]
        fuse_blown   [given]
```

A five-rule chain from four observations meets a one-rule chain from a fifth, on the same symbol with opposite polarity. Printing both is the point of keeping justifications: `alternator_charges` and `¬alternator_charges` sitting in the fact set together says only *something is wrong*, whereas the two trees say **which observations and which rules are in conflict** — which is what a knowledge engineer needs in order to fix it.

Modus Tollens reports the same clash independently, as the **empty clause**, from both directions:

```
    R4: ¬alternator_charges holds, but every antecedent (engine_starts) is
        known true, so no disjunct of the contrapositive survives.
    R9: alternator_charges holds, but every antecedent (fuse_blown) is
        known true, so no disjunct of the contrapositive survives.
```

This is the fourth branch of `mt_scan`. Deriving the empty clause is the resolution-theoretic definition of inconsistency, and it is the same finding the symbol-level test makes — one detected syntactically in the fact set, one detected by exhausting a clause.

> **A note on provenance.** Because `forward_mp` runs to its own fixed point before the first Modus Tollens scan, Modus Ponens gets to both sides of the clash first here, and Modus Tollens finds every contrapositive clause already falsified. That is not an accident of this rule base: in an *inconsistent* knowledge base the closure is the same set whichever rule reaches a literal first, so the phase that runs first supplies the justification and the other reports empty clauses. In a *consistent* one — the main run above — the two phases derive genuinely different literals and must alternate.

### The smallest set of input facts whose removal restores consistency

`minimal_repairs` enumerates subsets of the input facts by increasing size and re-saturates, returning the first size at which consistency is recovered:

```
  smallest sets of input facts whose removal restores consistency
  (size 1, 5 of them):
    remove {battery_charged}
    remove {starter_ok}
    remove {fuel_pump_ok}
    remove {fuel_in_tank}
    remove {fuse_blown}
```

Every repair is a **single** fact, and each one cuts one of the two chains. Removing `fuse_blown` kills the negative branch at its root. Removing any one of the four positive observations breaks the Modus Ponens chain somewhere before `R4` — `battery_charged` or `starter_ok` stops `R2`, `fuel_pump_ok` or `fuel_in_tank` stops `R6`, and either way `R3` never fires and `alternator_charges` is never derived.

That the repairs are exactly the leaves of the two justification trees is not a coincidence: a **minimal repair is a hitting set of the support sets of the conflict**, and with two chains meeting at one symbol, any single leaf of either chain suffices. This is the core of a truth maintenance system's conflict-resolution step, and of model-based diagnosis in general.

Two things the result does *not* say, and both matter:

- **It is not a diagnosis.** Five equally minimal repairs means the logic alone cannot choose. Deciding that `fuse_blown` is far likelier to be a bad observation than `fuel_in_tank`, or that a rule rather than a fact is at fault, requires knowledge the engine does not have: certainty factors, sensor reliability, or a preference ordering over the inputs. Minimality is a filter, not a decision procedure.
- **Only facts were considered.** The rules were held fixed and treated as beyond suspicion. A real inconsistency is just as often a wrong or over-general rule — `R4` claiming that a running engine *always* charges is exactly the kind of idealisation that breaks when a fuse blows — and a knowledge base under active development should let rules be candidates for repair too.

### Why this cannot happen in Experiment 1

In Experiment 1, `bird can fly` and `penguin cannot fly` coexist happily: the search order resolves them, the more specific fact wins, and neither is an error. Here `alternator_charges` and `¬alternator_charges` cannot be resolved by any ordering, because Modus Ponens and Modus Tollens are **monotonic** and give no fact precedence over another — nothing is a default, so nothing can be overridden. The contradiction is a genuine defect in the knowledge base, and the only repairs available are the ones above: retract an input, or fix a rule.

---

## Viva questions

**1. State Modus Ponens and Modus Tollens and show both are sound.**
Modus Ponens: from `P → Q` and `P`, infer `Q`. Modus Tollens: from `P → Q` and `¬Q`, infer `¬P`. `P → Q` is false only in the row where `P` is true and `Q` is false. Given the implication, that row is excluded; adding `P` leaves only the row `P = T, Q = T`, so `Q` holds — and adding `¬Q` leaves only `P = F, Q = F`, so `¬P` holds. In both cases every interpretation satisfying the premises satisfies the conclusion, which is the definition of soundness.

**2. Why does the engine need the contrapositive?**
Its matcher works in one direction only — it tests whether a rule's antecedents are present. A negated consequent matches no antecedent of the rule as stored, so `P → Q` is inert given `¬Q`. The contrapositive `¬Q → ¬P` is the same implication written so the available fact is its antecedent. The engine builds it on the fly from the stored rule rather than materialising it, so the rule base stays single-copy and cannot drift.

**3. What follows by Modus Tollens from a rule with three antecedents?**
Only the disjunction `¬A₁ ∨ ¬A₂ ∨ ¬A₃` — at least one is false. A definite literal is obtained only when *n − 1* of the antecedents are independently known true; each such fact deletes one disjunct by unit resolution. In the run, `¬fuel_delivered` gives a three-way disjunction, `fuel_pump_ok` shortens it to two, and `fuel_in_tank` collapses it to the definite `¬injectors_clean`.

**4. Give a counter-example to affirming the consequent.**
Take `battery_charged → headlights_work` with `headlights_work` observed, and the interpretation in which `headlights_work` is true and `battery_charged` is false. The implication is satisfied — vacuously, since its antecedent is false — the premise is true, and the conclusion is false. Premises true with conclusion false is precisely non-entailment, so the pattern is unsound. Intuitively, several causes can produce the same effect: an implication forbids only the row *antecedent true, consequent false*, and observing the consequent does not identify its cause.

**5. Is the fixed point sound? Is it complete?**
Sound: every literal in it is entailed, by induction over a derivation that uses only Modus Ponens — verified in the run against all 2048 interpretations. Not complete for full propositional logic: the fact set holds only literals, so reasoning by cases, proof by contradiction, and disjunction introduction are all out of reach. The run exhibits the gap — `hard_start` follows from `stale_fuel ∨ dirty_injectors` with the two rules, and the engine derives nothing. The restriction to definite clauses is what buys a linear-time closure and a single readable justification per conclusion.

**6. What does deriving both `p` and `¬p` mean here, and how does it differ from Experiment 1?**
An inconsistent knowledge base — a proof that the input facts and the rules cannot all be true. It is meaningful *because* the rules are sound. In Experiment 1 a comparable clash (`bird can fly` versus `penguin cannot fly`) is not an error at all: those are defaults, and the specificity-ordered search resolves them. Modus Ponens and Modus Tollens are monotonic and treat no fact as a default, so there is no ordering that could resolve the clash; the only repairs are to retract an input fact or fix a rule.

---

## File map

| Item in `main.py` | Purpose |
|---|---|
| `Literal`, `P`, `N` | A symbol plus a polarity, so `p` and `¬p` are distinct (Alg. 1 step 1) |
| `Rule`, `rule`, `Rule.contrapositive` | Antecedent literals + one consequent; contraposition (Alg. 2 step 1) |
| `RULES`, `FACTS_MP`, `FACTS_MT` | The car-diagnosis rule base and the two observation sets |
| `Step`, `Clause` | Justification records, and pending / empty clauses |
| `Engine.mp_scan`, `Engine.forward_mp` | Modus Ponens, one scan and the fixed point (Alg. 1 steps 2–3) |
| `Engine.mt_scan` | Modus Tollens as unit resolution (Alg. 2 steps 1–2) |
| `Engine.saturate` | Alternates the two algorithms (Alg. 2 step 3) |
| `Engine.contradictions`, `Engine.explain` | Inconsistency test and justification chains (Alg. 1 step 4) |
| `counterexample`, `entails`, `rule_holds` | Truth-table semantics — rejects the fallacies (Alg. 2 step 4) |
| `report_query` | Prints a query with its verdict and counter-example |
| `R8`, `exercise1` | Exercise 1 — three antecedents, collapsing the disjunction |
| `R9`, `minimal_repairs`, `exercise2` | Exercise 2 — contradiction detection and minimal repair |
| `RULES_CASES`, `cases_demo` | Sound but not complete — reasoning by cases |
| `demo` | Prints all of the above |

## Files

| File | Contents |
|---|---|
| [`main.py`](main.py) | Full implementation — both algorithms, the semantic checker, both exercises, and the demo driver |
| [`README.md`](README.md) | This write-up |
| `sample_output.txt` | Captured console output of a full run, ready to paste into the report |
| `Assignment - 4.pdf` | The original handout |

## Running

```
python main.py            # everything, ~1 s, standard library only
```

From a REPL:

```python
from main import *

eng = run(RULES, FACTS_MP)                 # Algorithm 1 to its fixed point
eng.derived()                              # the six literals it produced
print("\n".join(eng.explain(P("battery_stays_charged"))))

eng = run(RULES, [N("alternator_charges")])   # Algorithm 2 from negative evidence
eng.derived()                                 # [¬engine_starts, tow_required]
eng.pending                                   # [¬engine_cranks OR ¬fuel_ok]

counterexample(RULES, [P("headlights_work")], P("battery_charged"))   # the fallacy
minimal_repairs(RULES_EX2, FACTS_EX2)                                 # Exercise 2
```
