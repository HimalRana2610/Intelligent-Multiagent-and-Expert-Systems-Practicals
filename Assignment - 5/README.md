# Experiment 5 — Uncertainty Reasoning: Inference in Bayesian Networks

**Intelligent Multi-Agent and Expert Systems (AI401)** · B.Tech IV (AI), Semester VII
Run with `python main.py` (standard library only, well under a second).

---

## Aim

To specify a Bayesian network as a directed acyclic graph with a conditional probability table at each node, to evaluate entries of the full joint distribution from that factorisation, and to compute the posterior distribution of a query variable given evidence by enumeration.

## Theory

Experiment 4 derived conclusions from premises that were *certainly true*. Diagnostic knowledge rarely is: a lit warning lamp makes a flat battery likely, not certain, and the rule `warning_light → battery_flat` is simply false. Attaching a number to the arrow is the change this experiment makes, and it changes what an inference step *is* — no longer "write down the consequent", but "redistribute belief over the values of a variable".

### The factorisation

A **Bayesian network** is a directed acyclic graph whose nodes are random variables, whose edges denote direct dependence, and where each node carries the distribution of its variable conditioned on its parents. Ordering the variables topologically and applying the chain rule,

```
P(x₁, …, xₙ) = ∏ᵢ P(xᵢ | parents(xᵢ))
```

The chain rule alone would give `P(xᵢ | x₁ … xᵢ₋₁)` — conditioned on *every* earlier variable. The network asserts that the parents are sufficient, and every conditioning variable dropped is an edge left out of the graph. So the graph is not a picture attached to the numbers; the graph **is** the set of independence assumptions, and the numbers are what remains once those assumptions are made.

Two consequences follow immediately, and both are measured in the run:

- **An entry of the joint costs one multiplication per node.** No table of 2ⁿ entries is ever built.
- **The network is specified by its table parameters**, `∑ᵢ 2^|parents(xᵢ)|` for binary variables, instead of the `2ⁿ − 1` a full joint would need.

### Inference

Inference means computing the posterior of a query variable `X` given evidence `e`. Summing the joint over every assignment of the remaining **hidden** variables `Y` and normalising:

```
P(X | e) = α · ∑_y P(X, e, y)          α = 1 / P(e)
```

The normalising constant is not a bookkeeping device. `∑ₓ ∑_y P(x, e, y) = P(e)` exactly, so `1/α` **is the probability of the evidence** — the quantity a joint query would have returned, and the standard measure of how surprising an observation is. The program prints it for every query, and the cross-check confirms it comes out the same whichever variable is queried.

### What the graph asserts

A variable is independent of its non-descendants given its parents. Read off the paths, this gives three patterns, and this experiment's network contains all three:

| pattern | shape | dependent when… |
|---|---|---|
| **chain** | `B → C → R` | `C` is *hidden*; observing `C` blocks the path |
| **common cause** | `W ← C → R` | `C` is *hidden*; observing `C` blocks the path |
| **common effect** | `B → C ← S` | `C` is *observed* — the opposite rule |

The third is **explaining away**, and it is the one with no counterpart in Experiment 4. Two causes that are independent a priori become dependent once their common effect is observed: confirming one cause *lowers* the posterior of the other, because the effect no longer needs a second explanation.

### Why no rule of Experiment 4 can express this

In Experiment 4 the fact set only grows. Modus Ponens and Modus Tollens are **monotonic**, so a derived conclusion is never weakened by new evidence — adding `starter_faulty` to a knowledge base that has already derived `battery_flat` cannot retract it. Explaining away is precisely a conclusion being weakened by new evidence, so it is not merely absent from that engine, it is incompatible with its central property. The run shows `P(b | w, r)` falling from `0.636567` to `0.054722` on one extra observation; no sound monotonic rule can produce that number, in either direction.

The other side of the trade is that nothing here is *derived* in the Experiment 4 sense. There is no justification chain to print, because every variable's posterior depends on every table in the network. Explanation in a probabilistic system has to be reconstructed (most-probable explanation, sensitivity analysis), not read off a derivation — which is the price paid for the ability to change one's mind.

---

## Algorithm

### Algorithm 1 — Specifying the network and evaluating a joint entry

1. List the variables in a **topological order**, recording for each its parents and the conditional probability table giving `P(variable = true)` for every combination of parent values.
2. **Validate**: one row per combination of parent values, every row in `[0, 1]`, and count the network's parameters against the `2ⁿ − 1` of the full joint.
3. Evaluate a **full joint entry** for a complete assignment as the product over nodes of the table entry selected by that variable and its parents, complementing where the variable is false.
4. Verify the factorisation by summing over all `2ⁿ` assignments and confirming the total is **one**, and obtain the **marginal** of any variable by summing over the rest.

### Algorithm 2 — Posterior inference by enumeration with evidence

1. Fix the query variable `X` and the evidence `e`; the remaining variables form the **hidden set**.
2. For each value of `X`, sum the joint over every assignment of the hidden set with `X` and `e` held fixed. The two sums are the **unnormalised vector**, whose components are joint probabilities.
3. **Normalise** to obtain `P(X | e)`, recording the normalising constant separately, since it is `P(e)`.
4. Repeat with an extra evidence variable and compare, noting whether the belief **rises, falls or is unchanged**, and relate the direction to the paths in the graph.

---

## The network

A car that will not start — the same domain as Experiment 4, so the two engines can be compared on the same story.

```
      battery_flat   B --+
                         +--> engine_wont_crank  C --+--> warning_light  W
      starter_faulty S --+                           |
                                                     +--> tow_called     R
```

| variable | sym | parents | table |
|---|---|---|---|
| `battery_flat` | B | — | `P(b) = 0.05` |
| `starter_faulty` | S | — | `P(s) = 0.02` |
| `engine_wont_crank` | C | B, S | `P(c \| b, s) = 0.99` · `P(c \| b, ¬s) = 0.95` · `P(c \| ¬b, s) = 0.90` · `P(c \| ¬b, ¬s) = 0.01` |
| `warning_light` | W | C | `P(w \| c) = 0.90` · `P(w \| ¬c) = 0.05` |
| `tow_called` | R | C | `P(r \| c) = 0.70` · `P(r \| ¬c) = 0.01` |

The shape is chosen so that a single network exhibits everything the experiment asks for:

- **Converging arrows at C** — two independent causes of one effect, which is what Exercise 1 needs.
- **Diverging arrows out of C** — two indicators whose only parent is `C`, giving both a pair that is dependent until `C` is observed *and* the query whose posterior is unchanged by further evidence.
- **Every table strictly interior to `[0, 1]`.** `P(c | ¬b, ¬s) = 0.01` matters: it leaves room for a cause outside the model, so `¬b ∧ ¬s ∧ c` is improbable rather than impossible. A zero there would make some evidence sets impossible and the posterior undefined.
- **Noisy indicators.** `P(w | ¬c) = 0.05` means a lit lamp is evidence, not proof. This is what makes the numbers move by interesting amounts rather than collapsing to 0 and 1.

The running query throughout: **the lamp is lit and a tow has been called — how likely is a flat battery?**

---

## The code

Everything is in [`main.py`](main.py).

### Step 1 — a node stores only `P(true)`

```python
@dataclass
class Node:
    name: str
    symbol: str
    parents: Tuple[str, ...]
    cpt: Dict[Tuple[bool, ...], float]     # parent values -> P(name = true)

    def p(self, value: bool, a: Assignment) -> float:
        p_true = self.cpt[self.key(a)]
        return p_true if value else 1.0 - p_true
```

The CPT is keyed by the **tuple of parent values**, so `key()` is a lookup rather than a search, and with `T, F = True, False` the table reads as a table in the source:

```python
CRANK = child("engine_wont_crank", "C", ["battery_flat", "starter_faulty"], {
    (T, T): 0.99,      # both faults present
    (T, F): 0.95,      # a flat battery alone
    (F, T): 0.90,      # a faulty starter alone
    (F, F): 0.01,      # neither -- some cause outside the model
})
```

Storing only the probability of *true* and complementing in `p()` is the first economy of the whole scheme: a binary node with `k` parents costs `2^k` numbers, not `2^(k+1)`. Every distribution in the network is normalised by construction, so no table can be internally inconsistent.

### Step 2 — validation

`validate()` returns `(check, ok, detail)` triples so the demo can print them as a table. Four checks:

| check | what it rules out |
|---|---|
| topological order, hence acyclic | a cycle, and any node whose parent is undefined at evaluation time |
| one row per combination of parent values | a missing or spurious row — `set(cpt) == set(parent_rows())` is exact, not a count |
| every entry in the unit interval | a number that is not a probability |
| parameters against the full joint | reports `10` versus `31` |

The order check does double duty. Verifying that every parent appears *before* its child in the list confirms the list is a topological order, and a graph with a topological order is acyclic — so acyclicity is established by the same pass, with no separate cycle search. It is also exactly the precondition the evaluation needs: `Node.p` reads `a[p]` for each parent, and in a topological order those values are always available.

### Step 3 — a joint entry is one product

```python
def joint(self, a: Assignment) -> float:
    return math.prod(n.p(a[n.name], a) for n in self.nodes)
```

That single line is the chain-rule factorisation. `joint_factors` returns the same product term by term so the demo can print the derivation:

```
    P = P(b)          = 0.0500
        P(¬s)         = 0.9800
        P(c | b, ¬s)  = 0.9500
        P(w | c)      = 0.9000
        P(r | c)      = 0.7000
        -----------------------
        product       = 0.0293265000
```

Five multiplications for an entry of a 32-entry table, and each factor is selected by the values the assignment gives to that node and its parents alone.

### Step 4 — the factorisation is a distribution

```python
def total_probability(self) -> float:
    return sum(self.joint(a) for a in self.assignments())
```

This is not a formality. An arbitrary set of tables attached to an arbitrary graph would produce a product that is not a distribution; the sum coming to `1.000000000000` (to within `2.22e-16`) is the evidence that the DAG structure and the per-node normalisation are together sufficient. The marginals are the same computation grouped by one variable, and the check that lands is the **roots returning their priors exactly** — `P(b) = 0.050000`, `P(s) = 0.020000`. A root's marginal is only its prior if the summing-out is right, so this catches an indexing error that the total alone would not.

### Algorithm 2 — enumeration

```python
def enumeration_ask(net, X, evidence) -> Query:
    hidden = [v for v in net.variables if v != X and v not in evidence]
    unnorm, terms = {}, 0
    for x in (T, F):                                            # step 2
        total = 0.0
        for a in net.assignments(hidden, fixed={**evidence, X: x}):
            total += net.joint(a)
            terms += 1
        unnorm[x] = total
    k = unnorm[T] + unnorm[F]                                   # step 3
    post = {v: unnorm[v] / k for v in (T, F)} if k else {T: 0.0, F: 0.0}
    return Query(X, dict(evidence), hidden, unnorm, post, k, terms, terms * len(net.nodes))
```

Deliberately the literal reading of the algorithm — the hidden set, the two sums, the division — with no cleverness in between, so the printed trace and the code are the same object. Everything needed to *present* the query is kept in the `Query` record: the hidden set, the unnormalised vector, the posterior, `P(e)`, and the operation counts.

The demo prints the sum before the division, which is where the method is visible:

```
  assignment of the hidden set            joint with b      joint with ¬b
  ----------------------------------------------------------------------------
  s, c                                    0.0006237000      0.0107730000
  s, ¬c                                   0.0000000050      0.0000009500
  ¬s, c                                   0.0293265000      0.0058653000
  ¬s, ¬c                                  0.0000012250      0.0004608450
  ----------------------------------------------------------------------------
  sum                                     0.0299514300      0.0171000950
```

Eight joint entries, four per value of the query variable. Note how uneven they are: `¬s, c` carries almost all of the `b` column and `s, c` almost all of the `¬b` column. The sum is dominated by a few assignments, which is exactly why sampling methods work on networks too large to enumerate.

### The recursive form, and why it is only a constant factor better

`enumeration_ask_recursive` is the depth-first version — the same sum, but a factor is evaluated once per node of the enumeration *tree* rather than once per complete assignment, so common prefixes are shared:

```
    flat sum over assignments : 0.636566615003   40 multiplications
    depth-first over the DAG  : 0.636566615003   30 multiplications
    agree to within 1.11e-16
```

40 against 30. It is worth having both, because the comparison makes a point that a single implementation cannot: the saving is **real but constant**. The tree still has one leaf per hidden assignment, so both are `Θ(2^|hidden|)`, and no amount of prefix sharing changes that. Getting past the exponential requires variable elimination — summing out a variable *once* into a factor reused across the sum — which is a different algorithm, not an optimisation of this one.

### Independence, checked numerically rather than asserted

```python
def independence_rows(net, x, y, given=()):
    rows = []
    for z in net.assignments(list(given)):
        if enumeration_ask(net, x, z).normaliser <= TOL:
            continue                                   # impossible conditioning
        rows.append((z, posterior(net, x, {**z, y: T}), posterior(net, x, {**z, y: F})))
    return rows
```

`X ⫫ Y | Z` holds when observing `Y` never moves the belief in `X`, **for every** assignment of `Z` — so the check quantifies over `Z` rather than testing one convenient case. Conditioning events of probability zero are skipped, since a posterior given an impossible observation is undefined rather than equal. Every independence claim in this write-up is produced by this function from the actual posteriors; none is asserted from the graph alone. The graph predicts them, and the arithmetic confirms the prediction.

---

## Results

### Algorithm 1 — the network against the full joint

```
  variable            parents   rows = 2^parents
  ----------------------------------------------------
  battery_flat        0         1
  starter_faulty      0         1
  engine_wont_crank   2         4
  warning_light       1         2
  tow_called          1         2
  ----------------------------------------------------
  total                         10

  A full joint over 5 binary variables needs 2^5 - 1 = 31 numbers.
  The network needs 10, a saving of 21.
```

```
  sum of the joint over all 2^5 = 32 assignments = 1.000000000000
  equals 1 to within 2.22e-16

  variable              P(true)       P(false)
  --------------------------------------------------
  battery_flat          0.050000      0.950000
  starter_faulty        0.020000      0.980000
  engine_wont_crank     0.073950      0.926050
  warning_light         0.112857      0.887142
  tow_called            0.061025      0.938974
```

### Algorithm 2 — the posterior, and the three directions

```
    query     : P(b | w, r)
    hidden    : starter_faulty, engine_wont_crank   -> 4 assignments per value
    step 2    : unnormalised [0.02995143, 0.01710009]   = [P(b, e), P(¬b, e)]
    step 3    : P(e) = 0.04705152   posterior [0.636567, 0.363433]
    answer    : P(b | w, r) = 0.636567
```

Step 4, one query for each direction the belief can move:

| | query | before | after | direction |
|---|---|---|---|---|
| **(a)** supporting evidence | `P(b \| w)` → `P(b \| w, r)` | 0.380205 | **0.636567** | RISES (+0.256) |
| **(b)** competing evidence | `P(b \| w, r)` → `P(b \| s, w, r)` | 0.636567 | **0.054722** | FALLS (−0.582) |
| **(c)** screened-off evidence | `P(r \| c)` → `P(r \| b, c)` | 0.700000 | **0.700000** | UNCHANGED |

Three different behaviours from one network and one algorithm, with nothing in the code special-casing them. The direction is decided entirely by where the new evidence sits relative to the query in the graph.

### The independences, verified

```
  1.  B _||_ S              P(b | s)     = 0.050000   P(b | ¬s)     = 0.050000   diff 0.000000
  2.  R _||_ B | C          P(r | b, c)  = 0.700000   P(r | ¬b, c)  = 0.700000   diff 0.000000
                            P(r | b, ¬c) = 0.010000   P(r | ¬b, ¬c) = 0.010000   diff 0.000000
  3.  R _||_ W | C          P(r | w, c)  = 0.700000   P(r | ¬w, c)  = 0.700000   diff 0.000000
                            P(r | w, ¬c) = 0.010000   P(r | ¬w, ¬c) = 0.010000   diff 0.000000
  4.  R -/- W               P(r | w)     = 0.416911   P(r | ¬w)     = 0.015752   diff 0.401159
```

Rows 3 and 4 are **the same pair of variables**, and are the cleanest statement of what conditional independence means: `W` and `R` are strongly dependent (0.417 against 0.016 — a factor of 26) while their shared cause is hidden, because each is evidence about `C` and therefore about the other, and *exactly* independent the moment `C` is observed, because everything they had to say concerned `C`.

### Why enumeration is exponential

```
  evidence                  hidden  terms     multiplications     P(b | e)
  ------------------------------------------------------------------------------
  s, c, w, r                0       2         10                  0.054726
  c, w, r                   1       4         20                  0.642867
  w, r                      2       8         40                  0.636567
  w                         3       16        80                  0.380205
  (none)                    4       32        160                 0.050000
```

Every variable removed from the evidence doubles the work: `terms = 2 × 2^|hidden|`, each term a product over all `n` nodes, so the cost is `n × 2^(|hidden|+1)`.

Two things worth reading off this table. **The last row is the prior** — with no evidence, enumeration sums the entire joint and returns `P(b) = 0.050000`, agreeing with the marginal from Algorithm 1 step 4, so the two algorithms meet where they should. And **more evidence is cheaper**, which inverts the usual intuition: an observed variable is a constant in the sum rather than a summation index, so evidence is what makes a query tractable. The expensive query is the uninformed one.

---

## Observations

**How many parameters the network saves, and where the saving comes from.** Ten against thirty-one — the network needs 32% of the numbers a full joint over the same five variables would. The saving is bought entirely by the **edges left out**. `W` and `R` have one parent each and cost 2 rows apiece; had every variable been allowed to depend on all its predecessors, as the bare chain rule permits, they would have cost 8 and 16 rows and the total would have been exactly 31 again. Each missing edge is an assertion of conditional independence, and the parameter count is the exact price list: `2^k` for a node with `k` parents. The saving grows sharply with size — a network of 30 binary variables with at most 3 parents each needs at most 240 numbers against the joint's 10⁹ — which is what makes a network *specifiable* at all. Nobody can elicit 10⁹ consistent numbers from a domain expert; `P(engine_wont_crank | battery_flat, starter_faulty)` is a question a mechanic can actually answer.

**What the normalising constant measures.** It is `P(e)`, the probability of the evidence. Summing the unnormalised vector over the values of the query variable sums the joint over *all* variables with only `e` fixed, and that is the definition of `P(e)`. So it is a quantity of the model, not a scaling convenience: the cross-check confirms `P(w, r) = 0.04705152` whichever variable is queried. Two uses follow. It measures **how surprising the observation is** — `0.047` says this combination occurs in about one case in twenty-one — and it is the quantity that compares whole models on the same data, which is the basis of Bayesian model selection. A model that assigns higher probability to what was actually observed is the better model.

**The query whose posterior is unchanged, and the independence that explains it.** `P(r | c) = 0.700000` and `P(r | b, c) = 0.700000` — learning that the battery is flat changes nothing about the probability of a tow. The graph explains it exactly: the only path from `B` to `R` is `B → C → R`, and `C`, the sole parent of `R`, is already observed. `R` is independent of its non-descendants given its parents, `B` is a non-descendant, so `C` **screens off** everything upstream. Once you know the engine will not crank, *why* it will not crank is irrelevant to whether a tow is called. Note that the unchanged posterior is not an unchanged situation: `P(e)` moves from `0.07395` to `0.04754`, because the joint event `b, c` is rarer than `c` alone. The evidence is genuinely informative about the world; it is uninformative about *this query*.

**Why the cost grows exponentially with the hidden variables.** The posterior is defined as a sum over every assignment of the hidden set, and `k` binary hidden variables have `2^k` assignments. Each is a full product over `n` nodes, giving `n × 2^(k+1)` multiplications, confirmed by the table: 2, 4, 8, 16, 32 terms as variables move out of the evidence. The depth-first form shares common prefixes and cuts 40 multiplications to 30, but that is a constant factor — the enumeration tree still has one leaf per hidden assignment. The real fix is to notice that enumeration **recomputes the same subexpressions**: a factor not mentioning the summation variable is carried through the inner loop unchanged. Summing each variable out once into a reusable factor is **variable elimination**, and it reduces the cost to exponential in the network's *treewidth* rather than its variable count — polynomial for a polytree such as this one. Exact inference in general Bayesian networks is `#P`-hard, so no algorithm removes the exponential everywhere; approximate methods (likelihood weighting, MCMC) trade exactness for a cost that does not depend on treewidth at all.

---

## Exercise 1 — A second cause of the same observed effect

> *Recompute the posterior after adding, as evidence, a second cause of the same observed effect. Report both posteriors and the direction of the change, and explain how two variables that are independent when nothing is observed can become dependent once their common effect is known.*

`starter_faulty` is the second cause of `engine_wont_crank`, the effect the evidence bears on:

```
  P(b | w, r)                           = 0.636567      P(e) = 0.04705152
  P(b | s, w, r)                        = 0.054722      P(e) = 0.01139765
                                        FALLS  (-0.581844)
```

**Both posteriors: 0.636567 → 0.054722. The direction is down**, by a factor of about twelve. With `C` observed directly, isolating the mechanism from the noise in the two indicators, the effect is the same: `P(b | c) = 0.642867 → P(b | c, s) = 0.054726`.

### How independent variables become dependent

The three numbers that tell the story:

```
  prior                        P(b)          = 0.050000
  raised by the effect         P(b | c)      = 0.642867
  lowered again by the rival   P(b | c, s)   = 0.054726
```

And the same fact as an independence test, run twice:

```
    B _||_ S                  P(b | s)     = 0.050000   P(b | ¬s)     = 0.050000   diff 0.000000
    B -/- S | C               P(b | s, c)  = 0.054726   P(b | ¬s, c)  = 0.833333   diff 0.778607
```

With nothing observed the two causes are **exactly** independent — not approximately, to the last printed digit — because the network gives them no common ancestor and no path between them except through `C`, and an unobserved common effect blocks that path. Given `c` they are strongly dependent: `0.833333` against `0.054726`, a factor of fifteen.

The mechanism is a **constraint created by the observation**. Before anything is seen, `B` and `S` are two unrelated facts about the car. Observing `c` asserts that *something* stopped the engine cranking, and the tables make that something almost certainly `b` or `s` (`P(c | ¬b, ¬s) = 0.01`). That is a joint constraint on the pair, and a constraint on a pair makes its members dependent by definition: given the constraint, learning one determines what the other must supply. Confirming `s` discharges the constraint on its own, so `b` is no longer needed, and its probability falls back towards the prior. This is **explaining away**, and it is ordinary diagnostic reasoning — a mechanic who finds a seized starter stops suspecting the battery.

The residue is the interesting part. It falls to `0.054726`, **not** to the prior `0.050000`. The gap is small but real, and `P(c | ¬b, s) = 0.90 < 1` is the reason: a faulty starter does not *guarantee* the engine will not crank, so a small part of the observation is still unaccounted for, and a flat battery remains a marginally better explanation than chance. Were that entry `1.00`, the starter would explain the observation completely and the posterior would return to exactly the prior. Explaining away is graded, and the grade is set by how completely the rival cause accounts for the effect.

### Why this is the experiment's real conclusion

The direction of change here is **downward on new evidence**. Experiment 4's engine cannot do this at all — not because its rule base is too small, but because Modus Ponens and Modus Tollens are monotonic and a derived literal is never withdrawn. Explaining away is non-monotonic by construction, and it arrives here without any special machinery: no exception mechanism, no priority ordering, no retraction. It is a consequence of the factorisation, and the same enumeration that produced the rise in case (a) produces the fall in case (b).

---

## Exercise 2 — Extending the network with a new node

> *Extend the network with one new node whose single parent is an existing variable, give it a conditional probability table, and recompute the same posterior for both values of the new node. State the new parameter count, then identify one conditional independence that the extended graph asserts and one pair of variables that remains dependent, verifying each numerically.*

A workshop meter test of the battery — a second, independent reading of a fault already in the model:

```python
METER = child("meter_reads_low", "M", ["battery_flat"], {
    (T,): 0.95,        # a flat battery nearly always reads low
    (F,): 0.03,        # a healthy one occasionally does
})
```

```
      battery_flat   B --+--> meter_reads_low  M
                         |
                         +--> engine_wont_crank  C --+--> warning_light  W
      starter_faulty S --+                           |
                                                     +--> tow_called     R
```

### The new parameter count

```
  parameters : 10 -> 12   (one row per value of the single parent)
  full joint : 31 -> 63   (the joint doubles with every variable added)
  the sum of the joint over all 64 assignments is 1.000000000000
```

**12 against 63.** The asymmetry is the whole argument for the representation: adding a variable costs the network `2^k` for its `k` parents — here **2** — while it *doubles* the full joint. The saving is now 51 numbers where it was 21, and it widens with every node added.

### The same posterior for both values of the new node

```
    P(b | w, r, m ) = 0.982290      P(e) = 0.02896686      RISES
    P(b | w, r, ¬m) = 0.082809      P(e) = 0.01808466      FALLS

    unobserved, for reference   P(b | w, r) = 0.636567
```

A single reading swings the posterior from `0.98` to `0.08` around a base of `0.64`. The meter is a far sharper instrument than the lamp because its likelihood ratio is extreme — `0.95/0.03 ≈ 32` for a low reading, `0.05/0.97 ≈ 0.05` against — and it reports on `B` **directly** rather than through the noisy chain `B → C → W`.

One check worth keeping:

```
  With the new node left unobserved the earlier posterior is
  unchanged: 0.636567 in the old network, 0.636567 in the new one
```

An **unobserved leaf sums out of the query entirely** — `∑_m P(m | b) = 1`, so the node contributes a factor of one and vanishes. Extending a network can only change existing answers through nodes that are actually observed, which is what makes incremental network building safe: adding a sensor does not silently perturb conclusions drawn before it was installed.

### One conditional independence the extended graph asserts

**`M ⫫ C | B`** — the meter reading and the cranking failure, given the battery:

```
    M _||_ C | B   INDEPENDENT
        given b           P(m | c, b)  = 0.950000   P(m | ¬c, b)  = 0.950000   diff 0.000000
        given ¬b          P(m | c, ¬b) = 0.030000   P(m | ¬c, ¬b) = 0.030000   diff 0.000000
```

Verified at **both** values of the conditioning variable, which is what the definition requires. `M`'s only parent is `B` and `C` is a non-descendant of `M`, so once `B` is known, `C` carries no further information: the posteriors are the raw table rows `0.95` and `0.03`.

### One pair that remains dependent

**`M` and `C`, unconditionally** — the same pair, with `B` hidden:

```
    M -/- C   DEPENDENT
        given -           P(m | c)  = 0.621437   P(m | ¬c)  = 0.032444   diff 0.588994
```

A factor of nineteen. With `B` hidden the path `M ← B → C` is open: the cranking failure is evidence about the battery, and the battery is what the meter reads. Taking the two results together, `M` and `C` are strongly dependent and *exactly* independent given `B` — the same common-cause pattern as `W` and `R` given `C`, now one level up the graph, and confirmation that the pattern is structural rather than a property of those particular tables.

---

## Verification

Beyond the checks printed in the run, the implementation was cross-checked against an independent brute-force implementation that materialises the entire joint table and filters it:

```
1863 queries cross-checked over both networks
  max |posterior  - brute force| = 0.000e+00
  max |P(e)       - brute force| = 0.000e+00
  max |flat sum   - depth-first| = 2.220e-16
```

Every variable of both networks as the query, against every subset of the remaining variables as evidence, at every combination of values. The posteriors agree to the bit; the two enumeration orders agree to one unit in the last place, which is floating-point summation order and nothing more.

---

## Viva questions

**1. What does a Bayesian network represent, and why is it cheaper than a full joint?**
A DAG whose nodes are random variables, each carrying `P(variable | parents)`. By the chain rule the joint factorises as the product of those tables, so the network specifies the same distribution with `∑ᵢ 2^|parents(xᵢ)|` numbers instead of `2ⁿ − 1` — 10 against 31 here. The saving comes from the edges *not* drawn: the chain rule alone would condition each variable on all its predecessors, and every conditioning variable dropped is a conditional independence assumption. The graph is those assumptions.

**2. How is an entry of the full joint computed?**
One multiplication per node: pick the row of each table selected by the assignment's values for that node and its parents, complementing where the variable is false, and multiply. For `b, ¬s, c, w, r`: `0.05 × 0.98 × 0.95 × 0.90 × 0.70 = 0.02932650`. Summing this over all 32 assignments gives 1, which verifies that the factorisation is a distribution and not merely a product.

**3. State inference by enumeration and its cost.**
For each value of the query variable, sum the joint over every assignment of the hidden variables holding the query and evidence fixed; normalise the resulting two-component vector. With `k` hidden binary variables the sum has `2^k` terms per value, each a product over `n` nodes, so `n × 2^(k+1)` multiplications — exponential in the hidden count. The depth-first form shares common prefixes and saves a constant factor (30 against 40 here) but not the exponential. Variable elimination does better by summing each variable out once into a reusable factor.

**4. What is the normalising constant, and why keep it?**
`P(e)`, the probability of the evidence — summing the unnormalised vector over the query variable sums the joint with only `e` fixed, which is the definition. Here `P(w, r) = 0.04705152`, and it is the same whichever variable is queried. It measures how surprising the observation is, and it is what compares whole models on the same data.

**5. What is explaining away, and why can no rule of Experiment 4 produce it?**
Two causes independent a priori become dependent once their common effect is observed, so confirming one lowers the posterior of the other: `P(b | w, r) = 0.637` falls to `0.055` when `s` is added. It requires a conclusion to be *weakened* by new evidence, and Modus Ponens and Modus Tollens are monotonic — a derived fact is never withdrawn. The pattern is not merely missing from that engine; it contradicts its defining property.

**6. Give a query whose posterior does not change when evidence is added, and explain why.**
`P(r | c) = 0.700000` and `P(r | b, c) = 0.700000`. `R`'s only parent `C` is already observed and `B` is a non-descendant of `R`, so the only path `B → C → R` is blocked: a variable is independent of its non-descendants given its parents. Once the engine is known not to crank, the reason is irrelevant to whether a tow is called. `P(e)` does change (`0.07395 → 0.04754`) — the evidence is informative about the world, just not about this query.

**7. Why does adding an unobserved node leave earlier posteriors unchanged?**
Its table sums to one over its own values, so an unobserved leaf contributes a factor of 1 and sums out of the query — `P(b | w, r) = 0.636567` in both the five- and six-node networks. Only observed nodes can change an existing answer, which is what makes a network safe to extend incrementally.

---

## File map

| Item in `main.py` | Purpose |
|---|---|
| `Node`, `root`, `child` | A variable, its parents, and `P(true \| parents)` (Alg. 1 step 1) |
| `Node.p`, `Node.key` | Table lookup by parent values, complemented for false |
| `BayesNet.validate` | The four structural checks and the parameter count (Alg. 1 step 2) |
| `BayesNet.joint`, `joint_factors` | A joint entry as a product over nodes (Alg. 1 step 3) |
| `BayesNet.total_probability`, `marginal` | Sum to one, and marginals by summing out (Alg. 1 step 4) |
| `BATTERY`…`TOW`, `NET`, `DIAGRAM` | The car network — two causes, one effect, two indicators |
| `Query` | The record of one answered query, for printing the steps |
| `enumeration_ask` | Posterior by enumeration (Alg. 2 steps 1–3) |
| `enumeration_ask_recursive` | The depth-first sum — cross-check and cost comparison |
| `direction`, `compare`, `report_query` | Comparing two posteriors (Alg. 2 step 4) |
| `independence_rows`, `independent`, `report_independence` | Independence verified from posteriors, not asserted |
| `exercise1` | Exercise 1 — explaining away |
| `METER`, `NET2`, `exercise2` | Exercise 2 — the extended network |
| `demo` | Prints all of the above |

## Files

| File | Contents |
|---|---|
| [`main.py`](main.py) | Full implementation — both algorithms, the independence checks, both exercises, and the demo driver |
| [`README.md`](README.md) | This write-up |
| `sample_output.txt` | Captured console output of a full run, ready to paste into the report |
| `Assignment - 5.pdf` | The original handout |

## Running

```
python main.py            # everything, standard library only
```

On Windows, if the console cannot encode `¬`:

```
set PYTHONIOENCODING=utf-8 && python main.py
```

From a REPL:

```python
from main import *

NET.validate()                                    # Algorithm 1 step 2
NET.joint({"battery_flat": T, "starter_faulty": F, "engine_wont_crank": T,
           "warning_light": T, "tow_called": T})  # 0.0293265
NET.total_probability()                           # 0.9999999999999998 -- 1, to floating point
NET.marginal("battery_flat")                      # {True: 0.05, False: 0.9499999999999998}

q = enumeration_ask(NET, "battery_flat", EVIDENCE)
q.p_true()                                        # 0.636567
q.normaliser                                      # 0.04705152  = P(w, r)
q.hidden                                          # ['starter_faulty', 'engine_wont_crank']

posterior(NET, "battery_flat", {**EVIDENCE, "starter_faulty": T})   # 0.054722 -- explaining away
independent(NET, "tow_called", "battery_flat", ["engine_wont_crank"])   # True
independent(NET, "tow_called", "warning_light")                         # False
posterior(NET2, "battery_flat", {**EVIDENCE, "meter_reads_low": T})     # 0.982290
```
