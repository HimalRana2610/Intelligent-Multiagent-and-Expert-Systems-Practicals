# Experiment 3 — Agent Architectures: The BDI Deliberation Cycle

**Intelligent Multi-Agent and Expert Systems (AI401)** · B.Tech IV (AI), Semester VII
Run with `python app.py` (standard library only, ~8 s).

---

## Aim

To implement a Belief–Desire–Intention agent that revises its beliefs about a dynamic environment, deliberates over competing desires to commit to an intention, and executes plans under a chosen commitment strategy — and to measure how that commitment affects performance as the world changes faster.

## Theory

A **reactive** agent maps percepts straight to actions. A **deliberative** agent reasons about its goals first. The BDI architecture formalises that reasoning with three attitudes:

| Attitude | Meaning | In this Tileworld |
|---|---|---|
| **Belief** | what the agent takes to be true, revised from percepts | the set of hole cells currently sensed |
| **Desire** | states of affairs it would like to bring about — *options*, not commitments, and they conflict | one `fill(x,y)` option per believed hole |
| **Intention** | the desire it has committed to and is pursuing | the hole being chased |

An intention is more than a preference. It **constrains future deliberation** and persists until it is achieved, becomes impossible, or is dropped. That persistence is the whole reason the architecture exists: without it the agent would re-open the question "what should I want?" at every step and never get anywhere.

### The control loop

Four functions:

| Function | Role | Familiar from |
|---|---|---|
| `brf(B, p)` | fold a percept into the belief set | — |
| `options(B, I)` | propose desires consistent with current beliefs and intentions | the **match** phase of Experiment 2 |
| `filter(B, D, I)` | deliberation proper — return the intention committed to | the **conflict resolution** of Experiment 5 |
| `plan(B, I)` | means–ends reasoning to produce the actions that achieve `I` | — |

What is new in this experiment is **time pressure**. The world does not pause while the agent thinks, so *deliberation is itself an action with a cost*.

### Commitment as a design decision

That cost makes commitment a real trade-off:

- an agent re-deliberating after every action is perfectly informed but barely acts;
- one that never reconsiders may spend a whole plan chasing an intention that became unachievable at its second step.

Three strategies span the range:

| Strategy | Drops `I` when… |
|---|---|
| **blind** | `I` is believed achieved |
| **single-minded** | achieved, **or** believed unachievable |
| **open-minded** | achieved, unachievable, **or** no longer among the best options |

The **reconsideration interval γ** — actions executed between successive deliberations — parameterises the spectrum. γ = 1 is a *cautious* agent; a γ larger than any plan is a *bold* one. Kinny and Georgeff showed that **bold agents dominate in slowly changing worlds, and the advantage passes to cautious agents as the rate of change rises.** Section 4 of the program reproduces exactly that.

### The testbed

A simplified **Tileworld**: a 15 × 15 grid on which holes appear at random cells with fixed probability per world tick and expire after a random lifetime; the agent scores by moving onto a hole and filling it.

```
p(hole appears)   0.10 per world tick
hole lifetime     U[20, 60] world ticks
grid              15 x 15, no obstacles
effectiveness     holes filled / holes appeared
world dynamism    world ticks per agent action  (1, 2, 4, 8, ...)
```

Because holes both appear and expire on the world clock, raising the dynamism changes the *rate* at which options churn while leaving the steady-state number of holes roughly constant at `p × lifetime ≈ 4`. The agent therefore always faces about the same number of competing desires; only the speed at which they turn over changes. That is what isolates the effect being measured.

## Algorithm

### Algorithm 1 — the BDI deliberation cycle

1. Take the first percept `p`, set `B := brf(∅, p)`, initialise `I := null` and `π := empty`.
2. Obtain the current percept `p` and revise beliefs, `B := brf(B, p)`.
3. If `I` is null or `π` is empty, generate `D := options(B, I)`, commit by `I := filter(B, D, I)`, and construct `π := plan(B, I)`. If `D` is empty, take no action this step and let the world advance.
4. Execute the first action of `π` and delete it from `π`. The world advances by its own clock while the action is performed.
5. If `I` is achieved in the revised beliefs, drop `I` and `π` and return to step 2.
6. Otherwise invoke Algorithm 2 to decide whether to reconsider `I`, then return to step 2.

### Algorithm 2 — intention reconsideration

1. Maintain a counter `s` of actions executed since the last reconsideration; increment after every action in step 4.
2. If `s < γ`, return immediately with `I` and `π` unchanged — the agent stays committed and does **not** deliberate.
3. Otherwise set `s := 0` and **charge one agent time step to deliberation**, letting the world advance while the agent reasons.
4. Re-perceive and revise beliefs. If `I` is no longer achievable, drop `I` and `π`.
5. Re-run `options` and `filter`. If the preferred option differs from `I`, adopt it and re-plan; otherwise keep `I` and `π`.
6. Report γ with the results.

> **A note on step 5.** As literally written, Algorithm 2 combines a single-minded *drop* rule (step 4) with an open-minded *switch* rule (step 5). The code therefore treats the two separately: `SINGLE_MINDED` runs steps 1–4, and `OPEN_MINDED` runs steps 1–5 (i.e. Algorithm 2 in full). The traced run and the sweep use `OPEN_MINDED`, so every step of the handout's algorithm is exercised; section 3 compares all three strategies side by side.

---

## The code

Everything is in [`app.py`](app.py). Layout:

| Lines | Contents |
|---|---|
| `Percept`, `Tileworld` | the environment and its own clock |
| `Beliefs`, `Intention` | the agent's data |
| `brf`, `options`, `deliberate`, `plan` | the four BDI functions |
| `achievable`, `is_achieved` | the tests Algorithm 2 turns on |
| `run_bdi` | Algorithms 1 and 2, commented step by step |
| `sweep`, `report_sweep`, `ascii_plot` | Exercise 1 |
| `options_bounded`, `deliberate_bounded` | Exercise 2 |
| `demo` | the driver that prints everything below |

### The world has its own clock

```python
def advance(self, ticks=None):
    for _ in range(self.dynamism if ticks is None else ticks):
        self.world_tick()
```

`execute()` moves the agent and *then* calls `advance()`; `idle()` advances without moving. Every path through the agent's loop — acting, thinking, or having nothing to do — costs exactly one agent time step and lets `dynamism` world ticks pass. That uniformity is what makes γ comparable across strategies.

### Beliefs, and how an intention dies

```python
def brf(beliefs, percept):
    return Beliefs(position=percept.position, holes=set(percept.holes), ...)
```

The Tileworld is fully observable, so belief revision looks trivial — the percept overwrites the belief set. The consequential detail is what the percept does **not** carry: hole *lifetimes* are never sensed. The agent cannot predict expiry; it discovers that a hole is gone only by no longer seeing it. Holes it believed in silently vanish from `B`, and that is exactly how an intention becomes unachievable:

```python
def achievable(beliefs, intention):
    return intention.cell in beliefs.holes
```

### Desires compete

```python
def options(beliefs, intention):
    return [Intention("fill", cell) for cell in sorted(beliefs.holes)]
```

Every believed hole is an option. They genuinely conflict, because the agent can only be in one place at a time — pursuing one means not pursuing the others.

### Deliberation and means–ends reasoning

```python
def deliberate(beliefs, desires, intention):        # filter(B, D, I)
    return min(desires, key=lambda d: (manhattan(beliefs.position, d.cell), d.cell))

def plan(beliefs, intention):                       # plan(B, I)
    moves  = ["E"] * max(0, gx - x) + ["W"] * max(0, x - gx)
    moves += ["S"] * max(0, gy - y) + ["N"] * max(0, y - gy)
    return moves
```

`filter` commits to the nearest hole (ties broken by cell order, so runs reproduce). `plan` is the means–ends step: with no obstacles the shortest path is any interleaving of the required moves.

### The loop, with the algorithm's step numbers in the comments

```python
        # -- step 3: no intention or no plan left -> deliberate afresh ------
        if I is None or not pi:
            if I is not None:
                res.futile += 1      # walked a whole plan to a cell whose hole had gone
            D = options_fn(B, I)
            I = filter_fn(B, D, I)
            ...
        # -- step 4: execute the first action of pi and delete it from pi ---
        action = pi.pop(0)
        filled = world.execute(action)
        step += 1; s += 1
        ...
        # -- step 6: Algorithm 2 -------------------------------------------
        if strategy == BLIND or s < gamma:
            continue                          # A2.2 stay committed, do not think
        s = 0
        world.idle(); step += 1               # A2.3 deliberation costs a time step
        res.deliberations += 1
```

Two counters are worth naming because the observations turn on them:

- **`deliberations`** — agent time steps burned thinking rather than moving. This is the price of a small γ.
- **`futile`** — plans walked all the way to the end without achieving the intention, i.e. the agent arrived at a cell whose hole had already expired. This is the price of a large γ.

They move in opposite directions, and the optimum γ is where their sum is smallest.

---

## Procedure and Result

`python app.py` prints five sections.

### 1. One turn of the cycle

```
  p  = percept at world tick 40: position (7, 7), 4 holes sensed
  B  := brf(0, p)          -> 4 hole beliefs [(0, 14), (7, 3), (12, 9), (13, 4)]
  D  := options(B, I)      -> 4 desires ['fill(0,14)', 'fill(7,3)', 'fill(12,9)', 'fill(13,4)']
  I  := filter(B, D, I)    -> fill(7,3)   distances ['fill(7,3):4', 'fill(12,9):7', ...]
  pi := plan(B, I)         -> ['N', 'N', 'N', 'N']  (4 moves)

   A = agent   o = hole (a desire)   * = the hole intended

     . . . . . . . . . . . . . . .
     . . . . . . . . . . . . . . .
     . . . . . . . . . . . . . . .
     . . . . . . . * . . . . . . .
     . . . . . . . . . . . . . o .
     . . . . . . . . . . . . . . .
     . . . . . . . . . . . . . . .
     . . . . . . . A . . . . . . .
     . . . . . . . . . . . . . . .
     . . . . . . . . . . . . o . .
     . . . . . . . . . . . . . . .
     . . . . . . . . . . . . . . .
     . . . . . . . . . . . . . . .
     . . . . . . . . . . . . . . .
     o . . . . . . . . . . . . . .
```

Four desires; one becomes the intention; the rest are set aside.

### 2. Event trace

`seed = 7`, world speed = 4 ticks/action, **γ = 2**, open-minded, 140 agent steps. Every event the handout asks for appears:

```
   step   tick  pos      event     detail
   ------------------------------------------------------------------------
      1      4  (7,7)    COMMIT    I := fill(8,1)   |D| = 1   |pi| = 7 moves
     11     44  (8,1)    ACHIEVE   fill(8,1) achieved -> drop I, pi     (filled = 1)
     11     44  (8,1)    COMMIT    I := fill(6,0)   |D| = 4   |pi| = 3 moves
     15     60  (6,0)    ACHIEVE   fill(6,0) achieved -> drop I, pi     (filled = 2)
     ...
     28    112  (11,4)   COMMIT    I := fill(13,9)   |D| = 3   |pi| = 7 moves
     30    120  (12,4)   DROP      fill(13,9) no longer achievable -> commit afresh
     30    120  (12,4)   COMMIT    I := fill(6,8)   |D| = 2   |pi| = 10 moves
     33    132  (10,4)   DROP      fill(6,8) no longer achievable -> commit afresh
     33    132  (10,4)   COMMIT    I := fill(3,9)   |D| = 2   |pi| = 12 moves
     36    144  (8,4)    SWITCH    fill(3,9) -> fill(13,4)  (better option on reconsideration)
     43    172  (13,4)   ACHIEVE   fill(13,4) achieved -> drop I, pi     (filled = 5)
```

| Event | What it shows |
|---|---|
| `COMMIT` | `filter` picked one of `|D|` competing desires; `plan` produced `|π|` moves |
| `ACHIEVE` | the agent reached the hole and filled it; `I` and `π` are dropped (Alg. 1 step 5) |
| `DROP` | on reconsideration the hole had expired — `I` became unachievable (Alg. 2 step 4) |
| `SWITCH` | `I` was still achievable but no longer preferred — a nearer hole had appeared (Alg. 2 step 5) |

**Result for γ = 2:**

```
  holes appeared      59
  holes filled        16
  effectiveness       16/59 = 0.271

  intentions committed 25  achieved 16  dropped 8  switched 6  futile arrivals 0
  agent steps spent acting 95  deliberating 44  idle (no desires) 1
```

44 of 140 steps went to thinking. That is the cost the γ sweep is trading against.

### 3. Commitment strategies compared

Same worlds, same γ = 2, only the reconsideration policy differs. Each cell is `effectiveness (futile arrivals per run)`:

| commitment | speed 1 | speed 2 | speed 4 | speed 8 |
|---|---|---|---|---|
| blind | **0.892** (2) | **0.553** (16) | **0.242** (40) | 0.085 (67) |
| single-minded | 0.730 (0) | 0.411 (0) | 0.187 (0) | 0.076 (0) |
| open-minded | 0.753 (0) | 0.455 (0) | 0.222 (0) | **0.088** (0) |

The blind agent never pays the deliberation charge, and in a slow world that is a bargain. But its futile arrivals climb from 2 to 67 per run as the world speeds up, and by speed 8 open-minded has overtaken it. Open-minded beats single-minded at every speed: once you have already paid for the deliberation step, step 5 is free.

---

## Observations

**Persistence of intentions.** A `DROP` event is the whole point of an intention. The agent held `fill(13,9)` across several steps *without re-asking whether it still wanted it*, and gave it up only for cause — the hole expired, so the goal became impossible. Between commit and drop, deliberation is closed: the intention filters the options rather than competing with them afresh every step. A blind agent has no drop rule at all and walks the entire plan to an empty cell.

**Single-minded vs. open-minded.** A `SWITCH` event is exactly what separates them. Single-minded commitment asks only *"is `I` still possible?"* and so keeps walking towards a distant hole while a nearer one opens up under its feet. Open-minded also asks *"is `I` still best?"* and re-commits. The extra test costs nothing — both strategies have already paid for the deliberation step, and step 5 only reuses `options` and `filter` on beliefs that were revised anyway — which is why open-minded wins at every speed here. Its risk is **thrashing**: every switch discards the progress already made towards the old intention, so an agent that switches on every marginal improvement oscillates between two holes and reaches neither.

**Why deliberation must be charged a time step.** If thinking were free, γ = 1 would win by construction — perfect information at no cost — and the commitment question would not arise at all. Charging one agent time step is what makes the trade-off real. At γ = 1 the sweep shows ~280 of the 600 steps going to deliberation: the agent thinks about as often as it moves. That is why γ = 1 is the *worst* column at every world speed despite always being the best informed.

**Which γ won.** The traced run at γ = 2 scored 16/59 = 0.271. Across the sweep the optimum was **bold** at 1 and 2 ticks per action, **γ = 4** at 4 and 8 ticks, and **γ = 2** once the speed axis is extended to 16 and 32 — falling monotonically as the world is made to change faster.

---

## Exercise 1 — Boldness against dynamism

γ ∈ {1, 2, 4, 8, bold} × world speed ∈ {1, 2, 4, 8}, averaged over **25 random seeds of 600 agent steps** each (500 runs), open-minded commitment. `bold` is γ = 10⁶, larger than the longest possible plan (2 × 14 = 28 moves).

### Effectiveness (holes filled / holes appeared)

| world speed | γ=1 | γ=2 | γ=4 | γ=8 | bold | best |
|---|---|---|---|---|---|---|
| 1 tick/action | 0.655 | 0.753 | 0.825 | 0.851 | **0.892** | bold |
| 2 ticks/action | 0.361 | 0.455 | 0.505 | 0.542 | **0.553** | bold |
| 4 ticks/action | 0.171 | 0.222 | **0.256** | 0.250 | 0.242 | γ = 4 |
| 8 ticks/action | 0.068 | 0.088 | **0.095** | 0.090 | 0.085 | γ = 4 |

### Supporting counts (mean per run: deliberations / dropped / switched / futile)

| world speed | γ=1 | γ=2 | γ=4 | γ=8 | bold |
|---|---|---|---|---|---|
| 1 | 252/6/12/0 | 166/5/10/0 | 98/3/8/0 | 53/3/6/1 | 0/0/0/**2** |
| 2 | 265/19/19/0 | 181/17/17/0 | 110/15/12/2 | 62/10/8/5 | 0/0/0/**16** |
| 4 | 273/51/29/0 | 186/46/23/0 | 115/39/14/5 | 64/29/7/15 | 0/0/0/**40** |
| 8 | **280**/110/39/0 | 190/103/23/0 | 117/81/9/10 | 65/48/4/33 | 0/0/0/**67** |

### Plot

```
  effectiveness vs. gamma, one curve per world speed
   0.892 |                                        o
   0.833 |                      o        o
   0.774 |             o
   0.715 |
   0.656 |    o
   0.598 |
   0.539 |                               +        +
   0.480 |             +        +
   0.421 |
   0.362 |    +
   0.304 |
   0.245 |             x        x        x        x
   0.186 |    x
   0.127 |
   0.068 |    #        #        #        #        #
         +---------------------------------------------
              1        2        4        8       bold
   o = 1 tick/action   + = 2 ticks   x = 4 ticks   # = 8 ticks
```

### Beyond the required range

The program also extends the speed axis, to show the optimum keeps falling rather than levelling off at 4:

| world speed | γ=1 | γ=2 | γ=4 | γ=8 | bold | best |
|---|---|---|---|---|---|---|
| 1 | 0.655 | 0.753 | 0.825 | 0.851 | **0.892** | bold |
| 2 | 0.361 | 0.455 | 0.505 | 0.542 | **0.553** | bold |
| 4 | 0.171 | 0.222 | **0.256** | 0.250 | 0.242 | 4 |
| 8 | 0.068 | 0.088 | **0.095** | 0.090 | 0.085 | 4 |
| 16 | 0.027 | **0.031** | 0.029 | 0.031 | 0.027 | 2 |
| 32 | 0.011 | **0.013** | 0.011 | 0.010 | 0.010 | 2 |

**bold → bold → 4 → 4 → 2 → 2.** Monotonically down, exactly the Kinny–Georgeff result.

### Why the optimum falls as the world speeds up

The agent's 600 steps are spent in three ways: moving usefully, thinking, and moving uselessly. γ trades the second against the third.

- **The cost of checking is fixed.** One deliberation costs one agent time step, whatever the world is doing. Over 600 steps a γ-agent spends roughly `600/(γ+1)` steps thinking — 280 at γ = 1, 65 at γ = 8, 0 when bold.
- **The cost of *not* checking grows with dynamism.** Between two reconsiderations the agent is flying blind for γ actions. The chance that its intention dies during that window is the chance the hole expires, and that rises directly with world ticks per action. The futile column measures the damage: 2 wasted plans per run at speed 1, 67 at speed 8.

At speed 1 a hole lives 20–60 agent steps and a typical plan is ~7 steps, so an intention almost never dies mid-plan; checking buys nothing and its fixed cost is pure loss — hence bold wins. As the speed rises, hole lifetimes shrink to a handful of agent steps, plans die routinely, and the expected waste from waiting γ steps overtakes the one step a check costs. The break-even γ is where *marginal waste avoided = 1 step*, and since the waste per unit of γ scales with dynamism while the check price does not, that break-even point moves down.

### Why both extremes lose

They fail in opposite ways, and each failure is visible in a different column:

- **γ = 1 (reconsider every action)** — the `deliberations` column. It is never wrong about the world, but it spends about half its budget being right instead of moving. It is the worst column at *every* speed.
- **bold (never reconsider)** — the `futile` column. It spends nothing on thinking, but walks whole plans to cells whose holes vanished long ago. 67 futile arrivals per run at speed 8, each one several steps thrown away.

An intermediate γ pays the charge rarely enough to keep acting, yet often enough to catch a dead intention within a few steps of its death. That is the general shape of the result: **commitment is a resource-allocation decision between acting and knowing, and neither corner of that trade is ever optimal in a world that changes at a middling rate.**

---

## Exercise 2 — A resource-bounded filter

The agent gets a **battery of capacity 40**, losing one unit per move, and a **charger cell** at (7,7) that restores it. The plan library holds two schemas:

```python
def options_bounded(beliefs, intention):
    """fill-hole  applicable only when the battery covers the trip to the hole
                  PLUS the return to the charger
       recharge   always applicable"""
    D = []
    for cell in sorted(beliefs.holes):
        cost = manhattan(beliefs.position, cell) + manhattan(cell, beliefs.charger)
        if beliefs.battery >= cost:
            D.append(Intention("fill", cell))
    D.append(Intention("recharge", beliefs.charger))
    return D

def deliberate_bounded(beliefs, desires, intention):
    """Commit to the nearest hole whose precondition holds, recharge otherwise."""
    fills = [d for d in desires if d.kind == "fill"]
    if fills:
        return min(fills, key=lambda d: (manhattan(beliefs.position, d.cell), d.cell))
    return Intention("recharge", beliefs.charger)
```

Testing whether a schema is applicable is the **match** phase of Experiment 2; choosing among the applicable ones is the **conflict resolution** of Experiment 5. Here they are `options` and `filter` respectively.

### Result

Both agents, same 25 worlds, same γ = 2, same open-minded commitment, 600 steps. The *only* difference is the filter.

| agent | holes filled | holes appeared | effectiveness | steps stranded |
|---|---|---|---|---|
| battery-aware filter | **51.8** | 120.8 | **0.430** | **0.0** |
| battery-blind (nearest hole) | 6.6 | 120.2 | 0.055 | 347.5 |

The battery-blind agent ran flat in **25 of 25 worlds** and then spent 348 of its 600 steps stranded — nearly 60% of its life motionless. It fills 6.6 holes before dying; the bounded agent fills 51.8, roughly 8× as many.

### Why the aware agent never strands

The precondition is an **invariant**, not just an entry test. Walking one step towards the hole costs one unit of battery *and* removes one step from `d(agent, hole)`, while `d(hole, charger)` is unchanged. So

```
battery - 1  >=  (d(agent,hole) - 1) + d(hole,charger)
```

holds whenever the original did. A `fill-hole` schema that was affordable when adopted stays affordable for its whole execution, and the return leg is always covered. (With capacity 40 and the charger at the centre of a 15 × 15 grid, the worst round trip is 14 + 14 = 28, so a full battery can reach anywhere — the constraint bites only when the battery is already part-drained.)

### Which BDI component does the fix belong in — and why not the planner?

**`options` and `filter` — the deliberation stage.** `options` decides which schemas are *applicable*, and `filter` chooses among them, so the battery precondition screens an unaffordable hole out **before the agent commits to it**.

The planner cannot fix this, for a structural reason: **`plan(B, I)` runs after commitment, on an intention already adopted.** Given `fill(3,11)` it can only do means–ends reasoning towards (3,11), and the shortest path it returns is already *correct* — there is no better plan. The error is not in how the agent gets to the goal; the error is that the goal was adopted at all. A planner that noticed the problem and refused to plan would be worse than useless: the agent would be left holding an intention it can never discharge, and Algorithm 1 step 3 would re-plan the same dead goal every cycle, live-locking.

This is the general division of labour in BDI, and the reason the architecture separates the two: **`filter` decides *what is worth doing*; `plan` decides *how to do it*.** A resource constraint is a fact about which goals are worth adopting, so it belongs upstream of commitment — in the same place as any other applicability condition.

---

## Files

| File | Contents |
|---|---|
| [`app.py`](app.py) | Full implementation — Tileworld, the four BDI functions, Algorithms 1 & 2, both exercises, and the demo driver |
| [`README.md`](README.md) | This write-up |
| `sample_output.txt` | Captured console output of a full run, ready to paste into the report |
| `Assignment - 3.pdf` | The original handout |

## Running

```
python app.py            # everything, ~8 s, standard library only
```

Individual pieces from a REPL:

```python
from app import *

world = Tileworld(seed=7, dynamism=4)
res = run_bdi(world, gamma=2, strategy=OPEN_MINDED, steps=600, trace=True)
print(res.effectiveness, res.dropped, res.switched, res.futile)

report_sweep(sweep())                       # Exercise 1
report_battery(*compare_battery())          # Exercise 2
```
