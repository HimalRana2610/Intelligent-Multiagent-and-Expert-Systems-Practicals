# Experiment 6 — Coordination Mechanisms: Centralized and Decentralized Task Allocation

**Intelligent Multi-Agent and Expert Systems (AI401)** · B.Tech IV (AI), Semester VII

Run the complete experiment with:

```powershell
python main.py
```

The program uses only the Python standard library.  It prints the cost matrix, the two allocations, every decentralized bid round, both exercises, and deterministic validation checks.  `python main.py --test` runs only the checks; `--instances N` changes the 12 random instances averaged at each Exercise-1 size.

---

## Aim

Allocate a set of grid tasks to agents under two coordination mechanisms:

1. a **central coordinator** that has the complete cost matrix and enumerates assignments, and
2. a **decentralized announced-intention protocol** in which agents select tasks locally and resolve conflicts by a fixed rule.

Compare solution quality, message count, search effort, and resilience when an agent or the coordinator fails.

## Model and conventions

An agent or task has a fixed integer grid position `(x, y)`.  The cost of assigning agent `A` to task `T` is Manhattan distance:

```text
cost(A, T) = |Ax - Tx| + |Ay - Ty|
```

Both algorithms allocate the smaller of the agent and task sets, so a removed agent is handled honestly: one task remains unserved rather than being silently assigned to a nonexistent agent.

Message accounting is deliberately stated up front.

| mechanism | messages counted |
|---|---|
| centralized | one status/cost report from every participating agent plus one assignment instruction back: `2n` |
| decentralized | every intention is broadcast to every **other live agent**, including agents that committed in an earlier round: one bid in an `n`-agent system costs `n − 1` messages |

The brief says that each decentralized announcement is broadcast to everyone.  No acknowledgement messages are added, so both counts are reproducible from the printed trace.

## Algorithms implemented

### 1. Centralized allocation

1. Build the complete agent-by-task Manhattan-distance matrix.
2. Enumerate each legal one-to-one matching.  With `n` agents and `n` tasks this is `n!` permutations.
3. Select the matching with least total cost.  Equal-cost matchings are made deterministic by agent/task identifier order.
4. Report assignment, cost, permutations evaluated, messages, and any unserved task.

The relevant implementation is [`centralized_allocate`](main.py).  It does exhaustive search intentionally: this is what exposes the optimality/search trade-off in the experiment.

### 2. Decentralized announced intentions

1. Every unassigned agent chooses its lowest-cost free task; a tie goes to the lower task identifier.
2. Each agent broadcasts that bid to all other live agents.
3. For every contested task, its lowest-cost bidder wins; a cost tie goes to the lower agent identifier.
4. Commit winners and their tasks, then repeat with the losing agents and still-free tasks.

[`decentralized_allocate`](main.py) retains each `Bid`, the contenders, winner, round, and messages, so the result is a protocol trace rather than only a final matching.

## Procedure and result

### Fixed grid instance

```text
Agents: A1(0,0), A2(3,0), A3(8,7), A4(5,6)
Tasks : T1(1,0), T2(0,2), T3(8,6), T4(5,7)

             T1  T2  T3  T4
    A1        1   2  14  12
    A2        2   5  11   9
    A3       14  13   1   3
    A4       10   9   3   1
```

The matrix is deliberately shaped around `A1`, `A2`, `T1`, and `T2`: A1 is only slightly closer to T1 than A2, but A2 is much farther from T2.  This makes a locally attractive early choice globally harmful.

### Centralized result

```text
A1 -> T2   cost 2
A2 -> T1   cost 2
A3 -> T3   cost 1
A4 -> T4   cost 1

Total cost: 6
Assignments evaluated: 4! = 24
Messages: 8 = 4 reports + 4 instructions
```

The coordinator sees that assigning A1 to its *second* choice (`T2`) releases T1 for A2, avoiding A2's expensive alternative.

### Decentralized result and bid trace

```text
Round 1 bids: A1->T1 (1), A2->T1 (2), A3->T3 (1), A4->T4 (1)
              4 bids × 3 recipients = 12 messages
              T1: A1 wins over A2; T3: A3 wins; T4: A4 wins

Round 2 bids: A2->T2 (5)
              1 bid × 3 recipients = 3 messages
              T2: A2 wins

A1 -> T1   cost 1
A2 -> T2   cost 5
A3 -> T3   cost 1
A4 -> T4   cost 1

Total cost: 8
Rounds: 2
Messages: 15
```

The decentralized allocation is `2` cost units, or **33.3%**, above optimum.  Its outcome is valid — no agent or task is duplicated — but greedy rather than globally optimal.

## Exercise 1 — Random instances from 3 to 8 agents/tasks

Twelve equal-sized random grid instances were generated at each size using the fixed seed `20261004`; each row is the mean over those twelve instances.  The exact run is repeatable by `python main.py --instances 12`.

| n | mean optimal cost | mean decentralized cost | excess over optimum | centralized messages | decentralized messages | assignments evaluated per instance | mean rounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 32.00 | 33.17 | 3.6% | 6.0 | 8.7 | 6 | 2.08 |
| 4 | 34.58 | 37.42 | 8.2% | 8.0 | 16.5 | 24 | 2.17 |
| 5 | 41.50 | 45.50 | 9.6% | 10.0 | 28.7 | 120 | 2.42 |
| 6 | 52.33 | 57.67 | 10.2% | 12.0 | 46.2 | 720 | 2.75 |
| 7 | 55.25 | 61.58 | 11.5% | 14.0 | 68.0 | 5,040 | 3.08 |
| 8 | 57.33 | 63.50 | 10.8% | 16.0 | 93.3 | 40,320 | 3.33 |

The centralized communication count is linear, `2n`, but its search work is factorial.  The decentralized method needs no factorial search, but broadcasts make its communication grow much faster: even a one-round allocation has `n(n − 1)` messages, and repeated conflicts add rounds.  With this literal all-peer broadcast convention, a sequence in which only one agent commits each round has `Θ(n³)` messages.

`8! = 40,320` candidate matchings is comfortable for a single small lab run.  At `10! = 3,628,800` candidates **per instance** (and `11! = 39,916,800`), repeated exhaustive averages stop being practical in this straightforward Python implementation.  Thus 10 agents/tasks is the practical cutoff for this experiment; a real centralized system would replace exhaustive enumeration with an assignment algorithm such as Hungarian matching.

## Exercise 2 — Agent failure and coordinator failure

Remove `A2` before allocation, leaving `A1`, `A3`, and `A4` to serve four tasks.

| mechanism | allocation | cost | unserved task | messages |
|---|---|---:|---|---:|
| centralized | A1→T1, A3→T3, A4→T4 | 3 | T2 | 6 |
| decentralized | A1→T1, A3→T3, A4→T4 | 3 | T2 | 6 |

Here all surviving agents choose different tasks, so the decentralized protocol finishes in one round.  Neither mechanism can serve T2 because there are only three live agents; both report it instead of concealing the missing work.

If the **coordinator** fails before issuing instructions, centralized allocation returns no matching.  Agents may send their four status reports to the failed endpoint, but receive no decisions and every task remains unserved.  A decentralized group has no such single indispensable process: if the live agents can still hear one another, its bid rounds continue.

To make the centralized scheme survive this failure, replicate the coordinator's cost matrix and committed-decision log at several peers; elect a live replica on failure; and tag decisions with an epoch/acknowledgement so a recovered old coordinator cannot send stale instructions.

## Observations

- The central solution costs **6** and the decentralized solution costs **8**.  A2 pays the penalty: it loses T1 in round 1 and is left with T2 at cost 5 in round 2.  The locally committed choice that creates the problem is A1→T1 in **round 1**; A1 gains one unit compared with its centralized task, while A2 loses three.
- Centralization is best when a globally minimal allocation matters and a reliable coordinator is available.  It costs only `2n` messages, but exact enumeration becomes factorial.
- Decentralization avoids both global state and the coordinator single point of failure.  Its price is extra broadcast traffic and greedy commitments that cannot be reconsidered when a later agent is left with an expensive task.
- A normal agent failure reduces capacity in both mechanisms, making an unserved task explicit.  A coordinator failure is qualitatively worse for the unreplicated centralized design: it stops the whole allocation, whereas the decentralized protocol continues among the surviving agents.

## Verification

`main.py` performs deterministic checks before printing the report:

- every allocation has unique agents and unique tasks;
- the reported total equals the sum of assignment costs;
- the fixed instance has central cost `6`, decentralized cost `8`, `24` central candidates, `8` central messages, `15` decentralized messages, and two decentralized rounds;
- both stated tie-break rules are tested (lower task identifier, then lower agent identifier).

Run:

```powershell
python main.py --test
```

Expected result:

```text
All deterministic allocation checks passed.
```
