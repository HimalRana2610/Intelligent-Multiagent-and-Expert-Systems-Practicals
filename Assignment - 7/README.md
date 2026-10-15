# Experiment 7 — Negotiation Strategies

**Intelligent Multi-Agent and Expert Systems (AI401)** · B.Tech IV (AI), Semester VII

Run the complete experiment with:

```powershell
python main.py
```

Run only deterministic validation checks with:

```powershell
python main.py --test
```

## Aim

This experiment simulates negotiation using FIPA-ACL messages under:

1. **Contract Net**, where a manager broadcasts a call for proposals and awards
   each task to the lowest eligible bid.
2. **Alternating offers**, where a buyer and seller exchange time-dependent
   prices until their reservation ranges overlap or a deadline expires.

Every message is represented explicitly by its performative, sender, receiver,
content, conversation identifier, and protocol.  Conversation identifiers keep
simultaneous task negotiations separate.

## Model

The grid and Manhattan costs are the same as Experiment 6:

```text
Agents: A1(0,0), A2(3,0), A3(8,7), A4(5,6)
Tasks : T1(1,0), T2(0,2), T3(8,6), T4(5,7)
cost(A,T) = |Ax-Tx| + |Ay-Ty|
```

For each task, the Contract Net transcript contains `cfp`, one `propose` or
`refuse` from each free contractor, `accept-proposal` and
`reject-proposal` responses, and the winning contractor's `inform-done`.
Reservations are inclusive: a contractor proposes when its bid is no greater
than its reservation value.

The bargaining example uses a buyer opening at 4 with reservation 12 and a
seller opening at 16 with reservation 8.  A strategy exponent below one is
boulware (late concession), one is linear, and above one is conceder (early
concession).  The run checks rounds 1 through 9; an overlap is accepted at the
seller's offer, and an unresolved negotiation sends `cancel` at the deadline.

## Exercises and observations

### Procedure

The program prints the complete cost matrix, a Contract Net transcript with all
performative types, awards, payment, message count, and an alternating-offers
transcript.  It then prints:

* all nine buyer/seller strategy pairings, including price, round, and both
  utilities;
* a markup sweep for contractor A2, including won tasks, tasks served, total
  payment, and A2's profit.

### Interpretation

* A `refuse` still consumes the `cfp` and response messages, but it cannot win
  the task.  A losing `propose` additionally creates a manager response, so it
  costs more protocol traffic.
* Task order can change the final payment because a contractor becomes free or
  committed after each award.  An accepted award cannot be undone by a later
  conversation; this is the commitment rule of the Contract Net.
* When concession rates differ, the early conceder gives up more surplus while
  the boulware side generally obtains the more favourable share.
* The deadline limits bargaining delay and forces an explicit `cancel` instead
  of an implicit or invented agreement.  Two boulware agents reach agreement
  early here because the reservation ranges overlap; with non-overlapping
  reservations the same protocol would end with `cancel`.
* Increasing a contractor's markup can increase profit on a task it still wins,
  but eventually exceeds its reservation and makes it refuse, leaving work
  unserved in the sweep.  Thus the lowest-bid rule makes strategic markup
  risky; truthful bidding is not automatically dominant in a first-price
  Contract Net.

## Validation

The self-test checks unique awards, exact payment and profit accounting,
conversation identifiers, visibility of every required Contract Net
performative, refusal handling, deadline cancellation, and non-negative
utilities for an agreement.
