"""Experiment 6: centralized and decentralized task allocation.

Run ``python main.py`` to reproduce the fixed example, both exercises, and
the validation checks.  The implementation intentionally uses only the Python
standard library so that the message and search counts are visible rather than
hidden behind an optimisation package.
"""

from __future__ import annotations

import argparse
import itertools
import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Position = Tuple[int, int]


@dataclass(frozen=True, order=True)
class Agent:
    """An agent and its fixed position on the grid."""

    identifier: str
    position: Position


@dataclass(frozen=True, order=True)
class Task:
    """A task and the position at which it must be performed."""

    identifier: str
    position: Position


@dataclass(frozen=True)
class Allocation:
    agent: Agent
    task: Task
    cost: int


@dataclass(frozen=True)
class CentralizedResult:
    allocations: Tuple[Allocation, ...]
    total_cost: int
    assignments_evaluated: int
    messages: int
    unassigned_agents: Tuple[Agent, ...]
    unserved_tasks: Tuple[Task, ...]


@dataclass(frozen=True)
class Bid:
    agent: Agent
    task: Task
    cost: int


@dataclass(frozen=True)
class Commitment:
    """The winner for one task and every bid that contested it."""

    winner: Bid
    contenders: Tuple[Bid, ...]


@dataclass(frozen=True)
class RoundRecord:
    number: int
    bids: Tuple[Bid, ...]
    commitments: Tuple[Commitment, ...]
    messages: int


@dataclass(frozen=True)
class DecentralizedResult:
    allocations: Tuple[Allocation, ...]
    total_cost: int
    messages: int
    rounds: Tuple[RoundRecord, ...]
    unassigned_agents: Tuple[Agent, ...]
    unserved_tasks: Tuple[Task, ...]


@dataclass(frozen=True)
class ExperimentRow:
    agents_and_tasks: int
    optimal_cost: float
    decentralized_cost: float
    excess_percent: float
    centralized_messages: float
    decentralized_messages: float
    assignments_evaluated: float
    rounds: float


def manhattan(first: Position, second: Position) -> int:
    """The cost specified in the brief: |x1-x2| + |y1-y2|."""

    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def ordered(items: Iterable[Agent | Task]) -> Tuple[Agent | Task, ...]:
    return tuple(sorted(items, key=lambda item: item.identifier))


def validate_instance(agents: Sequence[Agent], tasks: Sequence[Task]) -> None:
    """Reject ambiguous identifiers before an allocation is attempted."""

    if not agents or not tasks:
        raise ValueError("at least one agent and one task are required")
    agent_ids = [agent.identifier for agent in agents]
    task_ids = [task.identifier for task in tasks]
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("agent identifiers must be unique")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task identifiers must be unique")


def cost_matrix(agents: Sequence[Agent], tasks: Sequence[Task]) -> Dict[Agent, Dict[Task, int]]:
    """Build the global matrix received by a centralized coordinator."""

    validate_instance(agents, tasks)
    return {
        agent: {task: manhattan(agent.position, task.position) for task in tasks}
        for agent in agents
    }


def _allocation_from_pairs(
    pairs: Iterable[Tuple[Agent, Task]], matrix: Mapping[Agent, Mapping[Task, int]]
) -> Tuple[Allocation, ...]:
    return tuple(
        sorted(
            (Allocation(agent, task, matrix[agent][task]) for agent, task in pairs),
            key=lambda item: item.agent.identifier,
        )
    )


def centralized_allocate(agents: Sequence[Agent], tasks: Sequence[Task]) -> CentralizedResult:
    """Enumerate all legal one-to-one matchings and retain the cheapest.

    If there are unequal numbers of agents and tasks, every member of the
    smaller side is matched; the remainder is reported explicitly.  Thus this
    function also handles the failed-agent exercise without inventing a dummy
    task or agent.
    """

    agents = tuple(sorted(agents, key=lambda agent: agent.identifier))
    tasks = tuple(sorted(tasks, key=lambda task: task.identifier))
    matrix = cost_matrix(agents, tasks)

    best_allocations: Tuple[Allocation, ...] | None = None
    best_key: Tuple[int, Tuple[Tuple[str, str], ...]] | None = None
    evaluated = 0

    if len(agents) <= len(tasks):
        # Every agent receives a distinct task; extra tasks remain unserved.
        candidates = (
            zip(agents, task_permutation)
            for task_permutation in itertools.permutations(tasks, len(agents))
        )
    else:
        # Every task receives a distinct agent; extra agents remain idle.
        candidates = (
            zip(agent_permutation, tasks)
            for agent_permutation in itertools.permutations(agents, len(tasks))
        )

    for pairs in candidates:
        allocations = _allocation_from_pairs(pairs, matrix)
        candidate_key = (
            sum(item.cost for item in allocations),
            tuple((item.agent.identifier, item.task.identifier) for item in allocations),
        )
        evaluated += 1
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_allocations = allocations

    assert best_allocations is not None  # guaranteed by validate_instance
    allocated_agents = {item.agent for item in best_allocations}
    allocated_tasks = {item.task for item in best_allocations}
    return CentralizedResult(
        allocations=best_allocations,
        total_cost=sum(item.cost for item in best_allocations),
        assignments_evaluated=evaluated,
        # One status report to the coordinator and one instruction in return.
        messages=2 * len(agents),
        unassigned_agents=tuple(agent for agent in agents if agent not in allocated_agents),
        unserved_tasks=tuple(task for task in tasks if task not in allocated_tasks),
    )


def decentralized_allocate(agents: Sequence[Agent], tasks: Sequence[Task]) -> DecentralizedResult:
    """Allocate by repeated announced intentions, exactly as in Algorithm 2.

    A still-free agent announces its nearest still-free task.  Contests are
    resolved by (cost, agent identifier); an agent's task choice is resolved by
    (cost, task identifier).  Each bid is broadcast to every *other live
    agent*, including agents that committed in an earlier round.  That is the
    literal "broadcast to everyone" convention from the experiment, so a bid
    made in an n-agent system contributes n - 1 messages.
    """

    agents = tuple(sorted(agents, key=lambda agent: agent.identifier))
    tasks = tuple(sorted(tasks, key=lambda task: task.identifier))
    matrix = cost_matrix(agents, tasks)
    still_unassigned = list(agents)
    still_free = list(tasks)
    allocations: List[Allocation] = []
    history: List[RoundRecord] = []
    message_total = 0

    while still_unassigned and still_free:
        bids = tuple(
            Bid(
                agent=agent,
                task=min(still_free, key=lambda task: (matrix[agent][task], task.identifier)),
                cost=min(matrix[agent][task] for task in still_free),
            )
            for agent in still_unassigned
        )

        by_task: Dict[Task, List[Bid]] = {}
        for bid in bids:
            by_task.setdefault(bid.task, []).append(bid)

        commitments: List[Commitment] = []
        for task in sorted(by_task, key=lambda candidate: candidate.identifier):
            contenders = tuple(sorted(by_task[task], key=lambda bid: bid.agent.identifier))
            winner = min(contenders, key=lambda bid: (bid.cost, bid.agent.identifier))
            commitments.append(Commitment(winner, contenders))

        # Exactly one message for every recipient other than the sender.
        round_messages = len(bids) * (len(agents) - 1)
        message_total += round_messages
        history.append(
            RoundRecord(
                number=len(history) + 1,
                bids=tuple(sorted(bids, key=lambda bid: bid.agent.identifier)),
                commitments=tuple(commitments),
                messages=round_messages,
            )
        )

        winners = {commitment.winner.agent for commitment in commitments}
        won_tasks = {commitment.winner.task for commitment in commitments}
        allocations.extend(
            Allocation(winner.agent, winner.task, winner.cost)
            for winner in (commitment.winner for commitment in commitments)
        )
        still_unassigned = [agent for agent in still_unassigned if agent not in winners]
        still_free = [task for task in still_free if task not in won_tasks]

    allocations = sorted(allocations, key=lambda item: item.agent.identifier)
    return DecentralizedResult(
        allocations=tuple(allocations),
        total_cost=sum(item.cost for item in allocations),
        messages=message_total,
        rounds=tuple(history),
        unassigned_agents=tuple(still_unassigned),
        unserved_tasks=tuple(still_free),
    )


def fixed_instance() -> Tuple[Tuple[Agent, ...], Tuple[Task, ...]]:
    """The hand-worked grid used in the procedure and observations.

    A1 is slightly closer to T1 than A2.  When A1 takes T1 in round 1, A2 is
    forced to T2 at cost 5 in round 2.  The coordinator sees the better swap
    A1->T2 and A2->T1.
    """

    return (
        (
            Agent("A1", (0, 0)),
            Agent("A2", (3, 0)),
            Agent("A3", (8, 7)),
            Agent("A4", (5, 6)),
        ),
        (
            Task("T1", (1, 0)),
            Task("T2", (0, 2)),
            Task("T3", (8, 6)),
            Task("T4", (5, 7)),
        ),
    )


def random_instance(size: int, rng: random.Random, side: int = 20) -> Tuple[Tuple[Agent, ...], Tuple[Task, ...]]:
    """Create one reproducible equal-sized grid instance with unique positions."""

    if size < 1 or 2 * size > (side + 1) ** 2:
        raise ValueError("size does not fit in the requested grid")
    positions = rng.sample([(x, y) for x in range(side + 1) for y in range(side + 1)], 2 * size)
    agents = tuple(Agent(f"A{i + 1}", positions[i]) for i in range(size))
    tasks = tuple(Task(f"T{i + 1}", positions[size + i]) for i in range(size))
    return agents, tasks


def experiment_series(instances_per_size: int, seed: int = 20261004) -> Tuple[ExperimentRow, ...]:
    """Exercise 1: average random 3x3 through 8x8 allocations."""

    if instances_per_size < 1:
        raise ValueError("instances_per_size must be positive")
    rng = random.Random(seed)
    rows: List[ExperimentRow] = []
    for size in range(3, 9):
        central = []
        decentralized = []
        for _ in range(instances_per_size):
            agents, tasks = random_instance(size, rng)
            central.append(centralized_allocate(agents, tasks))
            decentralized.append(decentralized_allocate(agents, tasks))

        optimal = mean(result.total_cost for result in central)
        greedy = mean(result.total_cost for result in decentralized)
        rows.append(
            ExperimentRow(
                agents_and_tasks=size,
                optimal_cost=optimal,
                decentralized_cost=greedy,
                excess_percent=100 * (greedy - optimal) / optimal if optimal else 0.0,
                centralized_messages=mean(result.messages for result in central),
                decentralized_messages=mean(result.messages for result in decentralized),
                assignments_evaluated=mean(result.assignments_evaluated for result in central),
                rounds=mean(len(result.rounds) for result in decentralized),
            )
        )
    return tuple(rows)


def assert_result_is_matching(
    result: CentralizedResult | DecentralizedResult, agents: Sequence[Agent], tasks: Sequence[Task]
) -> None:
    """Internal invariants: no duplicate agent/task and exact cost accounting."""

    allocations = result.allocations
    assert len({item.agent for item in allocations}) == len(allocations)
    assert len({item.task for item in allocations}) == len(allocations)
    assert result.total_cost == sum(item.cost for item in allocations)
    assert set(result.unassigned_agents) == set(agents) - {item.agent for item in allocations}
    assert set(result.unserved_tasks) == set(tasks) - {item.task for item in allocations}


def self_test() -> None:
    """Small deterministic checks, including the stated hand-worked result."""

    agents, tasks = fixed_instance()
    central = centralized_allocate(agents, tasks)
    decentralized = decentralized_allocate(agents, tasks)
    assert_result_is_matching(central, agents, tasks)
    assert_result_is_matching(decentralized, agents, tasks)
    assert central.total_cost == 6
    assert decentralized.total_cost == 8
    assert central.assignments_evaluated == 24
    assert central.messages == 8
    assert decentralized.messages == 15
    assert len(decentralized.rounds) == 2
    assert [(item.agent.identifier, item.task.identifier) for item in central.allocations] == [
        ("A1", "T2"), ("A2", "T1"), ("A3", "T3"), ("A4", "T4")
    ]

    # Tie breaking required by the brief: lower task ID, then lower agent ID.
    tied_agents = (Agent("A1", (0, 0)), Agent("A2", (0, 0)))
    tied_tasks = (Task("T1", (1, 0)), Task("T2", (0, 1)))
    tie_result = decentralized_allocate(tied_agents, tied_tasks)
    assert tie_result.allocations[0].task.identifier == "T1"
    assert tie_result.allocations[0].agent.identifier == "A1"
    assert tie_result.allocations[1].task.identifier == "T2"


def _names(items: Sequence[Agent | Task]) -> str:
    return ", ".join(item.identifier for item in items) if items else "none"


def print_cost_matrix(agents: Sequence[Agent], tasks: Sequence[Task]) -> None:
    matrix = cost_matrix(agents, tasks)
    task_headers = " ".join(f"{task.identifier:>5}" for task in tasks)
    print(f"  {'agent':<8}{task_headers}")
    print("  " + "-" * (8 + 6 * len(tasks)))
    for agent in agents:
        values = " ".join(f"{matrix[agent][task]:>5}" for task in tasks)
        print(f"  {agent.identifier:<8}{values}")


def print_allocations(allocations: Sequence[Allocation], indent: str = "  ") -> None:
    for item in allocations:
        print(f"{indent}{item.agent.identifier} -> {item.task.identifier}   cost {item.cost}")


def print_centralized(result: CentralizedResult) -> None:
    print("  chosen assignment:")
    print_allocations(result.allocations, "    ")
    print(f"  total cost             : {result.total_cost}")
    print(f"  assignments evaluated  : {result.assignments_evaluated}")
    print(f"  messages               : {result.messages}  "
          f"({result.messages // 2} reports + {result.messages // 2} instructions)")
    print(f"  unserved tasks         : {_names(result.unserved_tasks)}")


def print_decentralized(result: DecentralizedResult) -> None:
    for record in result.rounds:
        recipients = record.messages // len(record.bids) if record.bids else 0
        bid_text = ", ".join(
            f"{bid.agent.identifier}->{bid.task.identifier} ({bid.cost})" for bid in record.bids
        )
        print(f"  round {record.number}: {bid_text}")
        print(f"           {len(record.bids)} bids x {recipients} recipients = {record.messages} messages")
        for commitment in record.commitments:
            winner = commitment.winner
            losers = [
                f"{bid.agent.identifier} ({bid.cost})"
                for bid in commitment.contenders
                if bid.agent != winner.agent
            ]
            outcome = f"wins over {', '.join(losers)}" if losers else "uncontested"
            print(f"           {winner.task.identifier}: {winner.agent.identifier} ({winner.cost}) {outcome}")
    print("\n  final assignment:")
    print_allocations(result.allocations, "    ")
    print(f"  total cost             : {result.total_cost}")
    print(f"  rounds                 : {len(result.rounds)}")
    print(f"  messages               : {result.messages}")
    print(f"  unserved tasks         : {_names(result.unserved_tasks)}")


def print_fixed_example() -> Tuple[CentralizedResult, DecentralizedResult]:
    agents, tasks = fixed_instance()
    central = centralized_allocate(agents, tasks)
    decentralized = decentralized_allocate(agents, tasks)

    section("Fixed grid instance and cost matrix")
    print("  agents:", ", ".join(f"{agent.identifier}{agent.position}" for agent in agents))
    print("  tasks :", ", ".join(f"{task.identifier}{task.position}" for task in tasks))
    print("\n  Manhattan-distance cost matrix:")
    print_cost_matrix(agents, tasks)

    section("Algorithm 1 - centralized allocation")
    print_centralized(central)

    section("Algorithm 2 - decentralized announced intentions")
    print_decentralized(decentralized)
    print(f"\n  decentralized excess  : {decentralized.total_cost - central.total_cost} "
          f"({100 * (decentralized.total_cost - central.total_cost) / central.total_cost:.1f}% above optimum)")
    return central, decentralized


def print_exercise_one(instances_per_size: int) -> Tuple[ExperimentRow, ...]:
    rows = experiment_series(instances_per_size)
    section(f"Exercise 1 - random instances (mean of {instances_per_size}, seed 20261004)")
    print("  n    optimal    decentralized  excess      central msg  decentralized msg  assignments  rounds")
    print("  " + "-" * 94)
    for row in rows:
        print(
            f"  {row.agents_and_tasks:<4}{row.optimal_cost:>7.2f}"
            f"{row.decentralized_cost:>15.2f}{row.excess_percent:>8.1f}%"
            f"{row.centralized_messages:>14.1f}{row.decentralized_messages:>19.1f}"
            f"{row.assignments_evaluated:>13.0f}{row.rounds:>8.2f}"
        )
    print("\n  Candidate-matchings grow as n!: 8! = 40,320; 9! = 362,880; "
          "10! = 3,628,800.  Eight is comfortable for this demonstration, but")
    print("  ten is the practical cutoff for repeated exhaustive trials in plain Python;")
    print("  beyond it, the factorial coordinator search dominates the experiment.")
    return rows


def print_exercise_two() -> None:
    agents, tasks = fixed_instance()
    surviving_agents = tuple(agent for agent in agents if agent.identifier != "A2")
    central = centralized_allocate(surviving_agents, tasks)
    decentralized = decentralized_allocate(surviving_agents, tasks)
    section("Exercise 2 - remove A2 before allocation")
    print("  surviving agents:", _names(surviving_agents))
    print("\n  centralized:")
    print_allocations(central.allocations, "    ")
    print(f"    cost {central.total_cost}; unserved task: {_names(central.unserved_tasks)}; "
          f"messages: {central.messages}")
    print("\n  decentralized:")
    print_allocations(decentralized.allocations, "    ")
    print(f"    cost {decentralized.total_cost}; unserved task: {_names(decentralized.unserved_tasks)}; "
          f"messages: {decentralized.messages}; rounds: {len(decentralized.rounds)}")
    print("\n  If the coordinator itself fails before sending instructions, the centralized")
    print("  scheme returns no allocation: the agents can at most send four status reports")
    print("  into the failed endpoint, and T1, T2, T3, T4 remain unserved.  In contrast,")
    print("  the decentralized protocol has no coordinator; provided live agents can still")
    print("  hear each other's broadcasts, it continues with the announced-intention rounds.")
    print("  A fault-tolerant centralized design would replicate the coordinator's current")
    print("  cost matrix and decision log, elect a live replica on failure, and require an")
    print("  acknowledgement/epoch number so an old coordinator cannot issue stale orders.")


def section(title: str) -> None:
    print("\n" + "=" * 94)
    print(title)
    print("=" * 94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances", type=int, default=12,
        help="random instances to average at each size in Exercise 1 (default: 12)",
    )
    parser.add_argument("--test", action="store_true", help="run deterministic checks without the report")
    args = parser.parse_args()

    self_test()
    if args.test:
        print("All deterministic allocation checks passed.")
        return

    print("EXPERIMENT 6 - COORDINATION MECHANISMS")
    print("Centralized optimal matching versus decentralized announced intentions")
    print("\nMessage convention: a centralized participant sends one report and receives one")
    print("instruction (2n messages).  A decentralized bid is broadcast to every other live")
    print("agent, including already committed peers; no extra acknowledgement is counted.")
    print_fixed_example()
    print_exercise_one(args.instances)
    print_exercise_two()
    section("Validation")
    print("  self-test passed: matching uniqueness, exact cost totals, known 6-versus-8")
    print("  result, message counts, two rounds, and both prescribed tie-break rules.")


if __name__ == "__main__":
    main()