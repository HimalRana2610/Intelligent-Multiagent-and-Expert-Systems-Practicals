"""Experiment 7: Contract Net and alternating-offers negotiation.

Run ``python main.py`` for the complete transcript and exercises.
Run ``python main.py --test`` for deterministic protocol checks.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Position = Tuple[int, int]


@dataclass(frozen=True)
class Agent:
    identifier: str
    position: Position


@dataclass(frozen=True)
class Task:
    identifier: str
    position: Position


@dataclass(frozen=True)
class ACLMessage:
    performative: str
    sender: str
    receiver: str
    content: str
    conversation_id: str
    protocol: str = "fipa-contract-net"


@dataclass(frozen=True)
class Award:
    task: Task
    contractor: Agent
    cost: int
    bid: float


@dataclass(frozen=True)
class ContractNetResult:
    messages: Tuple[ACLMessage, ...]
    awards: Tuple[Award, ...]
    total_paid: float
    contractor_profit: float
    unserved_tasks: Tuple[Task, ...]


@dataclass(frozen=True)
class Strategy:
    name: str
    exponent: float


@dataclass(frozen=True)
class BargainingResult:
    messages: Tuple[ACLMessage, ...]
    agreement_price: float | None
    agreement_round: int | None
    buyer_utility: float
    seller_utility: float


def manhattan(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def fixed_instance() -> Tuple[Tuple[Agent, ...], Tuple[Task, ...]]:
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


def cost_matrix(agents: Sequence[Agent], tasks: Sequence[Task]) -> Dict[str, Dict[str, int]]:
    return {
        agent.identifier: {
            task.identifier: manhattan(agent.position, task.position)
            for task in tasks
        }
        for agent in agents
    }


def contract_net(
    agents: Sequence[Agent],
    tasks: Sequence[Task],
    reservations: Mapping[str, float] | None = None,
    markup_agent: str | None = None,
    markup_factor: float = 1.0,
    task_order: Sequence[str] | None = None,
) -> ContractNetResult:
    """Run one Contract Net allocation, retaining every FIPA-ACL message."""

    if markup_factor < 1:
        raise ValueError("markup_factor must be at least 1")
    matrix = cost_matrix(agents, tasks)
    reserve = {agent.identifier: float("inf") for agent in agents}
    reserve.update(reservations or {})
    by_task = {task.identifier: task for task in tasks}
    ordered_ids = list(task_order or [task.identifier for task in tasks])
    if set(ordered_ids) != set(by_task) or len(ordered_ids) != len(by_task):
        raise ValueError("task_order must contain every task exactly once")

    free = {agent.identifier: agent for agent in agents}
    messages: List[ACLMessage] = []
    awards: List[Award] = []
    manager = "manager"
    for task_id in ordered_ids:
        task = by_task[task_id]
        conversation = f"cn-{task.identifier}"
        candidates = tuple(free.values())
        for agent in candidates:
            messages.append(ACLMessage("cfp", manager, agent.identifier,
                                       f"perform {task.identifier}", conversation))
        proposals: List[Tuple[Agent, float, float]] = []
        for agent in candidates:
            cost = matrix[agent.identifier][task.identifier]
            bid = float(cost)
            if agent.identifier == markup_agent:
                bid *= markup_factor
            if bid <= reserve[agent.identifier]:
                proposals.append((agent, bid, float(cost)))
                content = f"bid={bid:g}; cost={cost}"
                messages.append(ACLMessage("propose", agent.identifier, manager,
                                            content, conversation))
            else:
                messages.append(ACLMessage(
                    "refuse", agent.identifier, manager,
                    f"reservation={reserve[agent.identifier]:g}", conversation,
                ))
        if not proposals:
            continue
        winner, bid, cost = min(proposals, key=lambda item: (item[1], item[0].identifier))
        for agent, other_bid, _ in proposals:
            performative = "accept-proposal" if agent == winner else "reject-proposal"
            messages.append(ACLMessage(
                performative, manager, agent.identifier,
                f"bid={other_bid:g}", conversation,
            ))
        messages.append(ACLMessage(
            "inform-done", winner.identifier, manager,
            f"{task.identifier} completed; payment={bid:g}", conversation,
        ))
        awards.append(Award(task, winner, int(cost), bid))
        del free[winner.identifier]

    profit = sum(award.bid - award.cost for award in awards)
    served = {award.task.identifier for award in awards}
    return ContractNetResult(
        tuple(messages), tuple(awards), sum(award.bid for award in awards),
        profit, tuple(task for task in tasks if task.identifier not in served),
    )


def offer(opening: float, reservation: float, round_number: int,
          deadline: int, exponent: float) -> float:
    progress = (round_number / deadline) ** exponent
    return opening + (reservation - opening) * progress


def alternating_offers(
    buyer: Strategy,
    seller: Strategy,
    deadline: int = 10,
    buyer_opening: float = 4,
    buyer_reservation: float = 12,
    seller_opening: float = 16,
    seller_reservation: float = 8,
) -> BargainingResult:
    """Exchange offers until the seller's ask meets the buyer's offer."""

    if deadline < 1:
        raise ValueError("deadline must be positive")
    messages: List[ACLMessage] = []
    for round_number in range(1, deadline):
        buyer_offer = offer(
            buyer_opening, buyer_reservation, round_number, deadline, buyer.exponent
        )
        seller_offer = offer(
            seller_opening, seller_reservation, round_number, deadline, seller.exponent
        )
        conversation = "ao-bike"
        messages.append(ACLMessage(
            "propose", "buyer", "seller", f"round={round_number}; price={buyer_offer:.2f}",
            conversation, "fipa-alternating-offers",
        ))
        messages.append(ACLMessage(
            "propose", "seller", "buyer", f"round={round_number}; price={seller_offer:.2f}",
            conversation, "fipa-alternating-offers",
        ))
        if seller_offer <= buyer_offer:
            messages.append(ACLMessage(
                "accept-proposal", "buyer", "seller",
                f"round={round_number}; price={seller_offer:.2f}",
                conversation, "fipa-alternating-offers",
            ))
            return BargainingResult(
                tuple(messages), seller_offer, round_number,
                buyer_reservation - seller_offer, seller_offer - seller_reservation,
            )
    messages.append(ACLMessage(
        "cancel", "buyer", "seller", "deadline expired",
        "ao-bike", "fipa-alternating-offers",
    ))
    return BargainingResult(tuple(messages), None, None, 0.0, 0.0)


STRATEGIES = (
    Strategy("boulware", 0.2),
    Strategy("linear", 1.0),
    Strategy("conceder", 2.0),
)


def print_matrix(agents: Sequence[Agent], tasks: Sequence[Task]) -> None:
    matrix = cost_matrix(agents, tasks)
    print("Cost matrix (Manhattan distance):")
    print("       " + " ".join(f"{task.identifier:>5}" for task in tasks))
    for agent in agents:
        print(f"  {agent.identifier:>3}  " +
              " ".join(f"{matrix[agent.identifier][task.identifier]:>5}" for task in tasks))


def print_transcript(messages: Iterable[ACLMessage]) -> None:
    for message in messages:
        print(f"  [{message.conversation_id}] {message.performative:<16} "
              f"{message.sender} -> {message.receiver}: {message.content}")


def print_contract_net(result: ContractNetResult) -> None:
    print("Contract Net transcript:")
    print_transcript(result.messages)
    print("Awards:")
    for award in result.awards:
        print(f"  {award.task.identifier} -> {award.contractor.identifier} "
              f"bid={award.bid:g}, cost={award.cost}")
    print(f"Total paid: {result.total_paid:g}")
    print(f"Messages exchanged: {len(result.messages)}")
    print(f"Unserved tasks: {', '.join(task.identifier for task in result.unserved_tasks) or 'none'}")


def exercise_one() -> None:
    print("\nExercise 1: alternating-offers strategy matrix")
    print("buyer/seller        price   round  buyer utility  seller utility")
    for buyer, seller in itertools.product(STRATEGIES, repeat=2):
        result = alternating_offers(buyer, seller)
        price = f"{result.agreement_price:.2f}" if result.agreement_price is not None else "none"
        round_text = str(result.agreement_round or "-")
        print(f"{buyer.name:>8}/{seller.name:<8} {price:>7} {round_text:>7}"
              f" {result.buyer_utility:>14.2f} {result.seller_utility:>14.2f}")


def exercise_two(agents: Sequence[Agent], tasks: Sequence[Task]) -> None:
    print("\nExercise 2: markup sweep for contractor A2")
    print("factor  won tasks  served  total paid  A2 profit")
    reservations = {agent.identifier: 20.0 for agent in agents}
    first_loss: float | None = None
    for factor in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        result = contract_net(
            agents, tasks, reservations=reservations,
            markup_agent="A2", markup_factor=factor,
        )
        won = ",".join(award.task.identifier for award in result.awards
                       if award.contractor.identifier == "A2") or "-"
        if won == "-" and first_loss is None:
            first_loss = factor
        print(f"{factor:>6.2f}  {won:>9}  {len(result.awards):>6}"
              f"  {result.total_paid:>11.2f}  {result.contractor_profit:>9.2f}")
    print(f"A2 first stops winning at markup factor: {first_loss:g}")


def self_test() -> None:
    agents, tasks = fixed_instance()
    result = contract_net(agents, tasks)
    assert len(result.awards) == 4
    assert len({award.contractor for award in result.awards}) == 4
    assert result.total_paid == 8
    assert all(message.conversation_id for message in result.messages)
    assert {message.performative for message in result.messages} == {
        "cfp", "propose", "accept-proposal", "reject-proposal", "inform-done",
    }
    refused = contract_net(agents, tasks, reservations={"A2": 2})
    assert any(message.performative == "refuse" for message in refused.messages)
    assert alternating_offers(STRATEGIES[0], STRATEGIES[0]).agreement_price is not None
    assert alternating_offers(
        STRATEGIES[0], STRATEGIES[0], buyer_reservation=10, seller_reservation=10
    ).agreement_price is None
    agreement = alternating_offers(STRATEGIES[2], STRATEGIES[0])
    assert agreement.agreement_round is not None
    assert agreement.agreement_price is not None
    assert agreement.buyer_utility >= 0 and agreement.seller_utility >= 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="run validation checks only")
    args = parser.parse_args()
    self_test()
    if args.test:
        print("self-test passed")
        return

    agents, tasks = fixed_instance()
    print_matrix(agents, tasks)
    print_contract_net(contract_net(agents, tasks, reservations={"A1": 20, "A2": 20,
                                                                  "A3": 20, "A4": 20}))
    print("\nAlternating-offers example: conceder buyer vs boulware seller")
    print_transcript(alternating_offers(STRATEGIES[2], STRATEGIES[0]).messages)
    exercise_one()
    exercise_two(agents, tasks)
    print("\nValidation: self-test passed")


if __name__ == "__main__":
    main()