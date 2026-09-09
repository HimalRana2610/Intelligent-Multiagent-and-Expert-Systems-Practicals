# Knowledge base as (subject, relation, object) triples
KB = [
    ("animal", "has", "skin"),
    ("fish", "is-a", "animal"),
    ("fish", "can", "swim"),
    ("shark", "is-a", "fish"),
    ("bird", "is-a", "animal"),
    ("bird", "can", "fly"),
    ("bird", "has", "wings"),
    ("penguin", "is-a", "bird"),
    ("penguin", "cannot", "fly"),
    ("penguin", "can", "swim"),
    # Exercise 1: ostrich
    ("ostrich", "is-a", "bird"),
    ("ostrich", "cannot", "fly"),
    ("ostrich", "can", "run_fast"),
]


def get_parents(node, kb=KB):
    return [obj for subj, rel, obj in kb if subj == node and rel == "is-a"]


def get_chain(node, kb=KB):
    chain = [node]
    visited = {node}
    queue = list(get_parents(node, kb))

    while queue:
        parent = queue.pop(0)
        if parent in visited:
            continue
        visited.add(parent)
        chain.append(parent)
        queue.extend(get_parents(parent, kb))

    return chain


def local_facts(node, relation, kb=KB):
    return [obj for subj, rel, obj in kb if subj == node and rel == relation]


def query(node, relation, kb=KB):
    for concept in get_chain(node, kb):
        # Exceptions win over inherited defaults (non-monotonic)
        cannot = local_facts(concept, "cannot", kb)
        if relation == "can" and cannot:
            return f"cannot {cannot[0]}"

        values = local_facts(concept, relation, kb)
        if values:
            return values[0]

    return "unknown"


# Exercise 2: Same knowledge as frames (slots / fillers / ako)
FRAMES = {
    "animal": {"has": "skin"},
    "fish": {"ako": "animal", "can": "swim"},
    "shark": {"ako": "fish"},
    "bird": {"ako": "animal", "can": "fly", "has": "wings"},
    "penguin": {"ako": "bird", "cannot": "fly", "can": "swim"},
    "ostrich": {"ako": "bird", "cannot": "fly", "can": "run_fast"},
}


def frame_chain(concept, frames=FRAMES):
    chain = [concept]
    current = concept
    while current in frames and "ako" in frames[current]:
        parent = frames[current]["ako"]
        if parent in chain:
            break
        chain.append(parent)
        current = parent
    return chain


def frame_query(concept, slot, frames=FRAMES):
    for name in frame_chain(concept, frames):
        frame = frames.get(name, {})
        if slot == "can" and "cannot" in frame:
            return f"cannot {frame['cannot']}"
        if slot in frame:
            return frame[slot]
    return "unknown"


# Exercise 3: Multiple parents — search order justification
KB_MULTI = KB + [
    ("bat", "is-a", "mammal"),
    ("bat", "is-a", "flyer"),
    ("mammal", "is-a", "animal"),
    ("mammal", "has", "fur"),
    ("flyer", "can", "fly"),
]


def demo():
    print("=" * 60)
    print("Knowledge Base (triples)")
    print("=" * 60)
    for triple in KB:
        print(f"  {triple}")

    print("\n" + "=" * 60)
    print("Inheritance chains")
    print("=" * 60)
    for node in ("shark", "penguin", "ostrich"):
        print(f"  {node}: {' -> '.join(get_chain(node))}")

    print("\n" + "=" * 60)
    print("Queries (semantic network)")
    print("=" * 60)
    tests = [
        ("shark", "has"),
        ("shark", "can"),
        ("penguin", "can"),
        ("penguin", "has"),
        ("ostrich", "can"),
        ("ostrich", "has"),
        ("bird", "can"),
    ]
    for node, rel in tests:
        print(f"  query({node!r}, {rel!r}) -> {query(node, rel)!r}")

    print("\n" + "=" * 60)
    print("Exercise 2: Frame inheritance")
    print("=" * 60)
    for node, slot in tests:
        print(f"  frame_query({node!r}, {slot!r}) -> {frame_query(node, slot)!r}")

    print("\n" + "=" * 60)
    print("Exercise 3: Multiple parents (bat)")
    print("=" * 60)
    print(f"  chain: {' -> '.join(get_chain('bat', KB_MULTI))}")
    print(f"  query('bat', 'can')  -> {query('bat', 'can', KB_MULTI)!r}")
    print(f"  query('bat', 'has')  -> {query('bat', 'has', KB_MULTI)!r}")
    print("  Search order: breadth-first, parents in KB list order.")


if __name__ == "__main__":
    demo()