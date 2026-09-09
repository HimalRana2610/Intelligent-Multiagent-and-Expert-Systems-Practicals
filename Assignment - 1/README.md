# Experiment 1 — Knowledge Representation: Semantic Network with Inheritance

**Intelligent Multi-Agent and Expert Systems (AI401)** · B.Tech IV (AI), Semester VII
Run with `python main.py` (standard library only, instant).

---

## Aim

To represent knowledge as a semantic network of (subject, relation, object) triples and answer queries using property inheritance along `is-a` links, including exceptions.

## Theory

Before an expert system can reason, knowledge has to be encoded in a form a machine can walk over. A **semantic network** does this with concepts as nodes and relationships as labelled, directed edges. One relation is special: `is-a` organises the concepts into a taxonomy, and **property inheritance** lets a specific concept acquire the properties of its ancestors. A shark has skin not because anyone said so, but because `shark is-a fish is-a animal` and animals have skin.

Storing a property once at the most general concept that has it is the whole economy of the representation:

| | facts stored | facts answerable |
|---|---|---|
| flat table | one row per (concept, property) pair | only what is stored |
| semantic network | one edge at the most general concept | every descendant, by walking `is-a` |

The catch is that real taxonomies have **exceptions**. Birds fly, but penguins do not. If inheritance were absolute the network would have to abandon the useful default `bird can fly` and re-state flight on every flying bird individually. Instead the query procedure searches **from the most specific node upward and lets local knowledge win** — the property found first, at the lowest level of the chain, is the answer, and anything more general is never consulted.

That single design decision is what makes the representation a **default logic** rather than a classical one, and it is what the later experiments inherit. The frame systems of Exercise 2, the production rules of Experiment 2, and the chaining engines of Experiments 4 and 5 all reason over knowledge encoded in exactly this declarative style.

## Algorithm

1. Store the network as a list of `(subject, relation, object)` triples.
2. To find the inheritance chain of a node, repeatedly follow its `is-a` links until no parent exists.
3. To answer `query(node, relation)`, walk up the chain; at each level check for an exception (`cannot`) and then for a local value; return the first value found.
4. Report `unknown` if the chain is exhausted.

---

## The code

Everything is in [`main.py`](main.py).

### Step 1 — the network as triples

```python
KB = [
    ("animal",  "has",    "skin"),
    ("fish",    "is-a",   "animal"),
    ("shark",   "is-a",   "fish"),
    ("bird",    "is-a",   "animal"),
    ("bird",    "can",    "fly"),
    ("bird",    "has",    "wings"),
    ("penguin", "is-a",   "bird"),
    ("penguin", "cannot", "fly"),
    ...
]
```

A flat list of triples is deliberately the *least* structured choice available. Nothing in the data distinguishes `is-a` from `can` or `has` — the taxonomy exists only because `get_parents` chooses to follow `is-a` edges and nothing else. The same list would support a different traversal (part-of hierarchies, causal links) without changing a byte of the data, which is the point of a declarative representation.

### Step 2 — the inheritance chain

```python
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
```

`queue.pop(0)` plus `queue.extend(...)` is a FIFO, so this is a **breadth-first** walk up the taxonomy: the node itself, then all its parents, then all their parents. With one `is-a` link per concept that is just a straight line; it starts to matter in Exercise 3, and the justification for choosing it is given there.

`visited` is not decoration. A cycle in the `is-a` edges — easy to introduce by accident in a hand-written knowledge base — would otherwise spin the loop forever.

### Step 3 — the query, exceptions first

```python
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
```

Two orderings are doing the work, and they are independent:

- **Across levels** — the loop runs most-specific-first, so `penguin` is inspected before `bird`, which is inspected before `animal`. This is what lets a subclass override a superclass.
- **Within one level** — the `cannot` test runs *before* the positive lookup. This is what lets a concept override a property it would otherwise state itself.

Both `return` statements are inside the loop, so the search stops at the first level that says anything at all. The higher levels are never even read.

### Step 4 — the base case

Falling out of the loop means the chain was exhausted without a match, and the answer is `unknown` — an explicit third value, not `False`. The network makes no closed-world assumption here: *not recorded* is reported as *not known*, which is the right reading for a knowledge base that is admittedly incomplete.

---

## Procedure and Result

`python main.py` prints the knowledge base, the chains, and every query.

### The knowledge base

```
  ('animal', 'has', 'skin')
  ('fish', 'is-a', 'animal')
  ('fish', 'can', 'swim')
  ('shark', 'is-a', 'fish')
  ('bird', 'is-a', 'animal')
  ('bird', 'can', 'fly')
  ('bird', 'has', 'wings')
  ('penguin', 'is-a', 'bird')
  ('penguin', 'cannot', 'fly')
  ('penguin', 'can', 'swim')
  ('ostrich', 'is-a', 'bird')
  ('ostrich', 'cannot', 'fly')
  ('ostrich', 'can', 'run_fast')
```

### Inheritance chains (Algorithm step 2)

```
  shark:   shark -> fish -> animal
  penguin: penguin -> bird -> animal
  ostrich: ostrich -> bird -> animal
```

### Queries (Algorithm step 3)

```
  query('shark', 'has')   -> 'skin'
  query('shark', 'can')   -> 'swim'
  query('penguin', 'can') -> 'cannot fly'
  query('penguin', 'has') -> 'wings'
  query('ostrich', 'can') -> 'cannot fly'
  query('ostrich', 'has') -> 'wings'
  query('bird', 'can')    -> 'fly'
```

Each answer comes from a different level of the chain, which is the result worth reading carefully:

| query | answered at | how far up | why |
|---|---|---|---|
| `shark has` | `animal` | 2 levels | nothing local; `fish` silent; `animal has skin` |
| `shark can` | `fish` | 1 level | `fish can swim` beats going further |
| `penguin can` | `penguin` | 0 levels | local `cannot fly` — the exception |
| `penguin has` | `bird` | 1 level | `bird has wings` |
| `bird can` | `bird` | 0 levels | the default itself |

`shark has skin` is inheritance across two links from a fact stated exactly once. `penguin can → cannot fly` is the exception overriding that same mechanism.

---

## Observations

### Why `penguin can` returns `cannot fly` even though birds can fly

Both facts are in the network and both are reachable, so the answer is decided entirely by **which one the search meets first**. The chain is `penguin → bird → animal`, and the loop begins at `penguin`. On that very first iteration `local_facts("penguin", "cannot")` returns `["fly"]`, the function returns `"cannot fly"`, and `bird` is never examined. The default `bird can fly` is not deleted, contradicted, or repaired — it is simply **shadowed** for this one descendant, and it remains fully in force for every other bird. Adding the penguin exception cost one triple and left the rest of the taxonomy untouched; that economy is the reason to represent knowledge this way.

### Why exceptions must be checked before inherited values *at every level*

If the exception test were moved outside the loop — check `node` for `cannot`, then walk the chain for values — it would only catch exceptions stated on the queried node itself. Any concept that inherits an exception from an intermediate ancestor would lose it. In a chain `emperor_penguin → penguin → bird`, `query("emperor_penguin", "can")` must still answer `cannot fly`: the exception lives at `penguin`, one level up, and only a per-level test finds it there. Placing the check inside the loop makes the rule uniform — *at each level, local knowledge beats inherited knowledge* — and it is that uniformity, rather than any special status given to the starting node, that makes the mechanism compose over chains of arbitrary depth.

The ordering also has to be strict *within* a level. If the positive lookup ran first, then at `penguin` the loop would find `penguin can swim` and return `swim`, and `cannot fly` would never be consulted at all. Exceptions are only exceptions if they are consulted first.

### What this says about default reasoning being non-monotonic

Classical logic is **monotonic**: adding premises can only ever add conclusions, never withdraw one. This network is not.

Add a sparrow that states nothing of its own, so its only source of a `can` value is the default it inherits:

```python
KB_S = KB + [("sparrow", "is-a", "bird")]
query("sparrow", "can", KB_S)                                  # -> 'fly'
```

Now add **one** more triple, `("sparrow", "cannot", "fly")` — strictly more knowledge, nothing removed — and re-ask:

```python
query("sparrow", "can", KB_S + [("sparrow", "cannot", "fly")])  # -> 'cannot fly'
```

A conclusion that was derivable has become underivable because knowledge *increased*. That is non-monotonicity, and it is not a bug in the implementation: it is the defining behaviour of default reasoning. `bird can fly` was never a universally quantified claim about all birds; it was a **default**, asserted to hold in the absence of information to the contrary, and the new triple supplies exactly that contrary information.

Two consequences follow, and both matter for the rest of the lab:

- **Conclusions are provisional.** Anything derived by inheritance is only as good as the current absence of an exception. A real system that had already *acted* on `sparrow can fly` would need truth maintenance to retract whatever it built on that belief.
- **The order of the search is part of the semantics.** In a monotonic system, the order in which you consult the axioms cannot change what follows from them. Here it decides the answer outright — which is precisely why Exercise 3 has to *justify* a traversal order rather than just pick one.

Experiment 4 sits at the opposite pole: Modus Ponens and Modus Tollens are monotonic, so a fact derived there is never withdrawn, and deriving both `p` and `¬p` signals an inconsistent knowledge base rather than a revisable default.

### A limitation this implementation makes visible

```
query('ostrich', 'can')  ->  'cannot fly'
```

An ostrich can also run fast, and `("ostrich", "can", "run_fast")` is in the network — but the answer never mentions it. The `cannot` test at the `ostrich` level fires and returns before the local `can` lookup is reached, so **one exception shadows every positive value at the same level**.

This is a consequence of treating a relation as single-valued: `query(node, relation)` asks *"what can an ostrich do?"* and can only return one thing, so the exception and the ability compete for the same answer slot. The question the representation actually supports well is the **yes/no** one — *"can an ostrich fly?"* — where the exception applies to the value `fly` and leaves `run_fast` alone:

```python
def can(node, value, kb=KB):
    for concept in get_chain(node, kb):
        if value in local_facts(concept, "cannot", kb):
            return False
        if value in local_facts(concept, "can", kb):
            return True
    return None                       # unknown
```

Under that reading `can("ostrich", "fly")` is `False` and `can("ostrich", "run_fast")` is `True`, both correct. The lesson is that in an inheritance network an exception is properly indexed by **(relation, value)**, not by relation alone.

---

## Exercise 1 — Extending the network with an ostrich

Three triples were added to `KB`:

```python
("ostrich", "is-a",   "bird"),
("ostrich", "cannot", "fly"),
("ostrich", "can",    "run_fast"),
```

and the queries verify:

```
  chain:                    ostrich -> bird -> animal
  query('ostrich', 'can') -> 'cannot fly'      # local exception, level 0
  query('ostrich', 'has') -> 'wings'           # inherited from bird, level 1
```

The point of the exercise is that **nothing in the query code changed**. The ostrich behaves exactly like the penguin — same override, same inheritance — because the mechanism is in the traversal, not in the data. `has wings` is still inherited even though flight, the reason wings exist, has been overridden; defaults are independent of one another, and overriding one does not disturb its neighbours.

## Exercise 2 — The same knowledge as frames

A frame system stores the same graph *concept-first* instead of *edge-first*: one dictionary per concept, its keys are **slots**, its values **fillers**, and the `ako` (a-kind-of) slot plays the role of `is-a`.

```python
FRAMES = {
    "animal":  {"has": "skin"},
    "fish":    {"ako": "animal", "can": "swim"},
    "shark":   {"ako": "fish"},
    "bird":    {"ako": "animal", "can": "fly", "has": "wings"},
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
```

### Result — identical answers

```
  frame_query('shark', 'has')   -> 'skin'
  frame_query('shark', 'can')   -> 'swim'
  frame_query('penguin', 'can') -> 'cannot fly'
  frame_query('penguin', 'has') -> 'wings'
  frame_query('ostrich', 'can') -> 'cannot fly'
  frame_query('ostrich', 'has') -> 'wings'
  frame_query('bird', 'can')    -> 'fly'
```

Every answer matches the triple store, line for line. That is the result: **the two notations are the same knowledge**, and the inheritance-with-exceptions behaviour lives in the lookup procedure, which is structurally the same in both (`frame_query` and `query` differ only in how they fetch a local value).

### What actually differs

| | triples | frames |
|---|---|---|
| lookup of one property | scan the whole list, `O(\|KB\|)` | one dict index, `O(1)` |
| all properties of a concept | scan and filter | already grouped — the frame *is* the answer |
| all concepts with a property | one scan | visit every frame |
| multiple parents | free — just add another `is-a` triple | needs `ako` to become a list |
| multiple values per slot | free — add another triple | needs the filler to become a list |

Triples are a **relation-centred** view and frames a **concept-centred** one. Frames win on the operation an expert system performs constantly — *give me everything known about this object* — which is why production systems and object-oriented knowledge bases (and, historically, KEE, KRL and CLIPS templates) settled on frames. The price is visible in the last two rows: the frame form here is rigid, allowing exactly one `ako` and exactly one filler per slot, while the triple form allows both for free.

The frame notation also has room the triple notation does not: real frames attach **facets** to a slot — a default filler, a type restriction, a range, an `if-needed` procedure to compute the value on demand, an `if-added` daemon to fire when it is filled. `{"can": "fly"}` is a bare filler; `{"can": {"default": "fly", "if-needed": compute_flight}}` is where the representation is actually going, and procedural attachment of that kind has no natural home in a flat triple list.

## Exercise 3 — Multiple parents, and the order to search them

Allowing several `is-a` links per node needs no change at all to `get_parents` or `get_chain` — both already return and enqueue *lists*. Only the data changes:

```python
KB_MULTI = KB + [
    ("bat",    "is-a", "mammal"),
    ("bat",    "is-a", "flyer"),
    ("mammal", "is-a", "animal"),
    ("mammal", "has",  "fur"),
    ("flyer",  "can",  "fly"),
]
```

### Result

```
  chain: bat -> mammal -> flyer -> animal
  query('bat', 'can')  -> 'fly'      # from flyer
  query('bat', 'has')  -> 'fur'      # from mammal
```

`bat` now inherits from two unrelated ancestors at once, and the answer to `can` comes from one of them while the answer to `has` comes from the other.

### The order, and why breadth-first

`get_chain` visits **breadth-first, parents in the order the `is-a` triples appear in the knowledge base**. For `bat` that gives `bat → mammal → flyer → animal`. A depth-first walk of the same graph would give `bat → mammal → animal → flyer` — it exhausts the first parent's whole ancestry before looking at the second.

The two orders are not interchangeable. Add one triple, `("animal", "can", "move")`, and they disagree:

| traversal | chain | `query('bat', 'can')` |
|---|---|---|
| breadth-first (used here) | `bat → mammal → flyer → animal` | **`fly`** — from `flyer` |
| depth-first | `bat → mammal → animal → flyer` | `move` — from `animal` |

Depth-first answers `move`, inheriting from `animal` — the **most general concept in the network** — while ignoring `flyer`, a direct parent of `bat` that has a more specific answer. That is the wrong result on the principle the whole experiment rests on: *more specific knowledge must win*. Depth-first breaks it because a long first branch drags the search up past the root before the second branch is ever considered, so distance from the queried node stops governing precedence.

Breadth-first is the right order because it visits ancestors in order of **inferential distance** — number of `is-a` links from the queried node. Every parent is consulted before any grandparent, so specificity is preserved by construction, and the guarantee holds no matter how uneven the branches are. (This is the same reasoning behind C3 linearisation in modern class hierarchies, which additionally guarantees that a class always precedes its own ancestors even when the graph is irregular. Python's own MRO changed from a depth-first scheme to C3 in version 2.3 for exactly the failure shown above.)

### Where order alone is not enough

Breadth-first fixes precedence *by level*. It does not decide **ties within a level**, and that is a genuine ambiguity rather than a missing feature. Suppose mammals do not fly:

```python
KB_AMBIG = KB_MULTI + [("mammal", "cannot", "fly")]
query("bat", "can", KB_AMBIG)   ->  'cannot fly'
```

`mammal` and `flyer` are both direct parents of `bat`, both at distance 1, and they say opposite things. The answer `cannot fly` is not derived from any principle — it is decided by `mammal` appearing before `flyer` in the triple list. Reordering two lines of the knowledge base silently reverses the conclusion, and the answer for a bat is wrong either way.

This is the classic **multiple-inheritance conflict** (the "Nixon diamond" of the non-monotonic reasoning literature: a Quaker is a pacifist, a Republican is not, Nixon is both). Textual order is a *tie-break*, not a justification, and the three honest ways out are:

1. **Report the ambiguity** rather than resolve it — return both candidate answers and let the caller or the user decide. Sceptical inheritance takes this line.
2. **Give the edges explicit priority** — attach a weight or a salience to each `is-a` link so the knowledge engineer states which parent dominates, instead of encoding it accidentally in file order.
3. **Assert the exception at the specific node** — add `("bat", "can", "fly")` directly. The bat's own knowledge is at distance 0 and beats both parents, so the conflict never arises. This is the same override that solved the penguin, applied one level down.

Option 3 is what a working knowledge base does most of the time, and it is worth noticing that it is available *only* because the search is specificity-ordered: the fix works precisely because level 0 is guaranteed to be read before level 1.

---

## Viva questions

**1. What is a semantic network, and what does the `is-a` link contribute?**
A directed labelled graph in which nodes are concepts and edges are binary relations, stored here as `(subject, relation, object)` triples. `is-a` is distinguished only by the traversal: it is the edge the query procedure follows upward, so it induces the taxonomy along which properties are inherited. That lets a property be stated once at the most general concept that has it (`animal has skin`) and be answerable for every descendant (`shark has skin`) without being stored again.

**2. How are exceptions handled, and why is the search direction critical?**
By searching most-specific-first and returning the first value found, with the `cannot` test placed before the positive lookup at each level. The specific fact is met before the general default and shadows it. Reverse the direction and the general default would always be found first, making exceptions unrepresentable; move the exception test outside the loop and exceptions inherited from intermediate ancestors would be missed.

**3. Why is this non-monotonic, and what does that cost?**
Because adding a triple can remove a conclusion: for a `sparrow` that is-a `bird` and states nothing of its own, `query('sparrow','can')` answers `fly` before `("sparrow","cannot","fly")` is asserted and `cannot fly` after. Inherited properties are defaults, valid only while no more specific fact contradicts them. The cost is that conclusions are provisional — a system that has acted on an inherited default may have to retract that work when an exception arrives (truth maintenance) — and that the traversal order becomes part of the meaning of the knowledge base rather than an implementation detail.

**4. Frames or triples?**
The same knowledge, and identical answers (Exercise 2 verifies this line for line). Triples are relation-centred: cheap to add an edge, cheap to ask which concepts have a property, but every lookup is a scan. Frames are concept-centred: `O(1)` slot access and everything about one object grouped together, which is the query an expert system makes constantly, plus room for facets — defaults, type restrictions, `if-needed` and `if-added` procedural attachments — that a flat triple has nowhere to put.

**5. With multiple parents, which is searched first?**
Breadth-first, ties broken by the order of the `is-a` triples. Breadth-first orders ancestors by inferential distance, so every parent is consulted before any grandparent and specificity is preserved; depth-first can reach the root of one branch before the second parent and let the most general concept in the network override a direct parent. The textual tie-break between two parents at the same distance is arbitrary and the conflict is genuine — it should be reported, given explicit priorities, or pre-empted by asserting the property on the specific node.

---

## File map

| Item in `main.py` | Purpose |
|---|---|
| `KB` | The network as `(subject, relation, object)` triples (algorithm step 1) |
| `get_parents` | The `is-a` edges out of one node |
| `get_chain` | Breadth-first inheritance chain (step 2) |
| `local_facts` | Values stated *at* one concept, with no inheritance |
| `query` | Walk the chain, exception first, return the first value (steps 3–4) |
| `FRAMES`, `frame_chain`, `frame_query` | Exercise 2 — the same knowledge as frames with `ako` slots |
| `KB_MULTI` | Exercise 3 — `bat` with two parents |
| `demo` | Prints the knowledge base, the chains and every query |

## Files

| File | Contents |
|---|---|
| [`main.py`](main.py) | Full implementation and the demo driver |
| [`README.md`](README.md) | This write-up |
| `Assignment - 1.pdf` | The original handout |
| `LAB_1_U23AI053.docx` | The submitted report |

## Running

```
python main.py            # everything, instant, standard library only
```

From a REPL:

```python
from main import *

get_chain("penguin")                 # ['penguin', 'bird', 'animal']
query("penguin", "can")              # 'cannot fly'
frame_query("penguin", "can")        # 'cannot fly'  -- same answer, frames
query("bat", "can", KB_MULTI)        # 'fly'         -- multiple parents
```
