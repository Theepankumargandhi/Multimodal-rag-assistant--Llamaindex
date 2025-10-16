# graph_queries.py
from dataclasses import dataclass
from typing import List
from neo4j import Driver
import re

@dataclass
class Fact:
    e1: str
    rel: str
    e2: str
    source_doc: str
    chunk_id: str

def _candidate_names(q: str, top_k: int) -> List[str]:
    # keep words with letters/numbers, drop punctuation, lowercase
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]*", q or "")
    # filter short/common words
    stop = {"how","why","the","a","an","and","or","to","of","in","is","are","between","connected","relationship"}
    names = []
    for t in toks:
        tl = t.lower()
        if len(tl) >= 3 and tl not in stop:
            names.append(tl)
    # de-dupe preserve order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:top_k] if top_k else out

def find_relational_subgraph(query_text: str, driver: Driver, max_hops: int = 2, top_entities: int = 5) -> List[Fact]:
    """
    Case-insensitive lookup of relations near candidate names from the query.
    """
    names = _candidate_names(query_text, top_entities)
    if not names:
        return []

    with driver.session() as session:
        # Case-insensitive match using toLower(a.name)
        result = session.run(
            """
            MATCH (a:Entity)-[r:REL]->(b:Entity)
            WHERE toLower(a.name) IN $names OR toLower(b.name) IN $names
            RETURN a.name AS e1, r.label AS rel, b.name AS e2,
                   coalesce(r.source_doc,'unknown') AS source_doc,
                   coalesce(r.chunk_id,'unknown') AS chunk_id
            LIMIT 50
            """,
            parameters={"names": names},
        )
        facts = [Fact(**record) for record in result]
        # de-dup
        uniq = {(f.e1, f.rel, f.e2, f.source_doc, f.chunk_id): f for f in facts}
        facts = list(uniq.values())

    # optional: sort for stable display
    facts.sort(key=lambda x: (x.source_doc or "", x.e1, x.rel, x.e2))
    print(f"[graph] found {len(facts)} facts for names={names}")
    return facts

def format_facts_for_llm(facts: List[Fact]) -> str:
    if not facts:
        return ""
    lines = ["Graph Context:"]
    for f in facts:
        lines.append(f" - {f.e1} {f.rel} {f.e2} (doc: {f.source_doc}, chunk: {f.chunk_id})")
    return "\n".join(lines)
