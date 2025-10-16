# graph_enrichment.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import json, re

# If you want to use the LLM extractor, keep these two lines.
# If you don't have OPENAI set yet, the function will still run & return empty.
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@dataclass
class Triple:
    e1: str
    rel: str
    e2: str
    confidence: float
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    meta: Optional[Dict] = None

def normalize_entity(name: str) -> str:
    return " ".join((name or "").strip().lower().split())

def _safe_json_array(text: str):
    """Try strict JSON first, then extract the first [...] block if needed."""
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return []
    return []

def extract_triples(chunks: List[Dict]) -> List[Triple]:
    """
    Extract (e1, rel, e2, confidence) from text chunks using an LLM.
    Returns [] if no OPENAI_API_KEY or on failure.
    """
    triples: List[Triple] = []
    if not client:
        # No key configured; skip quietly
        return triples

    system = (
        "You extract concise relationship triples from text. "
        "Output ONLY a JSON object with a 'triples' array. Each item: "
        '{"e1":"<entity1>","rel":"<RELATION_UPPERCASE>","e2":"<entity2>","confidence":0.0-1.0}.'
    )

    for ch in chunks:
        snippet = (ch.get("text") or "")[:1200]
        user = f"""Text:
\"\"\"{snippet}\"\"\"

Return ONLY:
{{"triples":[{{"e1":"...","rel":"...","e2":"...","confidence":0.8}}]}}
Use relations like: USES, PART_OF, CAUSES, RELATES_TO, PARTNERS_WITH, INVESTS_IN, DEVELOPS, INTEGRATES.
Uppercase the 'rel'. Include 1–5 items only.
"""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content.strip()
            obj = json.loads(raw)
            arr = obj.get("triples", []) if isinstance(obj, dict) else _safe_json_array(raw)
        except Exception:
            arr = []

        for t in (arr or []):
            try:
                e1 = normalize_entity(t.get("e1", ""))
                e2 = normalize_entity(t.get("e2", ""))
                rel = (t.get("rel", "") or "").upper().strip()
                conf = float(t.get("confidence", 0.7))
                if e1 and e2 and rel:
                    triples.append(
                        Triple(
                            e1=e1, rel=rel, e2=e2, confidence=conf,
                            doc_id=ch.get("doc_id"), chunk_id=ch.get("chunk_id")
                        )
                    )
            except Exception:
                continue

    print(f"[graph] extracted {len(triples)} triples")
    return triples

def upsert_triples(triples: List[Triple], driver) -> None:
    """
    Insert triples into Neo4j:
      (a:Entity {name})-[:REL {label, source_doc, chunk_id}]->(b:Entity)
    """
    if not triples or driver is None:
        return
    try:
        with driver.session() as session:
            for t in triples:
                session.run(
                    """
                    MERGE (a:Entity {name: $e1})
                    MERGE (b:Entity {name: $e2})
                    MERGE (a)-[r:REL {
                        label: $rel,
                        source_doc: $doc_id,
                        chunk_id: $chunk_id
                    }]->(b)
                    """,
                    parameters={
                        "e1": t.e1,
                        "e2": t.e2,
                        "rel": t.rel,
                        "doc_id": t.doc_id,
                        "chunk_id": t.chunk_id,
                    },
                )
    except Exception as e:
        print("⚠️ upsert_triples failed:", e)
