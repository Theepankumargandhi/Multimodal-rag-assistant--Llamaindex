# Graph Schema (Neo4j)

**Node:** `(:Entity { name, type? })`  
**Edge:** `(:Entity)-[:REL { label, source_doc, chunk_id }]->(:Entity)`

**Relation whitelist:** USES, PART_OF, CAUSES, RELATES_TO, PARTNERS_WITH, INVESTS_IN, DEVELOPS, INTEGRATES

**Example:**
(OpenAI)-[:REL {label:"PARTNERS_WITH", source_doc:"news_01", chunk_id:"news_01::3"}]->(Microsoft)

**Sample 1–2 hop Cypher (conceptual):**
```cypher
MATCH (a:Entity)-[r:REL]->(b:Entity)
WHERE a.name IN $names OR b.name IN $names
RETURN a.name AS e1, r.label AS rel, b.name AS e2, r.source_doc AS doc, r.chunk_id AS chunk
LIMIT 25;
