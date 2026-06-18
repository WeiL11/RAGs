"""Knowledge graph over the podcast, backed by networkx + a JSON file.

Nodes are entities (companies/tickers/people/concepts). Each node keeps a list of
*mentions* (the chunk text + episode + timestamps where it appeared) — these are the
provenance used to build ``RetrievedContext`` at query time. Edges are relations
between entities, each carrying the episode + evidence text it was extracted from.

networkx + JSON keeps the footprint tiny for the test corpus; swap for Neo4j if
traversal needs grow (the strategy only depends on the methods below).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.rag.base import RetrievedContext


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip()).lower()


class GraphStore:
    def __init__(self, path: str | Path) -> None:
        import networkx as nx  # lazy

        self.path = Path(path)
        self._g = nx.MultiDiGraph()

    # --- mutation (ingest-time) ------------------------------------------------

    def add_entity(self, name: str, etype: str, mention: dict[str, Any] | None = None) -> str:
        key = normalize_name(name)
        if not key:
            return ""
        if not self._g.has_node(key):
            self._g.add_node(key, name=name, type=etype, mentions=[])
        if mention:
            self._g.nodes[key]["mentions"].append(mention)
        return key

    def add_relation(
        self, subject: str, relation: str, obj: str, episode_id: str, evidence: str
    ) -> None:
        s, o = normalize_name(subject), normalize_name(obj)
        if not s or not o or not self._g.has_node(s) or not self._g.has_node(o):
            return
        self._g.add_edge(s, o, relation=relation, episode_id=episode_id, evidence=evidence)

    # --- query-time ------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._g.number_of_edges()

    def link_entities(self, query: str, terms: list[str]) -> list[str]:
        """Return node keys that match the query by substring or shared term."""
        matched: list[str] = []
        term_keys = {normalize_name(t) for t in terms if t.strip()}
        for key, data in self._g.nodes(data=True):
            name = data.get("name", "")
            if key and key in normalize_name(query):
                matched.append(key)
            elif normalize_name(name) in term_keys or key in term_keys:
                matched.append(key)
        return list(dict.fromkeys(matched))

    def neighbors(self, key: str, hops: int = 1) -> set[str]:
        import networkx as nx  # lazy

        if not self._g.has_node(key):
            return set()
        und = self._g.to_undirected(as_view=True)
        return set(nx.single_source_shortest_path_length(und, key, cutoff=hops)) - {key}

    def relation_triples(self, keys: set[str]) -> list[str]:
        """Human-readable triples among the given nodes, e.g. '輝達 —供應商→ 台積電 (EP512)'."""
        out: list[str] = []
        for u, v, d in self._g.edges(data=True):
            if u in keys or v in keys:
                un = self._g.nodes[u].get("name", u)
                vn = self._g.nodes[v].get("name", v)
                out.append(f"{un} —{d.get('relation','related')}→ {vn} ({d.get('episode_id','')})")
        return list(dict.fromkeys(out))

    def mentions_as_contexts(self, keys: set[str], limit: int = 12) -> list[RetrievedContext]:
        ctx: list[RetrievedContext] = []
        for key in keys:
            if not self._g.has_node(key):
                continue
            for m in self._g.nodes[key].get("mentions", []):
                ctx.append(
                    RetrievedContext(
                        text=m.get("text", ""),
                        episode_id=m.get("episode_id", ""),
                        ep_number=int(m.get("ep_number", 0)),
                        publish_date=m.get("publish_date"),
                        start_s=m.get("start_s"),
                        end_s=m.get("end_s"),
                        score=1.0,
                        source="graph",
                    )
                )
        # de-dupe by (episode, start)
        seen: dict[tuple[str, float], RetrievedContext] = {}
        for c in ctx:
            seen.setdefault((c.episode_id, c.start_s or 0.0), c)
        return list(seen.values())[:limit]

    # --- persistence -----------------------------------------------------------

    def save(self) -> None:
        import networkx as nx  # lazy

        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._g, edges="links")
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self) -> "GraphStore":
        import networkx as nx  # lazy

        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._g = nx.node_link_graph(data, multigraph=True, directed=True, edges="links")
        return self
