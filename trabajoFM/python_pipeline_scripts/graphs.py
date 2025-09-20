from __future__ import annotations

import networkx as nx
from typing import Any

from .utils import get_logger


def build_network(nodes: list[Any], edges: list[tuple[Any, Any]]) -> nx.Graph:
    log = get_logger(__name__)
    log.info("Building network graph (stub)")
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g

