#!/usr/bin/env python3
"""Utils for join graph metadata and handling."""

import collections
import re

import hashlib
import networkx as nx

# If join_graph is null, it will be deduced from join_keys.
JoinSpec = collections.namedtuple("JoinSpec", [
    "join_tables", "join_keys", "join_clauses", "join_graph", "join_tree",
    "join_root", "join_how", "join_name"
])


def match_join_clause_or_fail(clause):
    m = re.match(r'(.*)\.(.*)=(.*)\.(.*)', clause)
    assert m, clause
    groups = m.groups()
    assert len(groups) == 4, (clause, groups)
    return m


def _make_join_graph(join_clauses, root):
    """Constructs nx.Graph and nx.DiGraph representations of the specified join.

    Constructs an undirected graph in which vertices are tables, and edges are
    joins.

    Also returns an arborescence (directed tree) in which edges originate from
    the join root to every joined table.
    """
    clauses = []
    for line in join_clauses:
        groups = match_join_clause_or_fail(line).groups()
        clauses.append(groups)
    g = nx.Graph()
    for t1, c1, t2, c2 in clauses:
        assert not g.has_edge(t1, t2)
        g.add_edge(t1, t2, join_keys={t1: c1, t2: c2})
    assert nx.is_tree(g), g.edges

    paths = nx.single_source_shortest_path(g, root)
    dg = nx.DiGraph()
    for path in paths.values():
        prev = None
        for t in path:
            if prev is not None:
                dg.add_edge(prev, t)
            prev = t
    assert set(g.nodes) == set(dg.nodes)
    return g, dg


def get_bottom_up_table_ordering(join_spec):
    """
    Returns a reversed BFS traversal for bottom-up join counts calculation.
    """
    root = join_spec.join_root
    edges = list(nx.bfs_edges(join_spec.join_tree, root))
    nodes = [root] + [v for _, v in edges]
    return reversed(nodes)


def _infer_join_clauses(tables, join_keys, t0):
    """For backward compatibility with single equivalence class joins."""
    for keys in join_keys.values():
        assert len(keys) == 1, join_keys
    assert t0 in tables, tables
    t0_idx = tables.index(t0)
    k0 = join_keys[t0][0]
    return [
        "{}.{}={}.{}".format(t0, k0, t, join_keys[t][0])
        for i, t in enumerate(tables)
        if i != t0_idx
    ]


def get_join_spec(config):
    join_clauses = config.get("join_clauses")
    if join_clauses is None:
        join_clauses = _infer_join_clauses(config["join_tables"],
                                           config["join_keys"],
                                           config["join_root"])
    g, dg = _make_join_graph(join_clauses, config["join_root"])
    join_hash = hashlib.sha1(
        str([join_clauses, config["join_root"],
             config["join_how"]]).encode()).hexdigest()[:8]
    return JoinSpec(
        join_tables=config["join_tables"],
        join_keys=config["join_keys"],
        join_clauses=join_clauses,
        join_graph=g,
        join_tree=dg,
        join_root=config["join_root"],
        join_how=config["join_how"],
        join_name="{}-{}".format(config.get("join_name"), join_hash),
    )
