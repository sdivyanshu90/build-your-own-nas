"""Architecture lineage reconstruction.

Evolutionary search produces a forest: every mutated child records the parent it came
from and the mutation that produced it. Being able to walk that forest answers the
questions people actually ask after a search finishes — *where did the winner come
from?*, *which mutation caused the jump in accuracy?*, *did the population collapse
onto one ancestor?*

This module works on a minimal :class:`LineageNode` structure rather than on database
rows, so it has no dependency on the persistence layer and can be unit-tested with
plain data. The repository converts its rows into nodes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LineageNode:
    """One candidate's position in the ancestry graph.

    Attributes:
        candidate_id: Unique candidate identifier.
        architecture_hash: Canonical architecture hash.
        parent_id: Identifier of the parent candidate, or ``None`` for founders.
        mutation: Human-readable description of the mutation that produced this
            candidate from its parent.
        generation: Generation index assigned by the strategy, if any.
        objective_value: Primary objective value, used for annotation.
    """

    candidate_id: str
    architecture_hash: str
    parent_id: str | None = None
    mutation: str | None = None
    generation: int | None = None
    objective_value: float | None = None


@dataclass(frozen=True)
class LineageChain:
    """A root-to-leaf ancestry path.

    Attributes:
        nodes: Nodes ordered from the founding ancestor to the requested descendant.
        truncated: ``True`` when traversal stopped early because of a cycle or a
            missing parent record.
    """

    nodes: tuple[LineageNode, ...]
    truncated: bool = False

    @property
    def depth(self) -> int:
        """Number of nodes in the chain."""
        return len(self.nodes)

    def to_text(self) -> str:
        """Render the chain as an indented tree."""
        lines: list[str] = []
        for index, node in enumerate(self.nodes):
            prefix = "    " * index + ("└── " if index else "")
            mutation = f" [{node.mutation}]" if node.mutation else ""
            value = (
                f" (objective={node.objective_value:.4f})"
                if node.objective_value is not None
                else ""
            )
            lines.append(f"{prefix}{node.architecture_hash[:8]}{mutation}{value}")
        if self.truncated:
            lines.append("    (chain truncated: a parent record was missing or cyclic)")
        return "\n".join(lines)


@dataclass
class LineageGraph:
    """An indexed view over a set of :class:`LineageNode` records.

    Attributes:
        nodes: Nodes keyed by candidate id.
        children: Child ids keyed by parent id.
    """

    nodes: dict[str, LineageNode] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def from_nodes(cls, nodes: list[LineageNode]) -> LineageGraph:
        """Index a flat list of nodes.

        Args:
            nodes: Nodes in any order.

        Returns:
            An indexed graph.
        """
        graph = cls()
        for node in nodes:
            graph.nodes[node.candidate_id] = node
        for node in nodes:
            if node.parent_id is not None:
                graph.children[node.parent_id].append(node.candidate_id)
        for child_ids in graph.children.values():
            child_ids.sort()
        return graph

    def ancestry(self, candidate_id: str, *, max_depth: int = 1000) -> LineageChain:
        """Return the root-to-node ancestry chain for a candidate.

        Traversal is defensive: a candidate whose ``parent_id`` is missing from the
        graph, or a cycle introduced by corrupt data, terminates the walk and sets
        ``truncated`` rather than raising or looping forever.

        Args:
            candidate_id: Candidate whose ancestry is wanted.
            max_depth: Safety bound on chain length.

        Returns:
            The ancestry chain; empty when the candidate is unknown.
        """
        node = self.nodes.get(candidate_id)
        if node is None:
            return LineageChain(nodes=(), truncated=True)

        chain: list[LineageNode] = []
        seen: set[str] = set()
        truncated = False
        current: LineageNode | None = node
        while current is not None:
            if current.candidate_id in seen or len(chain) >= max_depth:
                truncated = True
                break
            seen.add(current.candidate_id)
            chain.append(current)
            if current.parent_id is None:
                break
            parent = self.nodes.get(current.parent_id)
            if parent is None:
                truncated = True
                break
            current = parent
        chain.reverse()
        return LineageChain(nodes=tuple(chain), truncated=truncated)

    def descendants(self, candidate_id: str) -> list[str]:
        """Return every descendant candidate id in breadth-first order.

        Args:
            candidate_id: Ancestor candidate.

        Returns:
            Descendant identifiers, nearest first.
        """
        result: list[str] = []
        queue = list(self.children.get(candidate_id, []))
        seen: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            result.append(current)
            queue.extend(self.children.get(current, []))
        return result

    def roots(self) -> list[str]:
        """Return candidate ids with no recorded parent, sorted for determinism."""
        return sorted(
            node.candidate_id
            for node in self.nodes.values()
            if node.parent_id is None or node.parent_id not in self.nodes
        )

    def statistics(self) -> dict[str, int]:
        """Return coarse structural statistics about the ancestry forest."""
        depths = [self.ancestry(node_id).depth for node_id in self.nodes]
        return {
            "nodes": len(self.nodes),
            "roots": len(self.roots()),
            "max_depth": max(depths) if depths else 0,
            "mutated_nodes": sum(1 for node in self.nodes.values() if node.parent_id),
        }


__all__ = ["LineageChain", "LineageGraph", "LineageNode"]
