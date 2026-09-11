"""Assign source groups before deriving targets; merge perceptually similar images."""
from __future__ import annotations

from h3ce.cache.keys import digest


class _BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def near(self, value, radius=4):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield from node[1]
            pending.extend(child for edge, child in node[2].items()
                           if distance - radius <= edge <= distance + radius)


def assign_splits(records: list[dict], *, validation_fraction: float, seed: int):
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        parents[find(j)] = find(i)

    groups, exact, perceptual = {}, {}, _BKTree()
    for i, record in enumerate(records):
        for mapping, key in ((groups, record["source_group"]), (exact, record["sha256"])):
            if key in mapping:
                union(i, mapping[key])
            else:
                mapping[key] = i
        fingerprint = record.get("_dhash")
        if fingerprint is not None:
            for j in perceptual.near(fingerprint):
                union(i, j)
            perceptual.add(fingerprint, i)
    members = {}
    for i, record in enumerate(records):
        members.setdefault(find(i), set()).add(record["source_group"])
    for i, record in enumerate(records):
        source_group = "group:" + digest(sorted(members[find(i)]))
        fraction = int(digest([seed, source_group])[:16], 16) / (1 << 64)
        record["source_group"] = source_group
        record["split"] = "val" if fraction < validation_fraction else "train"
        record.pop("_dhash", None)
    return records
