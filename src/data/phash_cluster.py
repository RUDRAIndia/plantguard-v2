"""Generic near-duplicate clustering by perceptual hash. No PlantVillage
knowledge, no file I/O — this is the reusable matching engine that
src/data/dedupe.py orchestrates against the actual dataset.

Finding every pair of items within a Hamming-distance threshold by brute
force is O(n^2): infeasible at PlantVillage's scale (~54k images -> ~1.5
billion pairs). Instead this uses banding (an LSH technique): split each
hash's bits into `threshold + 1` contiguous bands. By the pigeonhole
principle, if two hashes differ in at most `threshold` bits, at least one
band must be identical between them (if every band differed by >= 1 bit,
the total difference would be >= threshold + 1, a contradiction). So
bucketing items by each band's bit-pattern and only comparing items that
land in the same bucket is guaranteed not to miss any true match — it can
only produce extra candidates, which are then verified by exact Hamming
distance before being accepted.
"""

from collections import defaultdict
from typing import Hashable


class _UnionFind:
    """Minimal disjoint-set with path compression and union by size."""

    def __init__(self, items):
        self._parent = {item: item for item in items}
        self._size = {item: 1 for item in items}

    def find(self, item):
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a, b) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self._size[root_a] < self._size[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        self._size[root_a] += self._size[root_b]


def _band_slices(num_bits: int, num_bands: int) -> list[tuple[int, int]]:
    """Splits [0, num_bits) into num_bands contiguous slices, as equal in
    size as possible.
    """
    base, remainder = divmod(num_bits, num_bands)
    slices = []
    start = 0
    for i in range(num_bands):
        size = base + (1 if i < remainder else 0)
        slices.append((start, start + size))
        start += size
    return slices


def _candidate_pairs(flat_bits: dict[str, tuple], threshold: int) -> set[tuple[str, str]]:
    """Returns id pairs (a, b), a < b, guaranteed to include every pair
    whose hashes differ by <= threshold bits (may also include some pairs
    that don't — those are filtered by the caller via exact distance).
    """
    num_bands = threshold + 1
    num_bits = len(next(iter(flat_bits.values())))
    candidates: set[tuple[str, str]] = set()

    for start, end in _band_slices(num_bits, num_bands):
        buckets: dict[tuple, list[str]] = defaultdict(list)
        for item_id in sorted(flat_bits):
            buckets[flat_bits[item_id][start:end]].append(item_id)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    candidates.add((members[i], members[j]))

    return candidates


def cluster(
    items: dict[str, tuple[Hashable, object]], threshold: int
) -> tuple[dict[str, str], list[dict]]:
    """Clusters `items` (id -> (boundary_key, image_hash)) by perceptual
    hash similarity.

    Two items are merged into the same cluster only if their hash Hamming
    distance is <= threshold AND they share the same boundary_key (e.g. a
    class name) — a match across different boundary_keys is never merged,
    it is instead reported as a collision. This is what guarantees clusters
    never span a boundary: structurally, by construction, not by filtering
    results after the fact.

    Returns (id_to_group_id, collisions):
      - id_to_group_id: every input id maps to a group id, which is the
        lexicographically smallest id among all ids merged with it
        (deterministic; a singleton item is its own group id).
      - collisions: one dict per same-hash-neighborhood, different-
        boundary_key pair found: {"item_a", "boundary_a", "item_b",
        "boundary_b", "hamming_distance"}.
    """
    flat_bits = {item_id: tuple(bits) for item_id, (_, bits) in _flatten_hashes(items)}
    uf = _UnionFind(items.keys())
    collisions = []

    for a, b in sorted(_candidate_pairs(flat_bits, threshold)):
        boundary_a, hash_a = items[a]
        boundary_b, hash_b = items[b]
        distance = hash_a - hash_b
        if distance > threshold:
            continue
        if boundary_a == boundary_b:
            uf.union(a, b)
        else:
            collisions.append(
                {
                    "item_a": a,
                    "boundary_a": boundary_a,
                    "item_b": b,
                    "boundary_b": boundary_b,
                    "hamming_distance": int(distance),
                }
            )

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for item_id in items:
        members_by_root[uf.find(item_id)].append(item_id)

    id_to_group_id = {}
    for members in members_by_root.values():
        group_id = min(members)
        for item_id in members:
            id_to_group_id[item_id] = group_id

    return id_to_group_id, collisions


def _flatten_hashes(items: dict[str, tuple[Hashable, object]]):
    """Yields (item_id, (boundary_key, flat_bit_tuple)) — flattens each
    image_hash's underlying bit matrix into a 1D tuple of booleans so it can
    be sliced into bands.
    """
    for item_id, (boundary_key, image_hash) in items.items():
        yield item_id, (boundary_key, tuple(image_hash.hash.flatten().tolist()))
