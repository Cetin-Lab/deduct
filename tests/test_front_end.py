"""Clustering must be invariant to rotation and strand; alignment must be invertible."""
import numpy as np
import pytest
import torch

from deduct.align import align_cluster, reconstruct_member
from deduct.common import reverse_complement
from deduct.group import cluster_column, frame_length, smooth_length

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_smooth_length_is_seven_smooth_and_sufficient():
    for need in (1000, 4001, 88080, 250000):
        n = smooth_length(need)
        assert n >= need
        rest = n
        for factor in (2, 3, 5, 7):
            while rest % factor == 0:
                rest //= factor
        assert rest == 1, f"{n} is not 7-smooth"


def test_frame_leaves_room_for_the_tiled_target():
    for length in (1000, 50000):
        assert frame_length(length) >= 2 * length


def _family(rng, length=4000, members=8):
    """A rotated, strand-flipped, lightly mutated family plus two unrelated sequences."""
    base = rng.integers(1, 5, length).astype(np.uint8)
    out = []
    for i in range(members):
        s = base.copy()
        s[rng.choice(length, length // 100, replace=False)] = rng.integers(1, 5, length // 100)
        s = np.roll(s, int(rng.integers(0, length)))
        if i % 3 == 0:
            s = reverse_complement(s)
        out.append(np.ascontiguousarray(s))
    out.append(rng.integers(1, 5, length).astype(np.uint8))
    out.append(rng.integers(1, 5, length // 3).astype(np.uint8))
    return out, members


def test_clustering_is_rotation_and_strand_invariant():
    rng = np.random.default_rng(0)
    sequences, members = _family(rng)
    result = cluster_column(sequences, device=DEVICE, accept=0.6, max_clusters=5)
    largest = max(result["clusters"], key=lambda c: len(c["members"]))
    assert len(largest["members"]) == members, "the whole family must land in one cluster"
    assert all(m["score"] >= 0.6 for m in largest["members"])


def test_unrelated_sequences_do_not_join():
    rng = np.random.default_rng(1)
    sequences, members = _family(rng)
    result = cluster_column(sequences, device=DEVICE, accept=0.6, max_clusters=5)
    largest = max(result["clusters"], key=lambda c: len(c["members"]))
    joined = {m["index"] for m in largest["members"]}
    assert joined.isdisjoint({members + 1, members + 2})


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alignment_is_invertible(seed):
    """Rotation, strand, insertions and deletions must all be undone exactly."""
    rng = np.random.default_rng(seed)
    import random
    pick = random.Random(seed)
    base = rng.integers(1, 5, 12000).astype(np.uint8)
    sequences = []
    for i in range(24):
        s = base.copy()
        s[rng.choice(len(s), 40, replace=False)] = rng.integers(1, 5, 40)
        if i % 4 == 0:
            p = pick.randrange(100, len(s) - 100)
            s = (np.delete(s, slice(p, p + 5)) if i % 8 == 0
                 else np.insert(s, p, rng.integers(1, 5, 4)))
        s = np.roll(s, pick.randrange(len(s)))
        if i % 3 == 0:
            s = reverse_complement(s)
        sequences.append(np.ascontiguousarray(s.astype(np.uint8)))

    grouping = cluster_column(sequences, device=DEVICE, accept=0.6, max_clusters=3)
    cluster = max(grouping["clusters"], key=lambda c: len(c["members"]))
    aligned = align_cluster(sequences, cluster["members"], workers=4, device=DEVICE)

    for row, meta in enumerate(aligned["rows"]):
        back = reconstruct_member(meta["offset"], meta["strand"], aligned["matrix"][row],
                                  meta["insertions"])
        assert np.array_equal(back, sequences[meta["index"] - 1])
