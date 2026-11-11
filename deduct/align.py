"""Turn a cluster into an aligned matrix: rows are members, columns are aligned positions.

Three steps. A consensus is seeded from the members that already have the cluster's modal
length, since those carry no indel relative to it and can be combined by rotation alone. Every
member is then rotated and strand-oriented against that consensus and aligned to it with an
indel-tolerant alignment, which places it in consensus coordinates with explicit gaps. Members
too far from the consensus to align usefully are not forced into the matrix; they are reported
as a tail for the fallback coder.

The result is a matrix over `{1,2,3,4}` for ACGT and `5` for a gap, plus, per row, the rotation,
strand, original length and any insertions the consensus has no column for. Those per-row items
are what make the alignment invertible, and `reconstruct_member` is the inverse.
"""
from __future__ import annotations

import re

import numpy as np

from .common import GAP, reverse_complement, to_bytes
from .group import _best_lag, _lay_out, _reference_spectrum, _transform_rows

CIGAR = re.compile(r"(\d+)([=XID])")

IDENTITY_FRACTION = 0.01
"""Mismatch fraction below which a member is taken as gap-free without running the aligner."""


def _fit(seq: np.ndarray, length: int) -> np.ndarray:
    """Truncate or circularly extend `seq` to exactly `length`."""
    if len(seq) == length:
        return seq
    if len(seq) > length:
        return seq[:length]
    return np.concatenate([seq, seq[:length - len(seq)]])


def _rotations(sequences, reference, n, device):
    """Best rotation of each sequence against `reference`, forward orientation only.

    The frame is `n = len(reference)` and the sequences have been fitted to it, so the frame is
    exactly the sequence and every rotation is representable without tiling.
    """
    spectrum = _reference_spectrum(reference, n, device)
    chunk = max(1, int(1.5e9 // (16 * max(n, 1))))
    out = []
    for start in range(0, len(sequences), chunk):
        codes, bound = _lay_out(sequences[start:start + chunk], n, rc=False, ref_len=None)
        spectra = _transform_rows(codes, n, device)
        import torch
        cross = (spectrum.unsqueeze(0) * spectra).sum(1)
        del spectra
        correlation = torch.fft.irfft(cross, n=n, dim=1)
        _peak, lag = _best_lag(correlation, torch.from_numpy(bound).to(device))
        out.extend(int(v) for v in lag.cpu().numpy())
    return out


def _rotations_and_strands(sequences, consensus, n, device):
    """Best `(rotation, strand)` of each sequence against `consensus`, trying both strands."""
    import torch
    spectrum = _reference_spectrum(consensus, n, device)
    chunk = max(1, int(1.5e9 // (16 * max(n, 1))))
    out = []
    for start in range(0, len(sequences), chunk):
        batch = sequences[start:start + chunk]
        results = []
        for rc in (False, True):
            codes, bound = _lay_out(batch, n, rc=rc, ref_len=None)
            spectra = _transform_rows(codes, n, device)
            cross = (spectrum.unsqueeze(0) * spectra).sum(1)
            del spectra
            correlation = torch.fft.irfft(cross, n=n, dim=1)
            peak, lag = _best_lag(correlation, torch.from_numpy(bound).to(device))
            results.append((peak.cpu().numpy(), lag.cpu().numpy()))
        (peak_f, lag_f), (peak_r, lag_r) = results
        for i in range(len(batch)):
            if peak_f[i] >= peak_r[i]:
                out.append((int(lag_f[i]), "F"))
            else:
                out.append((int(lag_r[i]), "R"))
    return out


def build_consensus(sequences, members, device: str = "cuda") -> tuple[np.ndarray, int]:
    """Seed a consensus from the members that already have the cluster's modal length.

    Those members differ from the consensus by substitutions and rotation only, so they can be
    combined by rotating each into a common frame and taking a per-column majority vote.

    Raises
    ------
    ValueError
        If no member has the modal length, in which case there is nothing to seed from.
    """
    lengths = np.array([len(sequences[m["index"] - 1]) for m in members])
    values, counts = np.unique(lengths, return_counts=True)
    modal = int(values[counts.argmax()])

    exact = [m for m in members if len(sequences[m["index"] - 1]) == modal]
    if not exact:
        raise ValueError("no member of modal length; cannot seed a consensus")

    oriented = [reverse_complement(sequences[m["index"] - 1]) if m["strand"] == "R"
                else sequences[m["index"] - 1] for m in exact]
    reference = oriented[0]
    lags = _rotations(oriented, reference, modal, device)

    counts = np.zeros((modal, 5), np.int32)
    columns = np.arange(modal)
    for seq, lag in zip(oriented, lags):
        counts[columns, np.roll(seq, -lag)] += 1
    return counts[:, 1:5].argmax(1).astype(np.uint8) + 1, modal


def align_member(member: np.ndarray, consensus: np.ndarray, modal: int,
                 lag: int, strand: str, consensus_text: bytes):
    """Place one member in consensus coordinates.

    Returns `(offset, strand, row, insertions, distance, oriented_length)`, where `row` has one
    entry per consensus column and `insertions` lists `(column, symbol)` for bases the consensus
    has no column for.

    A member that is flush with the consensus after rotation is taken as gap-free without
    running the aligner. Losslessness does not require an *optimal* alignment, only a valid one
    (a suboptimal choice costs residual entries and nothing else), and the Hamming distance is
    an upper bound on the edit distance, so the identity threshold downstream stays conservative.
    """
    import edlib
    oriented = reverse_complement(member) if strand == "R" else member
    offset = lag % len(oriented)
    rolled = np.roll(oriented, -offset)

    if len(rolled) == modal:
        mismatches = int(np.count_nonzero(rolled != consensus))
        if mismatches <= IDENTITY_FRACTION * modal:
            return offset, strand, rolled.astype(np.uint8), [], mismatches, len(oriented)

    result = edlib.align(to_bytes(rolled).decode("ascii"),
                         consensus_text.decode("ascii"), task="path")
    row = np.zeros(modal, np.uint8)
    insertions = []
    query = target = 0
    for count, op in CIGAR.findall(result["cigar"]):
        n = int(count)
        if op in "=X":
            row[target:target + n] = rolled[query:query + n]
            query += n
            target += n
        elif op == "D":
            row[target:target + n] = GAP
            target += n
        elif op == "I":
            insertions.extend((target, int(b)) for b in rolled[query:query + n])
            query += n
    return offset, strand, row, insertions, result["editDistance"], len(oriented)


def reconstruct_member(offset: int, strand: str, row, insertions) -> np.ndarray:
    """Inverse of `align_member`: recover the member's original sequence."""
    row = np.asarray(row, np.uint8)
    if not insertions:
        oriented = np.roll(row[row != GAP], offset)
    else:
        at = {}
        for column, symbol in insertions:
            at.setdefault(column, []).append(symbol)
        out = []
        for column, symbol in enumerate(row):
            out.extend(at.get(column, ()))
            if symbol != GAP:
                out.append(int(symbol))
        out.extend(at.get(len(row), ()))
        oriented = np.roll(np.array(out, np.uint8), offset)
    return reverse_complement(oriented) if strand == "R" else oriented


def align_cluster(sequences, members, identity_threshold: float = 0.35,
                  workers: int = 24, device: str = "cuda") -> dict:
    """Align every member of one cluster to its consensus.

    Parameters
    ----------
    identity_threshold
        A member whose edit distance exceeds this fraction of the modal length is left out of
        the matrix and reported in `tail`.
    workers
        Threads for the aligner. edlib releases the GIL, so threads parallelise without
        copying the consensus per worker.

    Returns
    -------
    dict
        `matrix` of shape `(rows, modal)`, `rows` with the per-member items needed to invert
        the alignment, `modal` length, and `tail` of member indices that did not align.

    Raises
    ------
    ValueError
        If no member survives the identity threshold.
    """
    from concurrent.futures import ThreadPoolExecutor

    consensus, modal = build_consensus(sequences, members, device)
    consensus_text = to_bytes(consensus)

    member_sequences = [sequences[m["index"] - 1] for m in members]
    fitted = [_fit(s, modal) for s in member_sequences]
    placements = _rotations_and_strands(fitted, consensus, modal, device)
    del fitted                                            # a full copy of the cluster, done with

    def run(i):
        lag, strand = placements[i]
        return i, align_member(member_sequences[i], consensus, modal, lag, strand,
                               consensus_text)

    if workers > 1 and len(members) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = sorted(pool.map(run, range(len(members))), key=lambda r: r[0])
    else:
        results = [run(i) for i in range(len(members))]

    # The matrix is filled in place rather than stacked from a list of rows. On a cluster of a
    # few hundred thousand members the matrix runs to gigabytes, and stacking would hold the
    # rows, the stack and the cast copy at once; writing each row into the destination and
    # dropping it keeps one copy. Which members survive is known only after the identity
    # threshold is applied, so that check runs first and the destination is sized from its
    # verdict.
    keep, tail = [], []
    for j, (i, result) in enumerate(results):
        distance = result[4]
        if distance / modal > identity_threshold:
            tail.append(members[i]["index"])
        else:
            keep.append(j)
    if not keep:
        raise ValueError("no member survived the identity threshold")

    matrix = np.empty((len(keep), modal), np.uint8)
    meta = []
    for out_row, j in enumerate(keep):
        i, (offset, strand, row, insertions, _distance, oriented_length) = results[j]
        matrix[out_row] = row
        results[j] = None                                 # release the row as it is consumed
        meta.append({"index": members[i]["index"], "offset": int(offset), "strand": strand,
                     "length": int(oriented_length),
                     "insertions": [[int(p), int(b)] for p, b in insertions]})

    return {"matrix": matrix, "rows": meta, "modal": modal, "tail": tail}
