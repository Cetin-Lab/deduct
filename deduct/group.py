"""Cluster the sequences of one contig column by rotation- and strand-invariant correlation.

Assemblers cut a circular replicon at an arbitrary position and may emit either strand, so two
records of the same biological contig can differ by a rotation, a reverse complement, or both.
Comparing them therefore means maximising over every rotation and both orientations, which is a
circular cross-correlation and is done in the frequency domain.

Each sequence is expanded into four indicator channels, one per base. The correlation of a
reference `r` with a target `t` at lag `k` is the number of positions where they agree,

    rho(k) = sum_v sum_j [r_j = v] [t_(j+k mod L) = v]

which is a sum of four channel-wise circular correlations and is obtained for every `k` at once
from one inverse transform. Clustering is greedy: the longest unassigned sequence becomes a
reference, everything scoring at or above `accept` joins it, and the process repeats on what is
left until nothing remains or `max_clusters` is reached.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .common import CODE, reverse_complement


def smooth_length(need: int) -> int:
    """Smallest 7-smooth integer at least `need`, or a power of two if that is smaller.

    cuFFT has hand-written kernels for radix 2, 3, 5 and 7, so any 7-smooth length runs at full
    speed while wasting far less of the frame than rounding up to a power of two would.
    """
    power_of_two = 1
    while power_of_two < need:
        power_of_two <<= 1
    best = power_of_two
    a = 1
    while a < need * 2:
        b = a
        while b < need * 2:
            c = b
            while c < need * 2:
                m = c
                while m < need:
                    m <<= 1
                best = min(best, m)
                c *= 7
            b *= 5
        a *= 3
    return min(best, power_of_two)


def frame_length(max_len: int) -> int:
    """FFT frame for a column whose longest sequence is `max_len`.

    Correlation indices run over `[0, L_ref + L_tgt)`, which is below `2 * max_len`, so a frame
    of that size leaves the tiled target free of wrap-around.
    """
    return smooth_length(2 * max_len)


def _reference_spectrum(ref: np.ndarray, n: int, device) -> torch.Tensor:
    """Conjugated transform of the reference's four channels, shape `(4, n//2+1)`."""
    codes = torch.zeros(n, dtype=torch.uint8, device=device)
    codes[:len(ref)] = torch.from_numpy(np.ascontiguousarray(ref)).to(device)
    channels = torch.stack([(codes == c).float() for c in (1, 2, 3, 4)], 0)
    return torch.conj(torch.fft.rfft(channels, n=n, dim=1))


def _lay_out(batch, n: int, rc: bool, ref_len: int | None):
    """Write targets into the frame, twice back to back where that is valid.

    A single zero-padded copy gives a *linear* correlation, under which a rotation by `k`
    scores only `max(k, L-k)` matches and two identical sequences are rejected for most `k`.
    Writing the target at `[0, L)` and again at `[L, 2L)` makes `t[(j+k) mod L]` readable at
    index `j+k` for every `j < L_ref` and `k < L`, which is the circular correlation.

    Tiling is applied only when `2L <= n` and `L_ref <= 2L`; beyond that the reference would be
    compared against wrapped target, where an unrelated pair matches at chance. Rows that are
    not tiled are searched over the whole frame instead.

    Returns the packed codes and, per row, the highest lag the layout can score honestly.
    """
    codes = np.zeros((len(batch), n), np.uint8)
    bound = np.empty(len(batch), np.int64)
    for i, seq in enumerate(batch):
        s = reverse_complement(seq) if rc else seq
        length = len(s)
        codes[i, :length] = s
        if ref_len is not None and 2 * length <= n and ref_len <= 2 * length:
            codes[i, length:2 * length] = s
            bound[i] = length
        else:
            bound[i] = n
    return codes, bound


def _best_lag(correlation: torch.Tensor, bound: torch.Tensor):
    """Peak and its lag, searching each row only up to its own bound.

    Lags at or beyond a tiled row's length re-read the second copy and would double-count.
    """
    lags = torch.arange(correlation.shape[1], device=correlation.device).unsqueeze(0)
    return correlation.masked_fill(lags >= bound.unsqueeze(1), float("-inf")).max(dim=1)


def _is_capacity_error(exc: Exception) -> bool:
    """Whether an exception says the call was too big, rather than that something is wrong.

    cuFFT reports a workspace it cannot allocate as an internal error rather than as an
    allocation failure, and a kernel that outruns the driver's watchdog reports a launch
    timeout. Both mean the same thing to the caller: do less per call.
    """
    text = str(exc)
    return ("CUFFT_INTERNAL_ERROR" in text or "CUFFT_ALLOC_FAILED" in text
            or "launch timed out" in text or "out of memory" in text.lower())


POINT_BUDGET = 1 << 28
"""Transform points per call.

Bounds how long a single kernel runs, because a driver that drives a display kills a launch
that outruns its watchdog. Measured on the worst frame this pipeline produces (a 5 Mbp
contig, so a frame of 10,077,696 points), every budget up to 1<<29 completes, and on a frame
of 268,800 points the budget makes no measurable difference at all (3.5 s against 3.3 s for a
whole column). It is set below the largest value that worked rather than at it, and a launch
that times out anyway is treated as a request to do less per call and retried."""


def _rows_per_call(n: int) -> int:
    """Rows to transform at once for a frame of `n` points, at least one."""
    return max(1, POINT_BUDGET // max(1, 4 * n))


def _transform_rows(codes, n: int, device, sub_batch: int | None = None) -> torch.Tensor:
    """Transform a block of packed rows into `(rows, 4, n//2+1)`.

    cuFFT allocates a workspace that grows with the frame and raises rather than reporting an
    allocation failure when it does not fit, so bounding the input tensor is not sufficient. A
    failing sub-batch is halved and retried down to a single row before giving up.
    """
    rows = len(codes)
    out = torch.empty((rows, 4, n // 2 + 1), dtype=torch.complex64, device=device)
    size = min(sub_batch or rows, _rows_per_call(n))
    i = 0
    while i < rows:
        take = min(size, rows - i)
        try:
            block = torch.from_numpy(codes[i:i + take]).to(device)
            channels = torch.stack([(block == c).float() for c in (1, 2, 3, 4)], 1)
            out[i:i + take] = torch.fft.rfft(channels, n=n, dim=2)
            del block, channels
            i += take
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if size == 1 or not _is_capacity_error(exc):
                raise
            size = max(1, size // 2)
            torch.cuda.empty_cache()
    return out


def _score_uncached(ref_spectrum, targets, n, device, rc, ref_len, chunk):
    """Correlate a batch of targets against the reference, `chunk` rows at a time.

    The transform of a batch is `rows * 4 * (n/2+1)` complex values, which for a long column
    runs to hundreds of gigabytes if the whole batch is taken at once. `chunk` is sized from
    the frame by the caller.
    """
    peak = np.empty(len(targets))
    lag = np.empty(len(targets), np.int64)
    chunk = max(1, min(chunk, _rows_per_call(n)))
    for start in range(0, len(targets), chunk):
        batch = targets[start:start + chunk]
        codes, bound = _lay_out(batch, n, rc, ref_len)
        spectra = _transform_rows(codes, n, device)
        cross = (ref_spectrum.unsqueeze(0) * spectra).sum(1)
        del spectra
        correlation = torch.fft.irfft(cross, n=n, dim=1)
        p, l = _best_lag(correlation, torch.from_numpy(bound).to(device))
        peak[start:start + len(batch)] = p.cpu().numpy()
        lag[start:start + len(batch)] = l.cpu().numpy()
        del correlation, cross
    return peak, lag


def _build_cache(sequences, n, device, rc, chunk) -> torch.Tensor:
    """Transform every sequence once, untiled.

    Whether a target should be tiled depends on the reference length, which changes every
    greedy round, so the cache cannot bake that decision in. The tiled transform is recovered
    exactly at use time: writing a second copy at offset `L` is a shift, and a shift is a phase,
    `F{x at 0 and at L} = F{x} * (1 + exp(-2*pi*i*m*L/n))`.
    """
    out = torch.empty((len(sequences), 4, n // 2 + 1), dtype=torch.complex64, device=device)
    for start in range(0, len(sequences), chunk):
        codes, _ = _lay_out(sequences[start:start + chunk], n, rc, ref_len=None)
        out[start:start + len(codes)] = _transform_rows(codes, n, device)
    return out


def _score_cached(ref_spectrum, cache, index, n, chunk, lengths, ref_len):
    """As `_score_uncached`, reading untiled transforms from `cache` and applying the phase."""
    count = len(index)
    peak = np.empty(count)
    lag = np.empty(count, np.int64)
    freq = torch.arange(n // 2 + 1, device=cache.device, dtype=torch.float32)
    for start in range(0, count, chunk):
        sel = index[start:start + chunk]
        target_len = lengths[sel]
        tile = (2 * target_len <= n) & (ref_len <= 2 * target_len)
        spectra = cache[sel]
        if bool(tile.any()):
            phase = torch.exp(torch.complex(
                torch.zeros(1, device=cache.device),
                -2.0 * math.pi * freq[None, :] * target_len[:, None].float() / n))
            factor = torch.where(tile[:, None], 1.0 + phase, torch.ones_like(phase))
            spectra = spectra * factor.unsqueeze(1)
        cross = (ref_spectrum.unsqueeze(0) * spectra).sum(1)
        del spectra
        correlation = torch.fft.irfft(cross, n=n, dim=1)
        bound = torch.where(tile, target_len, torch.full_like(target_len, n))
        p, l = _best_lag(correlation, bound)
        peak[start:start + len(sel)] = p.cpu().numpy()
        lag[start:start + len(sel)] = l.cpu().numpy()
    return peak, lag


def load_column(path: str) -> list[np.ndarray]:
    """Read a contig column file into coded sequences, skipping blank lines."""
    out = []
    with open(path, "rb") as f:
        for line in f:
            line = line.rstrip(b"\r\n")
            if line:
                out.append(CODE[np.frombuffer(line.upper(), np.uint8)])
    return out


def _cache_budget(device) -> int:
    """Free VRAM the transform cache may claim, in bytes.

    torch keeps freed blocks in its own pool, so what the driver reports as free under-states
    what is reusable; the reserved-but-unallocated slice is added back. A reserve is left for
    the per-round gather, which the cache does not cover.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return 0
    dev = torch.device(device)
    free = (torch.cuda.mem_get_info(dev)[0]
            + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev))
    return int(0.8 * free) - int(2e9)


def cluster_column(sequences, device: str = "cuda", accept: float = 0.6,
                   max_clusters: int = 30, chunk: int = 32, cache: bool = True) -> dict:
    """Cluster one column's sequences.

    Parameters
    ----------
    sequences
        Coded sequences, in the line order of the column file. A member's `index` in the result
        is 1-based over this list.
    accept
        Match fraction at or above which a target joins the current reference's cluster. The
        fraction is the correlation peak over `min(L_ref, L_tgt)`, capped at 1.
    max_clusters
        Stop after this many clusters. Whatever is unassigned is reported as `leftover`.

    Returns
    -------
    dict
        `{"clusters": [{"reference": i, "reference_length": L, "members": [...]}, ...],
          "leftover": [...]}` where each member carries `index`, `lag`, `strand` and `score`.
        `lag` is the rotation that aligns the member to its reference and `strand` is `"F"` or
        `"R"`; both are what a decoder must replay, so they are part of the result rather than
        an internal detail.
    """
    if not sequences:
        return {"clusters": [], "leftover": []}

    lengths = np.array([len(s) for s in sequences])
    n = frame_length(int(lengths.max()))
    chunk = max(1, min(chunk, int(0.5e9 // (16 * n)) if n else chunk))
    lengths_device = torch.from_numpy(lengths).to(device)

    # The cache holds every sequence's transform for the whole run, which is worth it because
    # the greedy loop re-scores most sequences many times. It is an optimisation only: without
    # it the transforms are recomputed each round, which costs time and nothing else.
    forward = reverse = None
    if cache:
        one_orientation = len(sequences) * 4 * (n // 2 + 1) * 8
        budget = _cache_budget(device)
        try:
            if 2 * one_orientation <= budget:
                forward = _build_cache(sequences, n, device, False, chunk)
                reverse = _build_cache(sequences, n, device, True, chunk)
            elif one_orientation <= budget:
                forward = _build_cache(sequences, n, device, False, chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            forward = reverse = None

    unassigned = list(range(len(sequences)))
    clusters = []
    while unassigned and len(clusters) < max_clusters:
        ref_index = max(unassigned, key=lambda i: lengths[i])
        ref_len = int(lengths[ref_index])
        ref_spectrum = _reference_spectrum(sequences[ref_index], n, device)

        targets = [sequences[i] for i in unassigned]
        if forward is not None:
            index = torch.tensor(unassigned, dtype=torch.long, device=device)
            peak_f, lag_f = _score_cached(ref_spectrum, forward, index, n, chunk,
                                          lengths_device, ref_len)
        else:
            peak_f, lag_f = _score_uncached(ref_spectrum, targets, n, device, False, ref_len,
                                            chunk)
        if reverse is not None:
            index = torch.tensor(unassigned, dtype=torch.long, device=device)
            peak_r, lag_r = _score_cached(ref_spectrum, reverse, index, n, chunk,
                                          lengths_device, ref_len)
        else:
            peak_r, lag_r = _score_uncached(ref_spectrum, targets, n, device, True, ref_len,
                                            chunk)

        members, remaining = [], []
        for j, i in enumerate(unassigned):
            overlap = min(ref_len, int(lengths[i]))
            if peak_f[j] >= peak_r[j]:
                lag, strand, score = int(lag_f[j]), "F", peak_f[j] / overlap
            else:
                lag, strand, score = int(lag_r[j]), "R", peak_r[j] / overlap
            # A reference longer than the target spans more than one copy of the tiled target,
            # so the raw count can exceed the overlap; the score is a fraction by definition.
            score = min(score, 1.0)
            if score >= accept:
                members.append({"index": i + 1, "lag": lag, "strand": strand,
                                "score": float(score)})
            else:
                remaining.append(i)
        clusters.append({"reference": ref_index + 1, "reference_length": ref_len,
                         "members": members})
        unassigned = remaining

    return {"clusters": clusters, "leftover": [i + 1 for i in unassigned]}
