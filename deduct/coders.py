"""Byte coders for the three blobs a coded cluster's compressed output contains.

Each coder is an `(encode, decode)` pair and each is registered by name, so the manifest of the
compressed output can record which one wrote a blob and the decoder can select it without
guessing. Every encode is expected to satisfy `decode(encode(x)) == x` exactly;
`tests/test_coders.py` asserts that for every registered coder over representative inputs.

    latent      int16, shape (tiles, channels, frequencies, columns), mostly zero
    residual    uint8, shape (H, W), zero where the prediction was correct
    weights     a PyTorch state dict of float32 tensors

The weight coders take a state dict rather than an array, so decoding needs the parameter
shapes. Those are implied by the model configuration, which the manifest of the compressed
output already carries; nothing about the layout is stored twice.
"""
from __future__ import annotations

import io

import numpy as np

from .common import (MAX_LEVEL, bitpack, bitunpack, compress_smaller, decompress_tagged,
                     varint_decode, varint_encode, gzip_compress, gzip_decompress)

# --------------------------------------------------------------------------- shape headers


def _put_shape(shape) -> bytes:
    return np.asarray(shape, np.int32).tobytes()


def _take_shape(buf: bytes, ndim: int) -> tuple[tuple, bytes]:
    return tuple(np.frombuffer(buf[:4 * ndim], np.int32).tolist()), buf[4 * ndim:]


def _narrow(a: np.ndarray) -> tuple[bytes, bytes]:
    """Return `a` in the narrowest integer width that holds it exactly, with a width tag.

    int8 is the usual case for a quantised latent, but it is not guaranteed: a small
    quantisation step or a wide bottleneck can push a coefficient past 127, and casting
    would then wrap silently. The width is recorded so the decoder never has to guess.
    """
    if a.size and (a.min() < -128 or a.max() > 127):
        return b"2", a.astype(np.int16).tobytes()
    return b"1", a.astype(np.int8).tobytes()


def _widen(tag: bytes, raw: bytes, shape) -> np.ndarray:
    dtype = np.int16 if tag == b"2" else np.int8
    return np.frombuffer(raw, dtype).reshape(shape).astype(np.int16)


# --------------------------------------------------------------------------- latent

def latent_encode(Q: np.ndarray, level: int = MAX_LEVEL, both: bool = False) -> bytes:
    """Write the quantised latent, dropping frequency planes that are entirely zero.

    `Q` is indexed `(tile, channel, frequency, column)`. A plane is one value of the frequency
    index across every tile and column; a plane in which nothing survived quantisation carries
    no information and is not written. A one-byte-per-plane mask records which ones remain.
    """
    _tiles, _chan, n_freq, _cols = Q.shape
    keep = np.array([bool(Q[:, :, k, :].any()) for k in range(n_freq)], np.uint8)
    sub = np.ascontiguousarray(Q[:, :, keep.astype(bool), :])
    tag, raw = _narrow(sub)
    body = compress_smaller(raw, level) if both else b"G" + gzip_compress(raw, level)
    return _put_shape(Q.shape) + keep.tobytes() + tag + body


def latent_decode(blob: bytes) -> np.ndarray:
    """Inverse of `latent_encode`."""
    shape, rest = _take_shape(blob, 4)
    tiles, chan, n_freq, cols = shape
    keep = np.frombuffer(rest[:n_freq], np.uint8).astype(bool)
    sub = _widen(rest[n_freq:n_freq + 1], decompress_tagged(rest[n_freq + 1:]),
                 (tiles, chan, int(keep.sum()), cols))
    Q = np.zeros(shape, np.int16)
    Q[:, :, keep, :] = sub
    return Q


def latent_encode_both(Q: np.ndarray, level: int = MAX_LEVEL) -> bytes:
    """As `latent_encode`, taking whichever of gzip and lzma is smaller."""
    return latent_encode(Q, level, both=True)


def latent_encode_bitpacked(Q: np.ndarray, level: int = MAX_LEVEL) -> bytes:
    """As `latent_encode`, with the surviving planes packed at the width their range needs.

    Cheaper than a byte per coefficient when the quantised values span three bits or fewer,
    which is the common case; falls back to `latent_encode`'s layout when they do not.
    """
    _tiles, _chan, n_freq, _cols = Q.shape
    keep = np.array([bool(Q[:, :, k, :].any()) for k in range(n_freq)], np.uint8)
    sub = np.ascontiguousarray(Q[:, :, keep.astype(bool), :])
    lo = int(sub.min()) if sub.size else 0
    span = int(sub.max()) - lo if sub.size else 0
    bits = max(1, int(span).bit_length())
    if bits > 8:
        return b"\xff" + latent_encode(Q, level)
    body = bitpack((sub - lo).astype(np.uint8), bits)
    header = _put_shape(Q.shape) + keep.tobytes() + bytes([bits]) + np.int32(lo).tobytes()
    return b"\x01" + header + gzip_compress(body, level)


def latent_decode_bitpacked(blob: bytes) -> np.ndarray:
    """Inverse of `latent_encode_bitpacked`."""
    if blob[:1] == b"\xff":
        return latent_decode(blob[1:])
    shape, rest = _take_shape(blob[1:], 4)
    tiles, chan, n_freq, cols = shape
    keep = np.frombuffer(rest[:n_freq], np.uint8).astype(bool)
    bits = rest[n_freq]
    lo = int(np.frombuffer(rest[n_freq + 1:n_freq + 5], np.int32)[0])
    n_keep = int(keep.sum())
    values = bitunpack(gzip_decompress(rest[n_freq + 5:]),
                       tiles * chan * n_keep * cols, bits)
    Q = np.zeros(shape, np.int16)
    Q[:, :, keep, :] = values.astype(np.int16).reshape(tiles, chan, n_keep, cols) + lo
    return Q


LATENT_CODERS = {
    "dropk": (latent_encode, latent_decode),
    "dropk_best": (latent_encode_both, latent_decode),
    "dropk_bitpack": (latent_encode_bitpacked, latent_decode_bitpacked),
}


# --------------------------------------------------------------------------- residual

def residual_encode(R: np.ndarray, level: int = MAX_LEVEL) -> bytes:
    """Write the residual plane.

    `R` is stored dense (zero where the prediction was right, the true symbol elsewhere)
    rather than as a list of (row, column, symbol) triples. The two carry the same information,
    but the dense form keeps a given aligned position at the same offset in consecutive rows,
    which is what a long-range matcher needs in order to match the variation patterns that
    related members share.
    """
    # `memoryview` rather than `tobytes`: the plane is already contiguous and on a large cluster
    # the copy is gigabytes. The bytes fed to the entropy coder are the same either way.
    return _put_shape(R.shape) + gzip_compress(memoryview(np.ascontiguousarray(R)), level)


def residual_decode(blob: bytes) -> np.ndarray:
    """Inverse of `residual_encode`."""
    shape, rest = _take_shape(blob, 2)
    return np.frombuffer(gzip_decompress(rest), np.uint8).reshape(shape)


def residual_encode_smaller(R: np.ndarray, level: int = MAX_LEVEL) -> bytes:
    """As `residual_encode`, taking whichever of gzip and lzma is smaller."""
    return _put_shape(R.shape) + compress_smaller(R.tobytes(), level)


def residual_decode_smaller(blob: bytes) -> np.ndarray:
    shape, rest = _take_shape(blob, 2)
    return np.frombuffer(decompress_tagged(rest), np.uint8).reshape(shape)


def residual_encode_sparse(R: np.ndarray, level: int = MAX_LEVEL) -> bytes:
    """Write only the positions that were predicted wrong, and what belongs there.

    Nonzeros come out of `np.nonzero` ordered by row then column, so the row index is stored as
    a difference against the previous entry and the column either as a difference within the
    same row or, when the row advances, absolutely. Symbols take three bits.

    Which of this and the dense plane is smaller depends on how often the prediction is wrong.
    A model that is wrong on half the cells produces a residual that is mostly the matrix, and
    the matrix compresses very well because its rows are near-copies of one another; a model
    that is wrong on a fraction of a percent produces scattered isolated symbols, where naming
    the positions costs less than a plane of zeros around them. `dense_or_sparse` measures both
    rather than choosing by a rule.
    """
    rows, cols = np.nonzero(R)
    header = _put_shape(R.shape) + np.array([rows.size], np.int64).tobytes()
    if rows.size == 0:
        return header
    row_delta = np.diff(rows, prepend=0)
    col_delta = np.where(row_delta == 0, np.diff(cols, prepend=0), cols)
    body = (varint_encode(row_delta) + varint_encode(col_delta)
            + bitpack(R[rows, cols] - 1, 3))
    return header + gzip_compress(body, level)


def residual_decode_sparse(blob: bytes) -> np.ndarray:
    """Inverse of `residual_encode_sparse`."""
    shape, rest = _take_shape(blob, 2)
    count = int(np.frombuffer(rest[:8], np.int64)[0])
    R = np.zeros(shape, np.uint8)
    if count == 0:
        return R
    body = gzip_decompress(rest[8:])
    row_delta, offset = varint_decode(body, count)
    col_delta, offset = varint_decode(body, count, offset)
    rows = np.cumsum(row_delta)
    cols = np.empty(count, np.int64)
    previous = 0
    for i in range(count):
        cols[i] = previous + col_delta[i] if row_delta[i] == 0 else col_delta[i]
        previous = cols[i]
    R[rows, cols] = bitunpack(body[offset:], count, 3) + 1
    return R


SPARSE_ENTRY_LIMIT = 100_000_000
"""Nonzeros above which the sparse residual is not attempted, whatever its density.

Building the sparse form costs about 40 bytes per nonzero while it is being built (two int64
coordinate arrays out of `np.nonzero`, two more for the deltas, and the varint encoder's
intermediate list), so a plane with a billion of them needs tens of gigabytes to answer a
question about a few megabytes of output. This caps that cost at roughly 4 GB.

It is a bound on memory, not a judgement about which coder is better, and it is set far above
what any collection here reaches: the largest cluster whose residual the sparse form actually
won is 6.7 million entries, so the limit sits fifteen times above the top of the observed range
and changes no result in this paper. A collection that does exceed it is one where a single
cluster holds hundreds of millions of corrected cells, and the dense plane is the right answer
there anyway.
"""


def _sparse_can_win(R: np.ndarray) -> bool:
    """Whether the sparse form is worth building at all.

    Two reasons not to. The first is arithmetic: before either candidate reaches the entropy
    coder the dense one is exactly `R.size` bytes and the sparse one is at least `2 + 3/8` bytes
    per nonzero (one byte minimum for each of the two varints and three bits for the symbol),
    so past the density where those cross, at 42.1%, the sparse stream is larger going in and
    cannot come out smaller. Across the corpora in this paper the sparse form wins at densities
    up to 27.7%, so that crossing point sits well clear of anything observed.

    The second is `SPARSE_ENTRY_LIMIT`, which bounds the memory the attempt may cost on a
    cluster large enough for density alone not to protect it.
    """
    nonzero = int(np.count_nonzero(R))
    if nonzero > SPARSE_ENTRY_LIMIT:
        return False
    return 2 * nonzero + (3 * nonzero + 7) // 8 < R.size


def residual_encode_either(R: np.ndarray, level: int = MAX_LEVEL) -> bytes:
    """Write the residual whichever of the two ways is smaller, tagged so it decodes."""
    dense = residual_encode(R, level)
    if not _sparse_can_win(R):
        return b"D" + dense
    sparse = residual_encode_sparse(R, level)
    return b"S" + sparse if len(sparse) < len(dense) else b"D" + dense


def residual_decode_either(blob: bytes) -> np.ndarray:
    tag, body = blob[:1], blob[1:]
    if tag == b"D":
        return residual_decode(body)
    if tag == b"S":
        return residual_decode_sparse(body)
    raise ValueError(f"unknown residual layout tag {tag!r}")


RESIDUAL_CODERS = {
    "dense": (residual_encode, residual_decode),
    "sparse": (residual_encode_sparse, residual_decode_sparse),
    "either": (residual_encode_either, residual_decode_either),
    "dense_best": (residual_encode_smaller, residual_decode_smaller),
}


# --------------------------------------------------------------------------- weights

def weights_encode(state, dtype=np.float16, level: int = MAX_LEVEL) -> bytes:
    """Write a state dict as raw tensor data, in the dict's own key order.

    No names, shapes or pickle framing are stored: the decoder builds an identically
    configured model first and reads the values into its parameters in the same order.

    Rounding to float16 keeps the compressed output exact. Correctness in this codec is
    constructive (the residual is computed against whatever the decoder will produce), so a
    lossy weight format is safe as long as the *encoder* predicts with the same rounded
    weights. That means serialising before predicting, not after; `encode.py` does it in that
    order and asserts the reconstruction.
    """
    flat = b"".join(t.detach().cpu().numpy().astype(dtype).tobytes() for t in state.values())
    return gzip_compress(flat, level)


def weights_decode(blob: bytes, template, dtype=np.float16):
    """Inverse of `weights_encode`. `template` supplies the key order, shapes and sizes."""
    import torch
    flat = np.frombuffer(gzip_decompress(blob), dtype)
    out, off = {}, 0
    for key, ref in template.items():
        n = ref.numel()
        out[key] = torch.tensor(flat[off:off + n].astype(np.float32)).view(ref.shape)
        off += n
    if off != flat.size:
        raise ValueError(f"weight blob holds {flat.size} values, template expects {off}")
    return out


def weights_encode_f32(state, level: int = MAX_LEVEL) -> bytes:
    return weights_encode(state, np.float32, level)


def weights_decode_f32(blob: bytes, template):
    return weights_decode(blob, template, np.float32)


def weights_encode_torch(state, level: int = MAX_LEVEL) -> bytes:
    """Write a state dict through `torch.save`.

    Kept for comparison only. The pickle framing is a fixed overhead of roughly 800 bytes
    whatever the model contains, which dominates the blob for a small decoder.
    """
    import torch
    buf = io.BytesIO()
    torch.save(state, buf)
    return gzip_compress(buf.getvalue(), level)


def weights_decode_torch(blob: bytes, template=None):
    import torch
    return torch.load(io.BytesIO(gzip_decompress(blob)), map_location="cpu", weights_only=True)


WEIGHT_CODERS = {
    "raw16": (weights_encode, weights_decode),
    "raw32": (weights_encode_f32, weights_decode_f32),
    "torch": (weights_encode_torch, weights_decode_torch),
}
