"""Entropy coder for everything the learned path does not take.

Three kinds of sequence end up here: members of a cluster too divergent to align, sequences in
no cluster at all, and clusters below the admission threshold. What they have in common is that
no aligned matrix exists for them, so there is nothing for a column-structured coder to model.

They are coded as one pool rather than per contig or per file. Related sequences that were never
clustered still share long stretches, and a match can only be found if both copies are inside
the coder's window. Sorting by length puts sequences of similar size next to each other, which
brings near-duplicates within reach of the window more often than input order does.

The pool is coded with gzip, and the window is what bounds what that can find. DEFLATE's
window is 32 KiB and is not adjustable, so a repeat has to fall within 32 KiB of its earlier
copy to be coded as a match at all. On a pool of several gigabytes that is a small fraction of
the cross-sequence redundancy present. Length ordering is what brings near-duplicates close
enough together to be reachable, which is why it is not merely cosmetic here.
"""
from __future__ import annotations

import numpy as np

from .common import (gzip_compress, gzip_decompress, pack_2bit, unpack_2bit,
                     varint_decode, varint_encode)

WINDOW_LOG = 15
"""DEFLATE's window exponent: 2^15 bytes. Fixed by the format, not a tuning choice: there is
no way to raise it and no long-distance mode to reach past it. It is recorded here because it
bounds what the pool's ordering has to achieve: two copies further apart than this cannot be
coded as a match however redundant they are."""


def _compress_pool(data: bytes, level: int) -> bytes:
    return gzip_compress(data, level)


def _decompress_pool(blob: bytes) -> bytes:
    return gzip_decompress(blob)


def build_pool(entries, level: int = 9, pack: bool = False, order: str = "length") -> dict:
    """Compress a pool of sequences and return the blobs plus the index needed to split it.

    Parameters
    ----------
    entries
        `(key, sequence)` pairs, where `sequence` is ASCII ACGT bytes and `key` is whatever the
        caller needs to find it again.
    pack
        Pack to two bits per base before compressing. Halves the input but hides byte-aligned
        repeats from the matcher, which costs more than the packing saves whenever the pool is
        genuinely redundant.
    order
        `length` sorts by sequence length before pooling, `input` keeps the given order.

    Returns
    -------
    dict
        `pool` and `index` blobs, and the uncompressed size for reporting.
    """
    items = list(entries)
    if order == "length":
        items.sort(key=lambda kv: len(kv[1]))
    elif order != "input":
        raise ValueError(f"unknown order {order!r}; have 'length' and 'input'")

    keys = [k for k, _ in items]
    sequences = [s for _, s in items]
    if pack:
        payload, lengths = pack_2bit(sequences)
        lengths = lengths.tolist()
    else:
        payload = b"".join(sequences)
        lengths = [len(s) for s in sequences]

    pool = _compress_pool(payload, level)
    index = gzip_compress(_pack_index(keys, lengths, pack), level)
    return {"pool": pool, "index": index, "raw_bytes": len(payload),
            "total": len(pool) + len(index)}


def _pack_index(keys, lengths, packed: bool) -> bytes:
    """Serialise the pool's index as integers rather than as JSON text.

    A key is `"column:line"`, so the index is two small integer sequences and a length. Written
    as JSON they are decimal text with punctuation, larger than the varints that encode the same
    values.

    All three are stored absolutely rather than as differences. The pool is ordered by sequence
    length, not by column, so consecutive entries carry unrelated column and line numbers and
    differencing them would produce values no smaller than the originals and, for columns,
    negative ones.
    """
    column = np.array([int(k.split(":")[0]) for k in keys], np.int64)
    line = np.array([int(k.split(":")[1]) for k in keys], np.int64)
    return (bytes([1 if packed else 0])
            + np.array([len(keys)], np.int64).tobytes()
            + varint_encode(column) + varint_encode(line)
            + varint_encode(np.asarray(lengths, np.int64)))


def _unpack_index(blob: bytes):
    """Inverse of `_pack_index`, returning `(keys, lengths, packed)`."""
    packed = bool(blob[0])
    count = int(np.frombuffer(blob[1:9], np.int64)[0])
    column, offset = varint_decode(blob, count, 9)
    line, offset = varint_decode(blob, count, offset)
    lengths, _ = varint_decode(blob, count, offset)
    return [f"{c}:{l}" for c, l in zip(column.tolist(), line.tolist())], lengths, packed


def read_pool(blobs: dict) -> dict:
    """Inverse of `build_pool`: return `{key: sequence}`."""
    keys, lengths, packed = _unpack_index(gzip_decompress(blobs["index"]))
    payload = _decompress_pool(blobs["pool"])
    if packed:
        sequences = unpack_2bit(payload, np.asarray(lengths, np.int64))
    else:
        sequences, offset = [], 0
        for n in lengths.tolist():
            sequences.append(payload[offset:offset + n])
            offset += n
    return dict(zip(keys, sequences))
