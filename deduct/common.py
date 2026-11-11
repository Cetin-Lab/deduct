"""Shared primitives: entropy coding, hashing, integer packing, and the symbol alphabets.

Two alphabets are in use and they are not interchangeable.

    learned path    A,C,G,T -> 1,2,3,4 and alignment gap -> 5
    fallback path   A,C,G,T -> 0,1,2,3, two bits per base, no gap symbol

Cleaned sequences contain only ACGT. Any other character (N and other ambiguity codes) is
removed at separation time and its position recorded in the side channel, so neither alphabet
ever has to represent one.
"""
from __future__ import annotations

import gzip
import hashlib
import lzma

import numpy as np

# --------------------------------------------------------------------------- alphabets

GAP = 5
"""Symbol reserved for an alignment gap in the learned path's alphabet."""

CODE = np.zeros(256, np.uint8)
for _i, _b in enumerate(b"ACGT"):
    CODE[_b] = _i + 1
CODE[b"a"[0]], CODE[b"c"[0]], CODE[b"g"[0]], CODE[b"t"[0]] = 1, 2, 3, 4

DECODE = np.frombuffer(b"\x00ACGT", np.uint8)
"""Inverse of CODE for values 1..4; index 0 is unused."""

COMPLEMENT = np.array([0, 4, 3, 2, 1], np.uint8)
"""Complement over the 1..4 coding; index 0 is unused."""


def reverse_complement(seq: np.ndarray) -> np.ndarray:
    """Reverse complement of a sequence in the 1..4 coding."""
    return COMPLEMENT[seq][::-1]


def to_bytes(seq: np.ndarray) -> bytes:
    """Render a 1..4 coded sequence as ASCII ACGT."""
    return DECODE[seq].tobytes()


# --------------------------------------------------------------------------- entropy coding

MAX_LEVEL = 9
"""Highest DEFLATE level. gzip's scale runs 1..9."""


def gzip_compress(data: bytes, level: int = MAX_LEVEL, threads: int = 0) -> bytes:
    """Compress with gzip.

    `mtime=0` is not cosmetic. gzip stamps the current time into its header by default, which
    makes two compressions of the same input differ byte for byte; the compressed output would
    still decode, but no size or digest reported from it would be reproducible. Zeroing the
    stamp makes the output a function of the input alone.

    `threads` is accepted and ignored: DEFLATE as the standard library exposes it is
    single-threaded. The parameter exists so that a caller that wants to ask for parallelism
    does not have to special-case this coder.
    """
    return gzip.compress(data, compresslevel=level, mtime=0)


def gzip_decompress(data: bytes) -> bytes:
    """Decompress a gzip member. The level used to write it does not need to be known."""
    return gzip.decompress(data)


def lzma_compress(data: bytes) -> bytes:
    return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)


def lzma_decompress(data: bytes) -> bytes:
    return lzma.decompress(data)


def compress_smaller(data: bytes, level: int = MAX_LEVEL) -> bytes:
    """Compress with both gzip and lzma and keep the smaller result.

    The result carries a one-byte tag so `decompress_tagged` can undo it without being told
    which coder ran. Nothing in the released configuration reaches this: the latent, residual
    and weight coders all call `gzip_compress` directly, so every blob in a compressed output
    is DEFLATE. It is kept for the `dropk_best` and `dense_best` variants.
    """
    z = gzip_compress(data, level)
    try:
        x = lzma_compress(data)
    except Exception:
        return b"G" + z
    return b"L" + x if len(x) < len(z) else b"G" + z


def decompress_tagged(blob: bytes) -> bytes:
    """Inverse of `compress_smaller`."""
    tag, body = blob[:1], blob[1:]
    if tag == b"G":
        return gzip_decompress(body)
    if tag == b"L":
        return lzma_decompress(body)
    raise ValueError(f"unknown compressor tag {tag!r}")


# --------------------------------------------------------------------------- hashing

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- integer packing

def varint_encode(values) -> bytes:
    """LEB128 over non-negative integers."""
    out = bytearray()
    for v in np.asarray(values, np.int64).tolist():
        v = int(v)
        if v < 0:
            raise ValueError("varint_encode requires non-negative values")
        while True:
            byte = v & 0x7F
            v >>= 7
            out.append(byte | 0x80 if v else byte)
            if not v:
                break
    return bytes(out)


def varint_decode(buf: bytes, count: int, start: int = 0) -> tuple[np.ndarray, int]:
    """Read `count` LEB128 values from `buf` starting at `start`.

    Returns the values and the offset one past the last byte consumed.
    """
    out = np.empty(count, np.int64)
    pos = start
    for i in range(count):
        shift = 0
        value = 0
        while True:
            byte = buf[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        out[i] = value
    return out, pos


def bitpack(values, bits: int) -> bytes:
    """Pack small non-negative integers at `bits` each, least significant bit first."""
    flat = np.asarray(values, np.uint8).ravel()
    out = bytearray((flat.size * bits + 7) // 8)
    acc = nbits = pos = 0
    for v in flat.tolist():
        acc |= (int(v) & ((1 << bits) - 1)) << nbits
        nbits += bits
        while nbits >= 8:
            out[pos] = acc & 0xFF
            pos += 1
            acc >>= 8
            nbits -= 8
    if nbits:
        out[pos] = acc & 0xFF
    return bytes(out)


def bitunpack(buf: bytes, count: int, bits: int) -> np.ndarray:
    """Inverse of `bitpack`."""
    out = np.empty(count, np.uint8)
    acc = nbits = pos = 0
    for i in range(count):
        while nbits < bits:
            acc |= buf[pos] << nbits
            pos += 1
            nbits += 8
        out[i] = acc & ((1 << bits) - 1)
        acc >>= bits
        nbits -= bits
    return out


# --------------------------------------------------------------------------- 2-bit packing

_PACK2 = np.zeros(256, np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _PACK2[_b] = _i
_UNPACK2 = np.frombuffer(b"ACGT", np.uint8)


def pack_2bit(sequences) -> tuple[bytes, np.ndarray]:
    """Pack ACGT byte strings into one 2-bit stream.

    Returns the packed bytes and the length of each input sequence. The stream is the
    concatenation of every sequence, four bases per byte, zero-padded to a byte boundary;
    the lengths are what let `unpack_2bit` split it again.
    """
    lengths = np.fromiter((len(s) for s in sequences), np.int64, count=len(sequences))
    total = int(lengths.sum())
    codes = np.empty(total + (-total) % 4, np.uint8)
    codes[total:] = 0
    off = 0
    for s in sequences:
        if s:
            codes[off:off + len(s)] = _PACK2[np.frombuffer(s, np.uint8)]
            off += len(s)
    quads = codes.reshape(-1, 4)
    packed = quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)
    return packed.astype(np.uint8).tobytes(), lengths


def unpack_2bit(packed: bytes, lengths) -> list[bytes]:
    """Inverse of `pack_2bit`."""
    lengths = np.asarray(lengths, np.int64)
    total = int(lengths.sum())
    raw = np.frombuffer(packed, np.uint8)
    codes = np.empty((raw.size, 4), np.uint8)
    codes[:, 0] = raw & 3
    codes[:, 1] = (raw >> 2) & 3
    codes[:, 2] = (raw >> 4) & 3
    codes[:, 3] = (raw >> 6) & 3
    bases = _UNPACK2[codes.reshape(-1)[:total]]
    out, off = [], 0
    for n in lengths.tolist():
        out.append(bases[off:off + n].tobytes())
        off += n
    return out
