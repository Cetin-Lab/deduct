"""Transpose a normalised collection into contig columns, and record what transposition drops.

Column `n` holds contig `n` of every genome, one sequence per line, empty where a genome has no
such record. That is the decomposition the rest of the pipeline works on: sequences in the same
column are the same contig of different assemblies, so they are the ones likely to be
homologous.

Everything the transposition discards goes into a side channel, which is part of the compressed
output:

    headers     every record header, in file order then record order
    structure   per-file name, size and SHA-256; per-file record count; per-column list of the
                genomes that contribute a line, in line order
    exceptions  the position and identity of every non-ACGT character, stored as run-length
                records because the ambiguity content of real collections arrives in runs

With those four, the original normalised files rebuild exactly, and the per-file checksum is
what the decoder verifies against.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

from .common import gzip_compress, gzip_decompress, sha256_bytes

NON_ACGT = re.compile(rb"[^ACGT]")
NON_ACGT_RUN = re.compile(rb"([^ACGT])\1*")
"""A maximal run of one repeated non-ACGT symbol. `NNNRR` is two runs, not five characters."""

_COLUMN_PATTERNS = [
    # `normalise` appends the authoritative record index at the end of the header, and a
    # trailing "contigN" is unambiguous, so it is tried first. Some source headers carry an
    # unrelated "contigNNNNN" of their own mid-description; matching that one instead would
    # assign the record to a column numbered in the millions.
    re.compile(r"(?<![A-Za-z])contig(\d+)\s*$"),
    re.compile(r"(?:denovo|guided)\.(\d+)"),
    re.compile(r"SCcontig(\d+)"),
    re.compile(r"SCAFFOLD_(\d+)_"),
    re.compile(r"NODE_(\d+)_"),
    re.compile(r"(?<![A-Za-z])contig(\d+)\b"),
]


def column_of(header: bytes) -> int | None:
    """Read a record's column number out of its header, or None if no pattern matches."""
    text = header.decode("latin1")
    for pattern in _COLUMN_PATTERNS:
        m = pattern.search(text)
        if m:
            return int(m.group(1))
    return None


def _parse_file(args) -> tuple:
    """Read one normalised file into `(name, size, sha256, records, unparsed, symbols)`.

    A record is `(column, header, cleaned_sequence, exceptions)`. `exceptions` lists runs
    `(start, length, symbol)` in the coordinates of the *original* sequence, sorted by start,
    so that reinserting them in order restores it. A record whose header carries no column
    number keeps its sequence uncleaned and is stored verbatim in the side channel. A record
    whose cleaned sequence is empty is refused: the column files cannot carry it (below).

    `symbols` counts every character of every record as it was read, before any non-ACGT
    removal. It is the denominator of F-BPB: the exceptions blob is sequence the compressed
    output has to carry, so the characters it stands for belong in the count. Under
    `--ambiguity delete` it exceeds the ACGT base count by whatever share of the collection is
    ambiguous.
    """
    src_dir, name, ambiguity = args
    with open(os.path.join(src_dir, name), "rb") as f:
        raw = f.read()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]
    # A file that normalised to nothing, or to an odd number of lines, cannot be represented:
    # reassemble writes header/sequence pairs, so it would come back different bytes. Both
    # shapes mean the input was empty, truncated or not FASTA; refuse rather than miscode.
    if not lines:
        raise ValueError(f"{name} contains no records; the file is empty or is not FASTA")
    if len(lines) % 2:
        raise ValueError(f"{name} has an odd number of lines where header/sequence pairs "
                         f"were expected; the file may be truncated")

    records, unparsed, symbols = [], 0, 0
    for i in range(0, len(lines), 2):
        header, seq = lines[i], lines[i + 1]
        symbols += len(seq)                     # counted before any non-ACGT removal
        column = column_of(header)
        if column is None:
            unparsed += 1
            records.append((None, header, seq, None))
            continue
        exceptions = None
        if NON_ACGT.search(seq):
            exceptions = [(m.start(), m.end() - m.start(), m.group(1)[0])
                          for m in NON_ACGT_RUN.finditer(seq)]
            # `delete` removes the symbol, which shortens the sequence and shifts every
            # coordinate after it. That is invertible, but it is not neutral for the stage
            # that consumes the result: clustering scores a circular rotation correlation,
            # and a deletion frame-shifts the tail out of register. `mask` substitutes a
            # placeholder base instead, so the length and every downstream coordinate survive
            # and the true symbol is restored from the same exception list.
            seq = NON_ACGT.sub(b"A" if ambiguity == "mask" else b"", seq)
        if not seq:
            # The column files store one sequence per line, so an empty cleaned sequence is
            # indistinguishable from the separator, and every consumer downstream resolves
            # that by dropping the line; compression would complete silently and then fail
            # (or splice records together) at decompression. Refuse before anything is
            # written. An empty sequence can only arise here two ways, told apart by
            # `exceptions`: `delete` stripped a record that was entirely non-ACGT, or the
            # record was empty in the input.
            where = f"record {i // 2 + 1} of {name} ({header.decode('latin1')})"
            if exceptions is not None:
                raise ValueError(
                    f"{where} is entirely non-ACGT and --ambiguity delete would discard it; "
                    f"compress with --ambiguity mask, which preserves such records")
            raise ValueError(f"{where} has an empty sequence (a bare header, possibly a "
                             f"truncated download); fix or remove the record")
        records.append((column, header, seq, exceptions))
    return name, len(raw), sha256_bytes(raw), records, unparsed, symbols


EXCEPTIONS_FORMAT = 0x01
"""First byte of the decompressed exceptions blob. The decoder refuses anything else."""


def _append_varint(buf: bytearray, n: int) -> None:
    """Append `n` as an unsigned LEB128 varint: 7 bits per byte, high bit says "more"."""
    if n < 0:
        # Unreachable from run extraction, whose runs are sorted and disjoint; a negative
        # delta would otherwise loop forever below, and a hang is the worst way to fail.
        raise ValueError(f"varint cannot encode negative value {n}")
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("exceptions blob is truncated inside a varint")
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _pack_exceptions(per_file) -> bytes:
    """Serialize every record's non-ACGT runs, in file order then record order.

    One format byte; per file a varint record count; per record a varint run count; per run
    `varint delta_start` (from the previous run's end, so adjacent runs cost one zero byte),
    `varint run_length`, and the symbol as a raw byte: N dominates but R/Y/W/K and friends
    occur, so nothing here assumes an alphabet.
    """
    buf = bytearray([EXCEPTIONS_FORMAT])
    for records in per_file:
        _append_varint(buf, len(records))
        for _column, _header, _seq, exceptions in records:
            runs = exceptions or ()
            _append_varint(buf, len(runs))
            end = 0
            for start, length, symbol in runs:
                _append_varint(buf, start - end)
                _append_varint(buf, length)
                buf.append(symbol)
                end = start + length
    return bytes(buf)


def _unpack_exceptions(data: bytes, n_files: int) -> dict:
    """Parse the exceptions blob back into `{"file:record": [(start, length, symbol), ...]}`.

    Only records with at least one run get a key, which is what `reassemble` looks up. The
    parse fails loudly on an unknown format byte, on truncation, and on trailing bytes: a
    compressed output whose exceptions blob does not parse exactly is not one to guess about.
    """
    if not data:
        raise ValueError("exceptions blob is empty")
    if data[0] != EXCEPTIONS_FORMAT:
        raise ValueError(f"exceptions blob has unknown format byte 0x{data[0]:02x}; "
                         f"expected 0x{EXCEPTIONS_FORMAT:02x}")
    out = {}
    pos = 1
    for i in range(n_files):
        n_records, pos = _read_varint(data, pos)
        for j in range(n_records):
            n_runs, pos = _read_varint(data, pos)
            if not n_runs:
                continue
            runs, end = [], 0
            for _ in range(n_runs):
                delta, pos = _read_varint(data, pos)
                length, pos = _read_varint(data, pos)
                if pos >= len(data):
                    raise ValueError("exceptions blob is truncated inside a run")
                symbol = data[pos]
                pos += 1
                start = end + delta
                runs.append((start, length, symbol))
                end = start + length
            out[f"{i}:{j}"] = runs
    if pos != len(data):
        raise ValueError(f"exceptions blob has {len(data) - pos} trailing bytes "
                         f"after {n_files} files")
    return out


SIDE_CHANNEL_LEVEL = 9
"""gzip level for the side channel.

The side channel is text (record headers and a JSON of the per-file structure) and the
symbol planes are not, so the level is kept separate in case the two ever want different
settings. On DEFLATE's 1..9 scale both want the maximum, so they currently agree.
"""


def separate(src_dir: str, out_dir: str, workers: int = 32, level: int = SIDE_CHANNEL_LEVEL,
             checksums: bool = False, ambiguity: str = "delete") -> dict:
    """Transpose `src_dir` into `out_dir` and write the side channel.

    Produces `columns/contig{n}.txt`, `columns/genomes.txt` (the file order that indexes rows),
    and `sidechannel/{headers,structure,exceptions}.z`. Returns a summary of what was written.
    """
    column_dir = os.path.join(out_dir, "columns")
    side_dir = os.path.join(out_dir, "sidechannel")
    os.makedirs(column_dir, exist_ok=True)
    os.makedirs(side_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(src_dir) if f.endswith(".fna"))
    index_of = {name: i for i, name in enumerate(names)}

    per_file = [None] * len(names)
    meta = [None] * len(names)
    n_columns = total_records = total_bases = unparsed_total = total_symbols = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, size, digest, records, unparsed, symbols in pool.map(
                _parse_file, ((src_dir, n, ambiguity) for n in names), chunksize=4):
            i = index_of[name]
            per_file[i] = records
            meta[i] = {"name": name, "size": size}
            if checksums:
                # A per-file digest is what `deduct verify` compares against. It is not needed
                # to decode: reconstruction is constructive, not checked. On a collection of
                # many short records it is also the largest single side-channel term (half
                # the compressed output on 67,190 mitochondrial genomes), so it is opt-in.
                meta[i]["sha256"] = digest
            unparsed_total += unparsed
            total_symbols += symbols
            for column, _header, seq, _exc in records:
                total_records += 1
                total_bases += len(seq)
                if column is not None:
                    n_columns = max(n_columns, column)

    # One pass over the records, bucketing each into its column. Walking the corpus once per
    # column instead would be O(columns x records), which on a corpus with hundreds of columns
    # and a million records is hundreds of millions of comparisons for a pure re-indexing.
    # Genome order within a column follows file order, which is what `contributors` records.
    buckets: dict[int, list[bytes]] = {}
    contributors: dict[int, list[int]] = {}
    for i, records in enumerate(per_file):
        seen = set()
        for column, _header, seq, _exc in records:
            # A genome contributes at most one line to a column; a repeated index is the
            # second record claiming it, and the first one wins, as before.
            if column is None or column in seen:
                continue
            seen.add(column)
            buckets.setdefault(column, []).append(seq)
            contributors.setdefault(column, []).append(i)

    for column in range(1, n_columns + 1):
        lines = buckets.get(column, [])
        contributors.setdefault(column, [])
        with open(os.path.join(column_dir, f"contig{column}.txt"), "wb") as f:
            f.write(b"\n".join(lines) + (b"\n" if lines else b""))

    with open(os.path.join(column_dir, "genomes.txt"), "w") as f:
        f.write("\n".join(names) + "\n")

    headers = b"\n".join(h for records in per_file for _c, h, _s, _e in records)
    verbatim = {f"{i}:{j}": seq.decode("latin1")
                for i, records in enumerate(per_file)
                for j, (c, _h, seq, _e) in enumerate(records) if c is None}
    structure = {
        "n_columns": n_columns,
        "n_files": len(names),
        "total_records": total_records,
        "total_bases": total_bases,
        "files": meta,
        "record_counts": [len(r) for r in per_file],
        "record_columns": [[c for c, _h, _s, _e in r] for r in per_file],
        "contributors": {str(k): v for k, v in contributors.items()},
        "verbatim": verbatim,
        "ambiguity": ambiguity,
    }

    blobs = {
        "headers.z": gzip_compress(headers, level),
        "structure.z": gzip_compress(json.dumps(structure).encode(), level),
        "exceptions.z": gzip_compress(_pack_exceptions(per_file), level),
    }
    for name, blob in blobs.items():
        with open(os.path.join(side_dir, name), "wb") as f:
            f.write(blob)

    return {
        "genomes": len(names),
        "columns": n_columns,
        "records": total_records,
        "bases": total_bases,
        # `bases` counts what survived the non-ACGT strip; `symbols` counts every character
        # that was read. Under `--ambiguity delete` they differ by the collection's ambiguity
        # content; under `mask` the strip substitutes rather than removes, so the two are equal.
        # F-BPB divides by `symbols` either way, which is what makes the rate comparable across
        # the two modes.
        "symbols": total_symbols,
        "unparsed_records": unparsed_total,
        "sidechannel_bytes": {k: len(v) for k, v in blobs.items()},
        "sidechannel_total": sum(len(v) for v in blobs.values()),
    }


def load_sidechannel(side_dir: str) -> dict:
    """Read the side channel back into headers, structure and exceptions."""
    def read(name):
        with open(os.path.join(side_dir, name), "rb") as f:
            return gzip_decompress(f.read())

    structure = json.loads(read("structure.z"))
    return {
        "headers": read("headers.z").split(b"\n"),
        "structure": structure,
        "exceptions": _unpack_exceptions(read("exceptions.z"), structure["n_files"]),
    }


def restore_sequence(cleaned: bytes, exceptions, ambiguity: str = "delete") -> bytes:
    """Put the non-ACGT runs back at their recorded offsets.

    `exceptions` is a list of `(start, length, symbol)` runs sorted by start, in
    original-sequence coordinates. Under `delete` the symbols were removed, so the cleaned
    sequence is spliced back around each run; under `mask` a placeholder stands in their
    place, so the run is overwritten where it sits.
    """
    if not exceptions:
        return cleaned
    if ambiguity == "mask":
        out = bytearray(cleaned)
        for start, length, symbol in exceptions:
            out[start:start + length] = bytes([symbol]) * length
        return bytes(out)
    parts, cleaned_at, original_at = [], 0, 0
    for start, length, symbol in exceptions:
        kept = start - original_at
        parts.append(cleaned[cleaned_at:cleaned_at + kept])
        parts.append(bytes([symbol]) * length)
        cleaned_at += kept
        original_at = start + length
    parts.append(cleaned[cleaned_at:])
    return b"".join(parts)


def reassemble(columns: dict[int, list[bytes]], side: dict, out_dir: str) -> dict:
    """Rebuild the normalised files from cleaned column sequences and the side channel.

    `columns` maps a column number to its sequences in line order; `side` is what
    `load_sidechannel` returned. Each file's SHA-256 is compared against the one recorded at
    separation time, and the result reports which files matched.
    """
    os.makedirs(out_dir, exist_ok=True)
    structure, exceptions, headers = side["structure"], side["exceptions"], side["headers"]

    # Line order within a column is genome order, so a cursor per column walks them in step
    # with the per-file record lists.
    cursor = {int(k): 0 for k in structure["contributors"]}
    header_at = 0
    verified, failed, unchecked = [], [], []
    for i, info in enumerate(structure["files"]):
        lines = []
        for j, column in enumerate(structure["record_columns"][i]):
            header = headers[header_at]
            header_at += 1
            key = f"{i}:{j}"
            if column is None:
                seq = structure["verbatim"][key].encode("latin1")
            else:
                seq = restore_sequence(columns[column][cursor[column]], exceptions.get(key),
                                   structure.get("ambiguity", "delete"))
                cursor[column] += 1
            lines.append(header + b"\n" + seq)
        raw = b"\n".join(lines) + b"\n"
        with open(os.path.join(out_dir, info["name"]), "wb") as f:
            f.write(raw)
        if "sha256" in info:
            (verified if sha256_bytes(raw) == info["sha256"] else failed).append(info["name"])
        else:
            unchecked.append(info["name"])

    return {"verified": len(verified), "failed": failed, "unchecked": len(unchecked)}
