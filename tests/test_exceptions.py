"""The run-length exceptions blob: exact serialization and end-to-end exactness.

The blob is the compressed output's record of every non-ACGT symbol. Its format is deliberately
small (a format byte, then per file a varint record count, per record a varint run count, per
run a delta-coded start, a length and the raw symbol byte), so the tests here are the contract:
any run structure a real collection can produce must survive pack, unpack and restoration, and
a blob the decoder does not understand must be refused, not guessed at.
"""
import os
import random

import pytest
import torch

from deduct import pipeline
from deduct.encode import Config
from deduct.separate import (NON_ACGT, NON_ACGT_RUN, _pack_exceptions, _unpack_exceptions,
                             restore_sequence)

from test_end_to_end import _normalised_copy, make_corpus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set DEDUCT_MITO1000 to a rebuilt mito1000 corpus (scripts/fetch_corpus.py --corpus mito1000)
# to enable the real-data reconstruction test at the end of this file; it is skipped otherwise.
MITO1000 = os.environ.get("DEDUCT_MITO1000", "")


def runs_of(seq: bytes):
    """Extract `(start, length, symbol)` runs the way `_parse_file` does."""
    if not NON_ACGT.search(seq):
        return None
    return [(m.start(), m.end() - m.start(), m.group(1)[0])
            for m in NON_ACGT_RUN.finditer(seq)]


def record(exceptions):
    """A minimal `(column, header, seq, exceptions)` tuple; pack reads only the last slot."""
    return (1, b">h", b"", exceptions)


def test_pack_unpack_round_trip_over_the_edge_cases():
    per_file = [
        [],                                             # empty file: zero records
        [record(None)],                                 # record with no exceptions at all
        [record([(0, 1, ord("N"))])],                   # a run of length 1, at offset 0
        [record([(3, 4, ord("N")), (7, 2, ord("R"))]),  # adjacent runs: delta_start of 0
         record([(5, 1, ord("Y")), (6, 1, ord("W")), (100, 1, ord("K"))]),
         record(None)],
        [record([(2 ** 40, 2 ** 20, ord("M"))])],       # varints beyond any fixed width
    ]
    unpacked = _unpack_exceptions(_pack_exceptions(per_file), len(per_file))
    expected = {f"{i}:{j}": exc
                for i, records in enumerate(per_file)
                for j, (_c, _h, _s, exc) in enumerate(records) if exc}
    assert unpacked == expected


@pytest.mark.parametrize("ambiguity", ["delete", "mask"])
def test_random_sequences_round_trip_through_runs(ambiguity):
    """Property test: extraction, packing, unpacking and restoration compose to identity.

    The generator leans on long runs and adjacent distinct symbols because that is what the
    consensus-heavy collections actually contain; pure uniform noise would almost never
    produce either.
    """
    rng = random.Random(20260813)
    ambiguous = "NRYWKSMBDHV"
    for trial in range(300):
        parts, k = [], rng.randrange(0, 12)
        for _ in range(k):
            parts.append("".join(rng.choice("ACGT")
                                 for _ in range(rng.randrange(0, 40))))
            parts.append(rng.choice(ambiguous) * rng.randrange(1, 50))
        parts.append("".join(rng.choice("ACGT") for _ in range(rng.randrange(0, 40))))
        original = "".join(parts).encode()

        runs = runs_of(original)
        cleaned = NON_ACGT.sub(b"A" if ambiguity == "mask" else b"", original)
        per_file = [[record(runs)]]
        unpacked = _unpack_exceptions(_pack_exceptions(per_file), 1)
        restored = restore_sequence(cleaned, unpacked.get("0:0"), ambiguity)
        assert restored == original, f"trial {trial} under ambiguity={ambiguity}"


def test_long_runs_pack_small():
    """A megabase of N must cost bytes, not megabytes; that is the point of the format change."""
    per_file = [[record([(10, 1_000_000, ord("N"))])]]
    assert len(_pack_exceptions(per_file)) < 16


def test_decoder_refuses_what_it_does_not_understand():
    good = _pack_exceptions([[record([(3, 4, ord("N"))])]])

    with pytest.raises(ValueError, match="format byte"):
        _unpack_exceptions(b'{"0:0": [[3, "N"]]}', 1)    # a JSON-shaped blob
    with pytest.raises(ValueError, match="format byte"):
        _unpack_exceptions(bytes([0x02]) + good[1:], 1)
    with pytest.raises(ValueError, match="empty"):
        _unpack_exceptions(b"", 1)
    with pytest.raises(ValueError, match="truncated"):
        _unpack_exceptions(good[:-1], 1)
    with pytest.raises(ValueError, match="truncated"):
        _unpack_exceptions(bytes([0x01]), 1)             # no room for the record count
    with pytest.raises(ValueError, match="trailing"):
        _unpack_exceptions(good + b"\x00", 1)


def make_ambiguous_corpus(directory, genomes=60):
    """`make_corpus`, then ambiguity the way real collections carry it: long single-symbol
    runs, adjacent runs of distinct symbols, isolated codes, and runs touching either end."""
    make_corpus(directory, genomes=genomes)
    rng = random.Random(1453)
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        with open(path) as f:
            lines = f.read().splitlines()
        for i in range(1, len(lines), 2):
            seq = lines[i]
            roll = rng.random()
            if roll < 0.25:
                p = rng.randrange(len(seq))
                seq = seq[:p] + "N" * rng.randrange(30, 400) + seq[p:]
            elif roll < 0.45:
                p = rng.randrange(len(seq))
                seq = seq[:p] + "NNNNRRYY" + seq[p:]      # adjacent runs, several symbols
            elif roll < 0.6:
                p = rng.randrange(len(seq))
                seq = seq[:p] + rng.choice("WKMSBDHV") + seq[p:]
            elif roll < 0.7:
                seq = "N" * rng.randrange(1, 60) + seq + "N" * rng.randrange(1, 60)
            lines[i] = seq
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


@pytest.mark.parametrize("ambiguity", ["delete", "mask"])
def test_multi_symbol_corpus_is_byte_exact_end_to_end(tmp_path, ambiguity):
    src = tmp_path / "src"
    make_ambiguous_corpus(str(src))
    archive, out = tmp_path / "archive", tmp_path / "out"

    pipeline.compress(str(src), str(archive), Config(epochs=6, check_every=0, device=DEVICE),
                      workers=4, ambiguity=ambiguity)
    assert pipeline.decompress(str(archive), str(out), device=DEVICE)["failed"] == []

    reference = _normalised_copy(str(src), str(tmp_path / "normalised"))
    for name in os.listdir(reference):
        with open(os.path.join(reference, name), "rb") as a, open(out / name, "rb") as b:
            assert a.read() == b.read(), f"{name} under ambiguity={ambiguity}"


def _with_all_n_record(directory, name="g000.fna"):
    """Rewrite the first record of one file to be 100% N, keeping its length."""
    path = os.path.join(directory, name)
    with open(path) as f:
        lines = f.read().splitlines()
    lines[1] = "N" * len(lines[1])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_restore_rebuilds_an_all_n_record():
    """The exception machinery itself handles a whole-record run in both modes."""
    runs = runs_of(b"N" * 40)
    assert restore_sequence(b"", runs, "delete") == b"N" * 40
    assert restore_sequence(b"A" * 40, runs, "mask") == b"N" * 40


def test_all_n_record_is_refused_under_delete(tmp_path):
    """A record that is 100% non-ACGT must be a loud compress-time refusal under `delete`.

    `delete` strips such a record to an empty sequence, which the column files cannot carry:
    the empty line aliases with the separator and every consumer drops it. Before the read
    check, compression completed silently and decompression crashed after writing chimeric
    partial files.
    """
    src = tmp_path / "src"
    make_corpus(str(src), genomes=8, core_length=1500)
    _with_all_n_record(str(src))

    with pytest.raises(ValueError, match="ambiguity mask"):
        pipeline.compress(str(src), str(tmp_path / "archive"),
                          Config(epochs=1, check_every=0, device=DEVICE), workers=2,
                          ambiguity="delete")


def test_all_n_record_round_trips_under_mask(tmp_path):
    """The refusal message points at `mask`; this is the receipt that `mask` handles it."""
    src = tmp_path / "src"
    make_corpus(str(src), genomes=8, core_length=1500)
    _with_all_n_record(str(src))
    archive, out = tmp_path / "archive", tmp_path / "out"

    pipeline.compress(str(src), str(archive), Config(epochs=1, check_every=0, device=DEVICE),
                      workers=2, ambiguity="mask")
    assert pipeline.decompress(str(archive), str(out), device=DEVICE)["failed"] == []

    reference = _normalised_copy(str(src), str(tmp_path / "normalised"))
    for name in os.listdir(reference):
        with open(os.path.join(reference, name), "rb") as a, open(out / name, "rb") as b:
            assert a.read() == b.read(), name


@pytest.mark.skipif(not os.path.isdir(MITO1000),
                    reason="DEDUCT_MITO1000 does not name a mito1000 corpus directory")
def test_mito1000_is_byte_exact_end_to_end(tmp_path):
    """Compress-decompress cycle of the real mito1000 corpus, default configuration, digests on."""
    archive, out = tmp_path / "archive", tmp_path / "out"
    pipeline.compress(MITO1000, str(archive), Config(device=DEVICE), checksums=True)
    result = pipeline.decompress(str(archive), str(out), device=DEVICE)
    assert result["failed"] == []
    assert result["verified"] == 1000

    reference = _normalised_copy(MITO1000, str(tmp_path / "normalised"))
    for name in os.listdir(reference):
        with open(os.path.join(reference, name), "rb") as a, open(out / name, "rb") as b:
            assert a.read() == b.read(), name
