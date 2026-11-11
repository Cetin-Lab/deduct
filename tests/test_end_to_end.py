"""Compress a small corpus and decompress it back to byte-identical files."""
import os
import random

import numpy as np
import pytest
import torch

from deduct import pipeline
from deduct.common import reverse_complement, to_bytes
from deduct.encode import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_corpus(directory, genomes=80, core_length=22000, seed=7):
    """Write a corpus with one homologous family wide enough to reach the learned path.

    Each genome carries three records: a member of a shared family that has been mutated,
    rotated and sometimes reverse-complemented; a member of a second, narrower family; and an
    unrelated sequence. Some records carry an ambiguity symbol, which must survive the side
    channel.
    """
    rng = np.random.default_rng(seed)
    pick = random.Random(seed)
    os.makedirs(directory, exist_ok=True)
    core = rng.integers(1, 5, core_length).astype(np.uint8)
    minor = rng.integers(1, 5, 3000).astype(np.uint8)

    for g in range(genomes):
        records = []
        s = core.copy()
        s[rng.choice(len(s), 45, replace=False)] = rng.integers(1, 5, 45)
        if g % 7 == 0:
            p = pick.randrange(200, len(s) - 200)
            s = (np.delete(s, slice(p, p + 6)) if g % 14 == 0
                 else np.insert(s, p, rng.integers(1, 5, 4)))
        s = np.roll(s, pick.randrange(len(s)))
        if g % 3 == 0:
            s = reverse_complement(s)
        records.append(np.ascontiguousarray(s))

        t = minor.copy()
        t[rng.choice(len(t), 20, replace=False)] = rng.integers(1, 5, 20)
        records.append(np.ascontiguousarray(np.roll(t, pick.randrange(len(t)))))
        records.append(rng.integers(1, 5, pick.randrange(800, 1500)).astype(np.uint8))

        with open(os.path.join(directory, f"g{g:03d}.fna"), "w") as f:
            for k, record in enumerate(records, start=1):
                seq = to_bytes(record).decode()
                if pick.random() < 0.15:
                    p = pick.randrange(len(seq))
                    seq = seq[:p] + "N" + seq[p:]
                f.write(f">sample{g} record{k}\n{seq}\n")


def _normalised_copy(src, dst):
    from deduct.normalise import normalise_directory
    normalise_directory(src, dst)
    return dst


def test_round_trip_is_byte_exact(tmp_path):
    src = tmp_path / "src"
    make_corpus(str(src))
    archive, out = tmp_path / "archive", tmp_path / "out"

    # `checksums=True` so that decompress can compare against a stored digest; the byte
    # comparison below is the stronger check and holds either way.
    manifest = pipeline.compress(str(src), str(archive),
                                 Config(epochs=10, check_every=5, device=DEVICE), workers=4,
                                 checksums=True)
    assert manifest["coverage"]["coded_clusters"] >= 1, "the learned path never engaged"
    assert manifest["bytes"]["total"] > 0

    result = pipeline.decompress(str(archive), str(out), device=DEVICE)
    assert result["failed"] == []
    assert result["verified"] == manifest["corpus"]["genomes"]

    reference = _normalised_copy(str(src), str(tmp_path / "normalised"))
    for name in os.listdir(reference):
        with open(os.path.join(reference, name), "rb") as a, open(out / name, "rb") as b:
            assert a.read() == b.read(), name


def test_every_sequence_is_accounted_for_once(tmp_path):
    src = tmp_path / "src"
    make_corpus(str(src), genomes=70)
    manifest = pipeline.compress(str(src), str(tmp_path / "archive"),
                                 Config(epochs=6, check_every=0, device=DEVICE), workers=4)
    coverage = manifest["coverage"]
    assert coverage["neural_bases"] + coverage["fallback_bases"] == manifest["corpus"]["bases"]


def test_f_bpb_charges_the_exceptions_blob(tmp_path):
    """F-BPB is per *symbol* and its numerator carries the exceptions blob.

    The two terms are easy to lose, because the exceptions blob is written under
    `sidechannel/` alongside the headers and the record structure, which are excluded from
    both bitrates. They are not the same kind of thing: headers are container, the exception
    list is the non-ACGT symbols themselves. Charging it to the numerator and its characters
    to the denominator is what the paper's F-BPB means, and dropping either term understates
    the rate on any collection that carries ambiguity codes.
    """
    src = tmp_path / "src"
    make_corpus(str(src), genomes=70)
    manifest = pipeline.compress(str(src), str(tmp_path / "archive"),
                                 Config(epochs=6, check_every=0, device=DEVICE), workers=4)
    corpus, by = manifest["corpus"], manifest["bytes"]

    # The corpus make_corpus writes carries ambiguity symbols, so the two counts differ and
    # the test can tell the denominators apart at all.
    assert corpus["symbols"] > corpus["bases"]
    assert by["exceptions"] > 0
    assert by["exceptions"] + by["sidechannel"] == by["sidechannel_total"]
    # `total` is the compressed output as it sits on disk, the manifest included, because that
    # is what the compression ratio is a statement about.
    assert (by["clusters"] + by["fallback"] + by["sidechannel_total"] + by["manifest"]
            == by["total"])
    assert by["manifest"] == os.path.getsize(tmp_path / "archive" / "manifest.z")

    expected = (by["clusters"] + by["fallback"] + by["exceptions"]) * 8 / corpus["symbols"]
    assert manifest["f_bpb"] == pytest.approx(expected, rel=1e-6)

    # Both of the ways this drifted before: dropping the blob from the numerator, or dividing
    # by the ACGT count instead of the symbol count. Each understates the rate.
    without_blob = (by["clusters"] + by["fallback"]) * 8 / corpus["symbols"]
    over_bases = (by["clusters"] + by["fallback"] + by["exceptions"]) * 8 / corpus["bases"]
    assert without_blob < manifest["f_bpb"]
    assert over_bases > manifest["f_bpb"]


@pytest.mark.parametrize("overrides", [
    {"kernel": (5, 1)},
    {"vstride": 2},
    {"latent_channels": 2},
    {"block_height": 32},
    {"residual_coder": "dense_best", "latent_coder": "dropk_bitpack"},
])
def test_alternative_configurations_still_round_trip(tmp_path, overrides):
    """Options exposed on the command line must not break exactness."""
    src = tmp_path / "src"
    make_corpus(str(src), genomes=70, core_length=8000)
    archive, out = tmp_path / "archive", tmp_path / "out"
    config = Config(epochs=4, check_every=0, device=DEVICE, **overrides)
    pipeline.compress(str(src), str(archive), config, workers=4)
    assert pipeline.decompress(str(archive), str(out), device=DEVICE)["failed"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the flags only bind on cuDNN")
def test_decoding_pins_the_encoder_s_arithmetic(tmp_path):
    """`decode_matrix` must not inherit whatever arithmetic the calling process was left in.

    The residual carries a correction only for the cells the encoder predicted wrong, so a
    decoder running in different arithmetic can flip an argmax the compressed output has no
    correction for. Decompression happens in an isolated process, where cuDNN enables TF32 by
    default; the decoder is responsible for turning it off again.
    """
    from deduct.encode import decode_matrix

    matrix = np.random.default_rng(3).integers(1, 6, size=(64, 96), dtype=np.uint8)
    config = Config(device=DEVICE, epochs=1, check_every=0, block_height=32)
    from deduct.encode import encode_matrix
    result = encode_matrix(matrix, config, verify=False)

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = False

    decode_matrix(result.blobs, config, *matrix.shape)

    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.deterministic is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the flags only bind on cuDNN")
def test_a_guarded_archive_survives_a_change_of_arithmetic():
    """A cell decided by a narrow margin must be pinned by the residual, not by luck.

    The decoder is re-evaluated under TF32, which is the largest perturbation the hardware can
    apply to these convolutions. With a margin the reconstruction has to be unchanged; the
    companion assertion is that the margin is what did it, not that TF32 happened to be inert.
    """
    from deduct.encode import decode_matrix, encode_matrix

    matrix = np.random.default_rng(11).integers(1, 6, size=(128, 256), dtype=np.uint8)
    matrix[1:] = matrix[0]                      # a cluster the codec can nearly predict
    matrix[:, ::37] = np.random.default_rng(12).integers(
        1, 6, size=matrix[:, ::37].shape, dtype=np.uint8)

    config = Config(device=DEVICE, epochs=2, check_every=0, block_height=32,
                    decision_margin=0.1)
    result = encode_matrix(matrix, config, verify=True)
    assert result.verified

    # `decode_matrix` pins the arithmetic on entry, which is what the previous test asserts.
    # That pinning would also undo the perturbation this test exists to apply, so it is
    # suppressed for the duration: without this the decode runs with TF32 *off* and the
    # assertion below cannot fail for the reason it names.
    import deduct.encode as encode_mod
    original = encode_mod._set_arithmetic
    encode_mod._set_arithmetic = lambda *a, **k: None
    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        rebuilt = decode_matrix(result.blobs, config, *matrix.shape)
        assert torch.backends.cuda.matmul.allow_tf32 is True, "the perturbation was undone"
    finally:
        encode_mod._set_arithmetic = original
        original()
    assert np.array_equal(rebuilt, matrix)

    unguarded = encode_matrix(matrix, Config(device=DEVICE, epochs=2, check_every=0,
                                             block_height=32, decision_margin=0.0),
                              verify=True)
    assert unguarded.guarded == 0
    assert result.guarded > 0


def test_round_trip_is_byte_exact_without_checksums(tmp_path):
    """The default compressed output carries no digests, and reconstruction is unaffected.

    This is the claim the paper makes: correctness is constructive, so `--checksums` buys the
    ability to *check* it, not the ability to achieve it.
    """
    src = tmp_path / "src"
    make_corpus(str(src), genomes=60)
    archive, out = tmp_path / "archive", tmp_path / "out"

    manifest = pipeline.compress(str(src), str(archive),
                                 Config(epochs=6, check_every=0, device=DEVICE), workers=4)
    result = pipeline.decompress(str(archive), str(out), device=DEVICE)
    assert result["failed"] == []
    assert result["verified"] == 0, "no digests were stored, so nothing can be verified"
    assert result["unchecked"] == manifest["corpus"]["genomes"]

    reference = _normalised_copy(str(src), str(tmp_path / "normalised"))
    for name in os.listdir(reference):
        with open(os.path.join(reference, name), "rb") as a, open(out / name, "rb") as b:
            assert a.read() == b.read(), name


@pytest.mark.parametrize("ambiguity", ["delete", "mask"])
@pytest.mark.parametrize("tail", [
    ">sampleX empty\n>sampleY after\nACGTACGTACGTACGT\n",   # empty record mid-file
    ">sampleX truncated",                                   # bare header at EOF
])
def test_empty_record_is_refused_at_compress_time(tmp_path, ambiguity, tail):
    """A record with no sequence must fail loudly at compress time, in both ambiguity modes.

    The column files store one sequence per line, so an empty sequence aliases with the
    separator; before the read check it was silently dropped at compress time and every
    such compressed output crashed at decompression. A bare header at EOF is the shape a
    truncated download leaves behind.
    """
    src = tmp_path / "src"
    make_corpus(str(src), genomes=8, core_length=1500)
    with open(src / "g000.fna", "a") as f:
        f.write(tail)

    with pytest.raises(ValueError, match="empty sequence"):
        pipeline.compress(str(src), str(tmp_path / "archive"),
                          Config(epochs=1, check_every=0, device=DEVICE), workers=2,
                          ambiguity=ambiguity)


@pytest.mark.parametrize("payload", [b"", b"no fasta header at all\n"])
def test_a_file_with_no_records_is_refused(tmp_path, payload):
    """A file that normalises to zero records must be refused, not miscoded.

    Reassemble writes header/sequence pairs, so a recordless file would come back as a lone
    newline rather than its original bytes: silently inexact, the class the read checks
    exist to close. An empty file and a file with no header line both normalise to nothing.
    """
    src = tmp_path / "src"
    make_corpus(str(src), genomes=8, core_length=1500)
    (src / "empty.fna").write_bytes(payload)

    with pytest.raises(ValueError, match="no records"):
        pipeline.compress(str(src), str(tmp_path / "archive"),
                          Config(epochs=1, check_every=0, device=DEVICE), workers=2)


def test_a_refused_compress_does_not_poison_a_retry(tmp_path):
    """After a refusal, fixing the input and rerunning into the same output path must succeed.

    The first attempt normalises into the compressed output's work directory before the
    refusal fires; if those files outlived the failure, the retry would re-read the stale
    copy of the offending file (a phantom refusal) or silently include a genome the user
    removed.
    """
    src = tmp_path / "src"
    make_corpus(str(src), genomes=8, core_length=1500)
    with open(src / "bad.fna", "w") as f:
        f.write(">bad only a header\n")
    archive = tmp_path / "archive"

    with pytest.raises(ValueError, match="empty sequence"):
        pipeline.compress(str(src), str(archive),
                          Config(epochs=1, check_every=0, device=DEVICE), workers=2)

    os.remove(src / "bad.fna")
    manifest = pipeline.compress(str(src), str(archive),
                                 Config(epochs=1, check_every=0, device=DEVICE), workers=2)
    assert manifest["corpus"]["genomes"] == 8, "the removed file must not be re-read"


@pytest.mark.parametrize("names", [("a.fa", "a.fasta"), ("x.fna", "x.fna.gz")])
def test_colliding_input_stems_are_refused(tmp_path, names):
    """Two inputs whose stems collide must be refused, not silently merged.

    Every accepted suffix normalises to `stem.fna`, so before the check the later file in
    sorted order overwrote the earlier one's normalised output: a whole input vanished while
    compression and verify both reported success.
    """
    import gzip

    from deduct.normalise import normalise_directory

    src = tmp_path / "src"
    os.makedirs(src)
    payloads = [b">recA first\nACGTACGTACGTACGT\n", b">recB second\nGGGGCCCCAAAATTTT\n"]
    for name, payload in zip(names, payloads):
        if name.endswith(".gz"):
            with gzip.open(src / name, "wb") as f:
                f.write(payload)
        else:
            (src / name).write_bytes(payload)

    dst = tmp_path / "normalised"
    with pytest.raises(ValueError) as excinfo:
        normalise_directory(str(src), str(dst))
    # The full phrase, not substrings: `a.fa` is a substring of `a.fasta`, so matching the
    # names one by one could not tell a message naming both files from one naming just one.
    assert f"{names[0]} and {names[1]} both normalise to" in str(excinfo.value)
    assert os.listdir(dst) == [], "a refused input set must leave nothing behind"


def test_verify_without_digests_is_loud(tmp_path, capsys):
    """`verify` must not exit 0 when there was no digest to compare.

    The default compressed output stores no checksums, so `failed` is empty whatever
    happened; exit 0 on that basis would let "verify passed" mean "nothing was checked". The
    contract: exit 3 plus a stderr warning without digests (3, not 2, which argparse uses for
    usage errors), exit 0 only when digests were compared.
    """
    from deduct.cli import main

    src = tmp_path / "src"
    make_corpus(str(src), genomes=8, core_length=1500)

    archive = tmp_path / "archive"
    pipeline.compress(str(src), str(archive),
                      Config(epochs=1, check_every=0, device=DEVICE), workers=2)
    rc = main(["verify", str(archive), "--device", DEVICE])
    assert rc == 3
    assert "no checksums" in capsys.readouterr().err

    checked = tmp_path / "archive_checked"
    pipeline.compress(str(src), str(checked),
                      Config(epochs=1, check_every=0, device=DEVICE), workers=2,
                      checksums=True)
    assert main(["verify", str(checked), "--device", DEVICE]) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("ambiguity", ["delete", "mask"])
def test_ambiguity_handling_is_invertible(tmp_path, ambiguity):
    """Both ways of removing non-ACGT symbols must reconstruct the input exactly.

    `delete` shortens the sequence and shifts every coordinate after the symbol; `mask` keeps
    the length and substitutes a placeholder. They differ in what the clustering stage then
    sees, not in what comes back out.
    """
    src = tmp_path / "src"
    make_corpus(str(src), genomes=60)
    archive, out = tmp_path / "archive", tmp_path / "out"

    pipeline.compress(str(src), str(archive),
                      Config(epochs=6, check_every=0, device=DEVICE), workers=4,
                      ambiguity=ambiguity)
    result = pipeline.decompress(str(archive), str(out), device=DEVICE)
    assert result["failed"] == []

    reference = _normalised_copy(str(src), str(tmp_path / "normalised"))
    for name in os.listdir(reference):
        with open(os.path.join(reference, name), "rb") as a, open(out / name, "rb") as b:
            assert a.read() == b.read(), f"{name} under ambiguity={ambiguity}"
