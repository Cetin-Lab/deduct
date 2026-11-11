"""End-to-end compression and decompression.

Compression runs the five stages in order (normalise, separate, cluster, align, code) and
writes a compressed output directory:

    manifest.z              configuration, per-stage statistics, and the cluster list
    sidechannel/            headers, structure and ambiguity positions
    clusters/{name}/        weights, latent, residual and per-row alignment items
    fallback/               everything the learned path did not take, as one pooled blob

Decompression reads only that directory. Each cluster's matrix is rebuilt from its blobs, each
member is un-aligned back to its own sequence, the fallback pool supplies the rest, and the side
channel turns the resulting contig columns back into files.

Every sequence in the collection is placed exactly once: it is a member of one coded
cluster or it is in the fallback pool. `compress` reports the split, because a bitrate without
the share of the corpus it covers is not interpretable.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import warnings

from . import align as align_mod
from . import encode as encode_mod
from . import fallback as fallback_mod
from . import group as group_mod
from . import normalise as normalise_mod
from . import separate as separate_mod
from .common import to_bytes, gzip_compress, gzip_decompress


def _write_blob(path: str, blob: bytes) -> int:
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


def _read_blob(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


_DONE = object()


def _aligned_ahead(candidates, column_cache, identity_threshold, workers, device, depth=1):
    """Yield each candidate's aligned matrix, with alignment running ahead of the consumer.

    Alignment and coding use different resources for most of their time (the aligner spends
    the bulk of it in edlib on CPU threads, the coder in training on the GPU), so running them
    in lockstep leaves one of the two idle throughout. A worker thread aligns ahead into a
    bounded queue while the caller codes what is already aligned.

    `depth` bounds how many aligned matrices the queue holds, and therefore the memory. It is 1
    because the bound has to hold for the worst cluster, not the typical one: a bacterial
    cluster's matrix is a few hundred megabytes, but a collection of near-identical viral
    genomes can put most of its members into a single cluster whose matrix runs to gigabytes,
    and a deeper queue there costs a multiple of that. One in the queue still covers the
    producer and the consumer, which is what the overlap needs. Exceptions raised while aligning
    are re-raised in the caller's thread.

    Yields `(name, column, aligned, error)`, where exactly one of `aligned` and `error` is set.
    """
    pending: queue.Queue = queue.Queue(maxsize=depth)

    def produce():
        try:
            for _size, column, index, cluster in candidates:
                name = f"c{column}_g{index}"
                try:
                    aligned = align_mod.align_cluster(
                        column_cache[column], cluster["members"],
                        identity_threshold=identity_threshold, workers=workers, device=device)
                except ValueError as exc:
                    pending.put((name, column, None, str(exc)))
                    continue
                pending.put((name, column, aligned, None))
        except BaseException as exc:                      # noqa: BLE001 - re-raised below
            pending.put(exc)
        finally:
            pending.put(_DONE)

    worker = threading.Thread(target=produce, name="align-ahead", daemon=True)
    worker.start()
    while True:
        item = pending.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def compress(src: str, archive: str, config: encode_mod.Config, limit: int | None = None,
             accept: float = 0.6, max_clusters: int = 30, min_members: int = 64,
             identity_threshold: float = 0.35, workers: int = 24,
             work: str | None = None, keep_work: bool = False,
             checksums: bool = False, ambiguity: str = "delete") -> dict:
    """Compress a directory of FASTA files into a compressed output directory."""
    started = time.time()
    os.makedirs(archive, exist_ok=True)
    work = work or os.path.join(archive, "_work")
    normalised = os.path.join(work, "normalised")
    os.makedirs(work, exist_ok=True)
    # `separate` reads every .fna in this directory, so anything a previous interrupted or
    # refused run left behind would be silently re-read alongside the current corpus.
    shutil.rmtree(normalised, ignore_errors=True)

    t0 = time.time()
    try:
        normalise_mod.normalise_directory(src, normalised, limit=limit)
        t_normalise = time.time() - t0

        t0 = time.time()
        # The side channel is text and compresses better and far faster at a lower level than
        # the symbol planes do; separate.SIDE_CHANNEL_LEVEL records the measurement.
        separation = separate_mod.separate(normalised, work, workers=workers,
                                           checksums=checksums, ambiguity=ambiguity)
    except Exception:
        # A read refusal happens before the compressed output exists; the populated work
        # directory must not outlive it, or a retry into the same output path reads the leftovers.
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
        raise
    t_separate = time.time() - t0

    side_src = os.path.join(work, "sidechannel")
    side_dst = os.path.join(archive, "sidechannel")
    os.makedirs(side_dst, exist_ok=True)
    for name in os.listdir(side_src):
        shutil.copyfile(os.path.join(side_src, name), os.path.join(side_dst, name))
    sidechannel_bytes = separation["sidechannel_total"]
    # The exceptions blob lives under sidechannel/ because that is where the separation stage
    # writes it, but it is not *container*: it holds the non-ACGT symbols themselves, which
    # are sequence the compressed output has to carry. It is therefore charged inside F-BPB
    # while the headers and the record structure are not, and the two have to be split apart
    # here.
    exceptions_bytes = separation["sidechannel_bytes"]["exceptions.z"]
    container_bytes = sidechannel_bytes - exceptions_bytes

    column_dir = os.path.join(work, "columns")
    n_columns = separation["columns"]

    # Cluster every column, then take the largest candidates corpus-wide.
    t0 = time.time()
    candidates = []
    column_cache = {}
    for column in range(1, n_columns + 1):
        path = os.path.join(column_dir, f"contig{column}.txt")
        if not os.path.exists(path) or os.path.getsize(path) <= 1:
            continue
        sequences = group_mod.load_column(path)
        if not sequences:
            continue
        column_cache[column] = sequences
        # A column with fewer sequences than the admission threshold cannot produce an
        # admissible cluster, so correlating it would be work whose result is discarded.
        if len(sequences) < min_members:
            continue
        grouping = group_mod.cluster_column(sequences, device=config.device, accept=accept,
                                            max_clusters=max_clusters)
        for index, cluster in enumerate(grouping["clusters"]):
            if len(cluster["members"]) >= min_members:
                candidates.append((len(cluster["members"]), column, index, cluster))
    t_cluster = time.time() - t0
    candidates.sort(key=lambda c: -c[0])

    # Align and code each candidate.
    cluster_dir = os.path.join(archive, "clusters")
    os.makedirs(cluster_dir, exist_ok=True)
    coded, skipped = [], []
    taken: dict[int, set[int]] = {}
    t_code = 0.0
    t_align_start = time.time()
    for name, column, aligned, error in _aligned_ahead(
            candidates, column_cache, identity_threshold, workers, config.device):
        if error is not None:
            skipped.append({"name": name, "reason": error})
            continue

        matrix = aligned["matrix"]
        if matrix.shape[0] < min_members or matrix.shape[1] < config.tile_width:
            skipped.append({"name": name, "reason": "too small after alignment"})
            continue

        t0 = time.time()
        result = encode_mod.encode_matrix(matrix, config)
        t_code += time.time() - t0
        if not result.verified:
            raise AssertionError(f"cluster {name} did not reconstruct exactly")

        out = os.path.join(cluster_dir, name)
        os.makedirs(out, exist_ok=True)
        sizes = {k: _write_blob(os.path.join(out, f"{k}.z"), v)
                 for k, v in result.blobs.items()}
        rows_blob = gzip_compress(json.dumps({
            "column": column, "rows": aligned["rows"], "height": int(matrix.shape[0]),
            "width": int(matrix.shape[1])}, separators=(",", ":")).encode(),
            config.gzip_level)
        sizes["rows"] = _write_blob(os.path.join(out, "rows.z"), rows_blob)

        taken.setdefault(column, set()).update(r["index"] for r in aligned["rows"])
        # Coverage counts the members' own lengths, not the matrix's non-gap cells. A base
        # that the consensus has no column for is carried as an insertion in the row metadata,
        # so it is coded here but never appears in the matrix; counting cells would leave it
        # out of both the learned path's share and the fallback's.
        coded.append({
            "name": name, "column": column, "height": int(matrix.shape[0]),
            "width": int(matrix.shape[1]), "bytes": sizes,
            "total": sum(sizes.values()), "accuracy": result.accuracy,
            "epoch": result.epoch,
            "bases": sum(r["length"] for r in aligned["rows"]),
            "matrix_cells": int(matrix.size), "matrix_gaps": result.gaps,
            "insertions": sum(len(r["insertions"]) for r in aligned["rows"]),
            "frequencies_used": result.frequencies_used, "frequencies": result.frequencies,
            "train_seconds": result.train_seconds, "stopped_early": result.stopped_early,
            "curve": result.curve,
            "d_bpb": round(sum(sizes.values()) * 8
                           / max(1, sum(r["length"] for r in aligned["rows"])), 6),
        })

    # Everything else goes to the pooled fallback.
    t0 = time.time()
    leftovers = []
    for column, sequences in column_cache.items():
        used = taken.get(column, ())
        for line, seq in enumerate(sequences, start=1):
            if line not in used:
                leftovers.append((f"{column}:{line}", to_bytes(seq)))
    pool = fallback_mod.build_pool(leftovers, level=config.gzip_level)
    fallback_dir = os.path.join(archive, "fallback")
    os.makedirs(fallback_dir, exist_ok=True)
    _write_blob(os.path.join(fallback_dir, "pool.z"), pool["pool"])
    _write_blob(os.path.join(fallback_dir, "index.z"), pool["index"])
    t_fallback = time.time() - t0
    t_align_and_code = time.time() - t_align_start

    neural_bases = sum(c["bases"] for c in coded)
    fallback_bases = sum(len(s) for _k, s in leftovers)
    cluster_bytes = sum(c["total"] for c in coded)
    source_bytes = sum(os.path.getsize(os.path.join(normalised, f))
                       for f in os.listdir(normalised))
    manifest = {
        "config": encode_mod.asdict(config),
        "environment": describe_environment(),
        "front_end": {"accept": accept, "max_clusters": max_clusters,
                      "min_members": min_members,
                      "identity_threshold": identity_threshold},
        "corpus": {"genomes": separation["genomes"], "columns": n_columns,
                   "records": separation["records"], "bases": separation["bases"],
                   "symbols": separation["symbols"]},
        "clusters": coded,
        "skipped": skipped,
        "coverage": {
            "neural_bases": neural_bases,
            "fallback_bases": fallback_bases,
            "neural_share": round(neural_bases / max(1, neural_bases + fallback_bases), 4),
            "coded_clusters": len(coded),
            "candidate_clusters": len(candidates),
        },
        # `exceptions` and `sidechannel` sum to what is on disk under sidechannel/; they are
        # reported apart because only the first is charged to a bitrate.
        "bytes": {"clusters": cluster_bytes, "fallback": pool["total"],
                  "exceptions": exceptions_bytes, "sidechannel": container_bytes,
                  "sidechannel_total": sidechannel_bytes, "source": source_bytes},
        "seconds": {"normalise": round(t_normalise, 1), "separate": round(t_separate, 1),
                    "cluster": round(t_cluster, 1),
                    # Alignment overlaps coding, so the two do not add up to the elapsed time
                    # they share; `align_and_code` is that shared span.
                    "code": round(t_code, 1),
                    "align_and_code": round(t_align_and_code - t_fallback, 1),
                    "fallback": round(t_fallback, 1),
                    "total": round(time.time() - started, 1)},
    }
    # Everything on disk except the manifest, whose own size is added after it is written.
    blob_total = cluster_bytes + pool["total"] + sidechannel_bytes
    # Four figures, because one does not carry the information the others do.
    #
    #   D-BPB  bits per base over the bases the trained coder received, counting only what
    #          that path wrote. This is the rate of the method itself.
    #   F-BPB  bits per input *symbol* over the whole collection, counting the learned path,
    #          the fallback and the exceptions blob. This is the rate of the sequence data as a
    #          whole, and the denominator is every character read rather than only the ACGT.
    #   ratio and space saving are over the whole compressed output against the source
    #          files, so they include the side channel: reconstructing a file needs it.
    #
    # The side channel proper (headers and record structure) is excluded from both
    # bitrates and reported on its own. Its size scales with the number of records rather
    # than the number of bases, so on a collection of short records it would otherwise
    # dominate a figure whose units are per base. The exceptions blob is the exception to
    # that, in both senses: it is stored alongside the side channel but counted as sequence.
    # Per-column calculation, for the breakdown tables: how many sequences a contig column
    # held, how many clusters it produced, how many of those the trained coder took, and how
    # its bases divided between the two paths.
    per_column = {}
    for column, sequences in column_cache.items():
        used = taken.get(column, set())
        entry = per_column.setdefault(column, {
            "sequences": len(sequences), "clusters": 0, "coded": 0,
            "neural_bases": 0, "fallback_bases": 0, "cluster_bytes": 0})
        entry["fallback_bases"] = sum(len(s) for i, s in enumerate(sequences, start=1)
                                      if i not in used)
    for _size, column, _index, cluster in candidates:
        per_column.setdefault(column, {"sequences": 0, "clusters": 0, "coded": 0,
                                       "neural_bases": 0, "fallback_bases": 0,
                                       "cluster_bytes": 0})["clusters"] += 1
    for entry in coded:
        column = per_column[entry["column"]]
        column["coded"] += 1
        column["neural_bases"] += entry["bases"]
        column["cluster_bytes"] += entry["total"]
    for column, entry in per_column.items():
        entry["d_bpb"] = (round(entry["cluster_bytes"] * 8 / entry["neural_bases"], 6)
                          if entry["neural_bases"] else None)
    manifest["per_column"] = {str(k): v for k, v in sorted(per_column.items())}

    # D-BPB is per base over the bases the trained coder received; F-BPB is per *symbol* over
    # the whole collection, and its numerator carries the exceptions blob because those bytes
    # encode sequence. The denominators are unrelated quantities, not two roundings of one: the
    # first is a subset of the collection (whatever the front end admitted), the second is a
    # superset of its ACGT (every character read, ambiguity codes included). They coincide only
    # when coverage is 100% and the collection is pure ACGT.
    manifest["d_bpb"] = round(cluster_bytes * 8 / max(1, neural_bases), 6)
    manifest["f_bpb"] = round((cluster_bytes + pool["total"] + exceptions_bytes) * 8
                              / max(1, separation["symbols"]), 6)

    # The ratio and the space saving are statements about *files*: the original bytes over the
    # compressed output as it sits on disk, manifest included. A number that counts the manifest
    # cannot also live inside it (writing it changes the size it reports), so the manifest stores
    # the component byte counts, which are facts, and the two derived figures are computed
    # after the write from what is actually on disk. `compress` returns them; the stored copy
    # carries `bytes.blobs` so that anyone reading the compressed output can recompute them the
    # same way.
    manifest["bytes"]["blobs"] = blob_total
    _write_blob(os.path.join(archive, "manifest.z"),
                gzip_compress(json.dumps(manifest).encode(), config.gzip_level))

    manifest_bytes = os.path.getsize(os.path.join(archive, "manifest.z"))
    total = blob_total + manifest_bytes
    manifest["bytes"]["manifest"] = manifest_bytes
    manifest["bytes"]["total"] = total
    manifest["ratio"] = round(source_bytes / max(1, total), 2)
    manifest["space_saving"] = round(100 * (source_bytes - total) / max(1, source_bytes), 4)
    manifest["bpb_including_sidechannel"] = round(total * 8 / max(1, separation["symbols"]), 6)

    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return manifest


def describe_environment() -> dict:
    """The parts of the machine that decide what the decoder's convolutions evaluate to.

    Recorded so that a decompression somewhere else can say what changed. It is a report, not a
    guarantee: what keeps the compressed output readable on other hardware is
    `Config.decision_margin`, which corrects every cell whose argmax was too close to call.
    This block is how a decoder tells the user which of the two it is relying on.
    """
    import torch
    device = None
    if torch.cuda.is_available():
        try:
            device = torch.cuda.get_device_name(0)
        except Exception:                                # a driver that will not answer
            device = None
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device_name": device}


def compare_environment(manifest: dict) -> list[str]:
    """Fields of the recorded environment that differ from this machine's, as readable lines."""
    written = manifest.get("environment")
    if not written:
        return ["the compressed output records no environment; it predates that field"]
    here = describe_environment()
    return [f"{k}: written with {written.get(k)!r}, decoding with {here.get(k)!r}"
            for k in here if written.get(k) != here.get(k)]


def read_manifest(archive: str) -> dict:
    return json.loads(gzip_decompress(_read_blob(os.path.join(archive, "manifest.z"))))


def _rebuild_columns(archive: str, manifest: dict, device: str) -> dict[int, list[bytes]]:
    """Recover every contig column's sequences, in line order."""
    config = encode_mod.Config(**{**manifest["config"], "device": device})
    config.kernel = tuple(config.kernel)

    differences = compare_environment(manifest)
    if differences:
        margin = manifest["config"].get("decision_margin", 0)
        note = (f"the decision margin of {margin} covers arithmetic that moves a logit by less "
                f"than {margin / 2}" if margin else
                "this compressed output was written without a decision margin, so it relies "
                "on the arithmetic being reproduced exactly")
        warnings.warn("decoding in a different environment from the one that wrote this "
                      "compressed output:\n  " + "\n  ".join(differences) + f"\n{note}",
                      stacklevel=2)

    columns: dict[int, dict[int, bytes]] = {}
    for entry in manifest["clusters"]:
        out = os.path.join(archive, "clusters", entry["name"])
        blobs = {k: _read_blob(os.path.join(out, f"{k}.z"))
                 for k in ("weights", "latent", "residual")}
        rows = json.loads(gzip_decompress(_read_blob(os.path.join(out, "rows.z"))))
        matrix = encode_mod.decode_matrix(blobs, config, rows["height"], rows["width"])
        column = rows["column"]
        for r, meta in enumerate(rows["rows"]):
            seq = align_mod.reconstruct_member(meta["offset"], meta["strand"], matrix[r],
                                               meta["insertions"])
            columns.setdefault(column, {})[meta["index"]] = to_bytes(seq)

    for key, seq in fallback_mod.read_pool({
            "pool": _read_blob(os.path.join(archive, "fallback", "pool.z")),
            "index": _read_blob(os.path.join(archive, "fallback", "index.z"))}).items():
        column, line = (int(x) for x in key.split(":"))
        columns.setdefault(column, {})[line] = seq

    return {column: [lines[i] for i in sorted(lines)] for column, lines in columns.items()}


def decompress(archive: str, out: str, device: str = "cuda") -> dict:
    """Rebuild the normalised files from a compressed output."""
    started = time.time()
    manifest = read_manifest(archive)
    columns = _rebuild_columns(archive, manifest, device)
    side = separate_mod.load_sidechannel(os.path.join(archive, "sidechannel"))
    result = separate_mod.reassemble(columns, side, out)
    result["seconds"] = round(time.time() - started, 1)
    return result


def verify(archive: str, device: str = "cuda") -> dict:
    """Decompress into a temporary directory and check every file's checksum."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        return decompress(archive, tmp, device=device)
