"""Rewrite a directory of FASTA files into the single-line, indexed form the pipeline reads.

Every record becomes exactly two lines:

    >{original header} contig{k}
    {sequence, upper-cased, on one line}

where `k` is the 1-based index of the record within its file. Two things follow. Multi-line
FASTA becomes one line per record, which is what the separation stage parses; and the record
index is written into the header, which is where the separation stage reads a record's column
from.

Reconstruction is verified against these normalised files, not against the originals as
downloaded. Upper-casing is the one transformation that loses information (soft-masking is not
recoverable afterwards), so a corpus that carries meaningful lower-case must be normalised
differently or not at all.
"""
from __future__ import annotations

import gzip
import os

SUFFIXES = (".fna", ".fa", ".fasta", ".fna.gz", ".fa.gz", ".fasta.gz")


def read_records(path: str):
    """Yield `(header, sequence)` for a FASTA file, joining wrapped sequence lines.

    The header is returned without its leading '>'. Gzip is detected by extension.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        header, chunks = None, []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            elif line:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def normalise_file(src: str, dst: str) -> int:
    """Write one normalised file and return the number of records in it."""
    with open(dst, "w") as out:
        n = 0
        for n, (header, seq) in enumerate(read_records(src), start=1):
            out.write(f">{header} contig{n}\n{seq.upper()}\n")
        return n


def output_name(name: str) -> str:
    """The normalised name an input file maps to: its stem with a `.fna` extension."""
    stem = name
    for suffix in (".gz", ".fasta", ".fa", ".fna"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem + ".fna"


def normalise_directory(src_dir: str, dst_dir: str, limit: int | None = None,
                        progress=None) -> list[str]:
    """Normalise every FASTA file in `src_dir` into `dst_dir`.

    Returns the destination file names in sorted order, which is the order every later stage
    uses to index genomes. Output names keep the input stem with a `.fna` extension, so two
    inputs can map to the same name (`a.fa` and `a.fasta`; `x.fna` and `x.fna.gz`). That
    would make the later file silently replace the earlier one's output, so a colliding
    input set is refused before anything is written.
    """
    os.makedirs(dst_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(src_dir) if f.endswith(SUFFIXES))
    if limit:
        names = names[:limit]

    by_output: dict[str, list[str]] = {}
    for name in names:
        by_output.setdefault(output_name(name), []).append(name)
    collisions = [(out, srcs) for out, srcs in sorted(by_output.items()) if len(srcs) > 1]
    if collisions:
        detail = "; ".join(f"{' and '.join(srcs)} "
                           f"{'both' if len(srcs) == 2 else 'all'} normalise to {out}"
                           for out, srcs in collisions)
        raise ValueError(f"input file names collide after normalisation: {detail}. "
                         f"Nothing was written; rename the inputs so every file keeps a "
                         f"unique stem")

    written = []
    for i, name in enumerate(names, start=1):
        out_name = output_name(name)
        normalise_file(os.path.join(src_dir, name), os.path.join(dst_dir, out_name))
        written.append(out_name)
        if progress:
            progress(i, len(names), out_name)
    return written
