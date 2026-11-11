#!/usr/bin/env python3
"""Rebuild any of the paper's public corpora from NCBI.

Two ways in, and they are not equivalent:

  --corpus NAME      fetch the exact records the paper used, by accession list.
  --query '...'      re-run the original search, which returns whatever NCBI holds today.

Prefer ``--corpus``. The accession lists under ``corpora/`` pin the corpus to the
records that produced the published numbers; a query re-run does not, because the
databases grow and a search that returned N records in 2026 returns more later. The
lists are how the queries below were resolved on the fetch dates recorded in
``corpora/PROVENANCE.md``.

    python fetch_corpus.py --corpus mito1000 --out ./mito1000
    python fetch_corpus.py --corpus sars_2021 --out ./sars_2021
    python fetch_corpus.py --verify ./mito1000 --corpus mito1000

Records are written one FASTA file per accession, sequence upper-cased on a single
line, the same form the corpora were in when they were compressed. ``deduct
compress`` normalises again on read, so this only has to be close; it is exact so
that ``--verify`` can be a byte comparison rather than a sequence comparison.

NCBI asks that automated requests identify themselves and stay under three per
second without an API key. Set NCBI_API_KEY to raise the ceiling to ten.
"""
import argparse
import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"
TOOL = "deduct-corpus-builder"
HERE = os.path.dirname(os.path.abspath(__file__))
LISTS = os.path.join(HERE, "corpora")

# The searches that produced the released accession lists. Kept here rather than in
# prose so that what the paper describes and what the code does cannot drift apart.
SARS_BASE = ('"Severe acute respiratory syndrome coronavirus 2"[Organism]'
             ' AND "complete genome"[Title] AND 29000:30000[SLEN]')
MITO_BASE = ('"Homo sapiens"[Organism] AND mitochondrion[Filter]'
             ' AND "complete genome"[Title] AND 16000:17000[SLEN]')

QUERIES = {
    "mito1000": (MITO_BASE, 1000),
    "mito_full": (MITO_BASE, 67190),
    **{f"sars_{y}": (f'{SARS_BASE} AND ("{y}/01/01"[PDAT] : "{y}/12/31"[PDAT])', 50000)
       for y in range(2020, 2027)},
}

# ecoli1 and senterica are assembled genomes, not nuccore records, so they come
# through the Datasets API instead. They were selected from NCBI Pathogen Detection
# rather than by an Entrez query; the accession list is the only definition of them.
ASSEMBLY_CORPORA = {"ecoli1", "senterica"}


def _api_key_params() -> dict:
    key = os.environ.get("NCBI_API_KEY", "").strip()
    return {"api_key": key} if key else {}


def _pause() -> float:
    return 0.11 if os.environ.get("NCBI_API_KEY", "").strip() else 0.4


def _get(url: str, retries: int = 5) -> bytes:
    """GET with backoff. NCBI returns 429 and 5xx under load; both are transient."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  HTTP {exc.code}, retry in {wait}s", flush=True)
            time.sleep(wait)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry in {wait}s after {type(exc).__name__}: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _eutils(endpoint: str, params: dict) -> str:
    params = {**params, "tool": TOOL, **_api_key_params()}
    email = os.environ.get("NCBI_EMAIL", "").strip()
    if email:
        params["email"] = email
    url = f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"
    return _get(url).decode("utf-8", "replace")


def read_accessions(path: str) -> list:
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()
                and not line.startswith("#")]


def _write_record(out_dir: str, header: str, sequence: str, suffix: str) -> str:
    accession = header[1:].split()[0].replace("/", "_")
    name = f"{accession}{suffix}.fna"
    with open(os.path.join(out_dir, name), "w") as handle:
        handle.write(f"{header}\n{sequence}\n")
    return name


def _split_fasta(text: str):
    for record in text.split("\n>"):
        record = record.strip()
        if not record:
            continue
        if not record.startswith(">"):
            record = ">" + record
        header, _, body = record.partition("\n")
        sequence = body.replace("\n", "").upper()
        if sequence:
            yield header, sequence


def fetch_nuccore(accessions: list, out_dir: str, suffix: str, batch: int = 200) -> int:
    """Pull GenBank nucleotide records by accession, in batches."""
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for start in range(0, len(accessions), batch):
        chunk = accessions[start:start + batch]
        text = _eutils("efetch.fcgi", {"db": "nuccore", "id": ",".join(chunk),
                                       "rettype": "fasta", "retmode": "text"})
        for header, sequence in _split_fasta(text):
            _write_record(out_dir, header, sequence, suffix)
            written += 1
        print(f"  {written:,}/{len(accessions):,}", flush=True)
        time.sleep(_pause())
    return written


def fetch_by_query(query: str, count: int, out_dir: str, suffix: str,
                   batch: int = 200) -> int:
    """Re-run the original search. Order is NCBI's, so this is not reproducible."""
    os.makedirs(out_dir, exist_ok=True)
    text = _eutils("esearch.fcgi", {"db": "nuccore", "term": query, "retmax": 0,
                                    "usehistory": "y"})
    available = int(re.search(r"<Count>(\d+)</Count>", text).group(1))
    web = re.search(r"<WebEnv>(\S+)</WebEnv>", text).group(1)
    key = re.search(r"<QueryKey>(\d+)</QueryKey>", text).group(1)
    total = min(available, count)
    print(f"query matches {available:,} records; taking the first {total:,} "
          f"in NCBI's return order", flush=True)
    if available > count:
        print("  NOTE: the cap binds. Which records you get depends on NCBI's "
              "ordering today, so this will NOT reproduce the paper's corpus. "
              "Use --corpus for that.", flush=True)
    written = 0
    for start in range(0, total, batch):
        take = min(batch, total - start)
        text = _eutils("efetch.fcgi", {"db": "nuccore", "WebEnv": web,
                                       "query_key": key, "retstart": start,
                                       "retmax": take, "rettype": "fasta",
                                       "retmode": "text"})
        for header, sequence in _split_fasta(text):
            _write_record(out_dir, header, sequence, suffix)
            written += 1
        print(f"  {written:,}/{total:,}", flush=True)
        time.sleep(_pause())
    return written


def fetch_assemblies(accessions: list, out_dir: str, batch: int = 20) -> int:
    """Pull assembled genomes through the Datasets API, one zip per batch.

    Assemblies arrive as multi-record FASTA (one record per contig) and are kept
    that way: the contig structure is what the front end decomposes, so flattening
    it here would destroy the input the method is about.
    """
    import io
    import zipfile

    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for start in range(0, len(accessions), batch):
        chunk = accessions[start:start + batch]
        url = (f"{DATASETS}/genome/accession/{','.join(chunk)}/download"
               f"?include_annotation_type=GENOME_FASTA")
        key = os.environ.get("NCBI_API_KEY", "").strip()
        if key:
            url += f"&api_key={key}"
        payload = _get(url)
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            for name in bundle.namelist():
                if not name.endswith(("_genomic.fna", ".fna")):
                    continue
                if "/data/" not in name:
                    continue
                body = bundle.read(name).decode("utf-8", "replace")
                target = os.path.join(out_dir, os.path.basename(name))
                with open(target, "w") as handle:
                    for header, sequence in _split_fasta(body):
                        handle.write(f"{header}\n{sequence}\n")
                written += 1
        print(f"  {written:,}/{len(accessions):,}", flush=True)
        time.sleep(_pause())
    return written


def corpus_digest(path: str) -> tuple:
    """(file count, total bytes, aggregate sha256) over a corpus directory.

    The aggregate is sha256 of the sorted "<per-file sha256>  <name>\\n" lines, so it
    is independent of directory order and changes if any byte or any name changes.
    """
    names = sorted(f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
    agg = hashlib.sha256()
    total = 0
    for name in names:
        full = os.path.join(path, name)
        total += os.path.getsize(full)
        agg.update(f"{file_sha256(full)}  {name}\n".encode())
    return len(names), total, agg.hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checksums() -> dict:
    path = os.path.join(LISTS, "CHECKSUMS.txt")
    table = {}
    if not os.path.exists(path):
        return table
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            name, files, size, digest = line.split()
            table[name] = (int(files), int(size), digest)
    return table


def do_verify(out_dir: str, corpus: str) -> int:
    """Check a rebuilt corpus against the digest recorded when the paper was written."""
    expected = load_checksums().get(corpus)
    if expected is None:
        wanted = read_accessions(os.path.join(LISTS, f"{corpus}.txt"))
        present = [f for f in os.listdir(out_dir) if f.endswith(".fna")]
        print(f"no digest recorded for {corpus}; comparing counts only: "
              f"expected {len(wanted):,}, found {len(present):,}")
        return 0 if len(present) == len(wanted) else 1

    print(f"hashing {out_dir} ...", flush=True)
    files, size, digest = corpus_digest(out_dir)
    ok = (files, size, digest) == expected
    print(f"  files  {files:,}" + ("" if files == expected[0] else f"   expected {expected[0]:,}"))
    print(f"  bytes  {size:,}" + ("" if size == expected[1] else f"   expected {expected[1]:,}"))
    print(f"  sha256 {digest}")
    if ok:
        print("MATCH: this is the corpus the paper used.")
        return 0
    print(f"  expected {expected[2]}")
    print("MISMATCH. NCBI revises and suppresses records over time, so a corpus rebuilt\n"
          "later is not guaranteed to be the one the paper used. Numbers computed on it\n"
          "will be close but need not be identical.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=sorted(set(QUERIES) | ASSEMBLY_CORPORA),
                    help="fetch this corpus by its released accession list")
    ap.add_argument("--accessions", help="fetch an arbitrary accession list instead")
    ap.add_argument("--query", help="re-run a search (not reproducible; see --corpus)")
    ap.add_argument("--count", type=int, default=0, help="cap for --query")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--verify", metavar="DIR",
                    help="check a rebuilt corpus against the released checksums")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="show the available corpora")
    a = ap.parse_args()

    if a.list:
        print(f"{'corpus':12} {'records':>9}  source")
        for name in sorted(set(QUERIES) | ASSEMBLY_CORPORA):
            path = os.path.join(LISTS, f"{name}.txt")
            n = len(read_accessions(path)) if os.path.exists(path) else 0
            kind = "assembly" if name in ASSEMBLY_CORPORA else "nuccore"
            print(f"{name:12} {n:>9,}  {kind}")
        return 0

    if a.verify:
        if not a.corpus:
            ap.error("--verify needs --corpus")
        return do_verify(a.verify, a.corpus)

    if not a.out:
        ap.error("--out is required")

    if a.corpus:
        path = os.path.join(LISTS, f"{a.corpus}.txt")
        if not os.path.exists(path):
            ap.error(f"no accession list at {path}")
        accessions = read_accessions(path)
        print(f"{a.corpus}: {len(accessions):,} accessions -> {a.out}", flush=True)
        if a.corpus in ASSEMBLY_CORPORA:
            written = fetch_assemblies(accessions, a.out, a.batch or 20)
        else:
            # mito1000 was fetched with a _genomic suffix; the others were not.
            suffix = "_genomic" if a.corpus == "mito1000" else ""
            written = fetch_nuccore(accessions, a.out, suffix, a.batch or 200)
    elif a.accessions:
        accessions = read_accessions(a.accessions)
        print(f"{len(accessions):,} accessions -> {a.out}", flush=True)
        if accessions and accessions[0].startswith(("GCA_", "GCF_")):
            written = fetch_assemblies(accessions, a.out, a.batch or 20)
        else:
            written = fetch_nuccore(accessions, a.out, "", a.batch or 200)
    elif a.query:
        if not a.count:
            ap.error("--query needs --count")
        written = fetch_by_query(a.query, a.count, a.out, "", a.batch or 200)
    else:
        ap.error("one of --corpus, --accessions, --query is required")

    sizes = [os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)]
    print(f"wrote {written:,} files, {sum(sizes) / 1e6:.1f} MB", flush=True)

    if a.corpus:
        expected = len(read_accessions(os.path.join(LISTS, f"{a.corpus}.txt")))
        if written != expected:
            print(f"WARNING: expected {expected:,} records, wrote {written:,}. "
                  f"Records are withdrawn and suppressed over time; a shortfall here "
                  f"means the corpus is no longer exactly the one the paper used.",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
