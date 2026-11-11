# scripts

## `fetch_corpus.py`: rebuild the public corpora

```
python fetch_corpus.py --list                                  # what is available
python fetch_corpus.py --corpus mito1000  --out ./mito1000     # rebuild one
python fetch_corpus.py --verify ./mito1000 --corpus mito1000   # check it is the right one
```

`--corpus` fetches by the accession lists in `corpora/`, which name the exact records the
paper used. Prefer it. `--query` re-runs the original search instead and is *not*
reproducible: the databases grow, and where a per-year cap binds you get whichever
records NCBI returns first today. The script says so when it happens. `--accessions FILE`
fetches an arbitrary accession list.

`--verify` compares a rebuilt directory against `corpora/CHECKSUMS.txt`, a digest over
every file's SHA-256 and name, and tells you if the corpus has drifted.

Sizes, from `corpora/CHECKSUMS.txt`, so nothing is a surprise: `mito1000` is 17 MB and
takes under a minute; `mito_full` is 1.12 GB; the SARS-CoV-2 years are 0.95 GB (2020),
1.49 to 1.50 GB each for 2021 to 2025, and 0.28 GB (2026); `ecoli1` is 11.21 GB and
`senterica` 5.20 GB, both through the Datasets API. The whole set is 26.2 GB.

Set `NCBI_API_KEY` to raise the request ceiling from three per second to ten, and
`NCBI_EMAIL` to identify yourself as NCBI asks.

### What is not here

DNACorpus is distributed by its authors; use their release. `corpora/PROVENANCE.md`
gives the queries, the fetch dates, and the sampling caveats that go with the
SARS-CoV-2 collection.

## Running the codec

Corpus preparation is the only thing that differs between the runs in the paper;
everything else is the same two commands.

```
deduct compress /path/to/corpus compressed/ --checksums
deduct verify   compressed/
```

`verify` must exit 0. It decompresses into a temporary directory and compares each
reconstructed file against the SHA-256 taken when it was read, so a size reported in the
paper is always a size that was decoded back. Those digests are what `--checksums`
stores; it is off by default, and without it `verify` still reconstructs everything but
exits 3 and reports how many files it could not check. Run it as a separate invocation in
an isolated process, not inside the one that compressed: a verification inside that
process inherits its arithmetic flags and cannot detect an encoder/decoder mismatch.

The sizes reported in the paper are for the released configuration
(`configs/default.json`), with the stored digest bytes subtracted.
