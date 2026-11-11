# Where each corpus came from

One accession list per corpus, one accession per line, sorted. These lists *are* the
corpus definition: they name the exact records that produced the numbers in the paper.
`CHECKSUMS.txt` carries a digest per corpus so a rebuild can be checked rather than
assumed.

Rebuild and check:

```
python fetch_corpus.py --corpus mito1000 --out ./mito1000
python fetch_corpus.py --verify ./mito1000 --corpus mito1000
```

## The NCBI nucleotide corpora

Retrieved with E-utilities `esearch`/`efetch` against `nuccore`, one FASTA file per
record, sequence upper-cased on one line.

| corpus | records | fetched | query |
|---|---|---|---|
| `mito1000` | 1,000 | 2026-07-26 | mito base query, capped at 1,000 |
| `mito_full` | 67,190 | 2026-07-31 | mito base query, every record it returned |
| `sars_2020` | 31,782 | 2026-08-02 | SARS base query, `2020/01/01` to `2020/12/31` |
| `sars_2021` | 50,000 | 2026-08-02 | SARS base query, `2021/01/01` to `2021/12/31` |
| `sars_2022` | 50,000 | 2026-08-02 | SARS base query, `2022/01/01` to `2022/12/31` |
| `sars_2023` | 50,000 | 2026-08-02 | SARS base query, `2023/01/01` to `2023/12/31` |
| `sars_2024` | 50,000 | 2026-08-02 | SARS base query, `2024/01/01` to `2024/12/31` |
| `sars_2025` | 50,000 | 2026-08-02 | SARS base query, `2025/01/01` to `2025/12/31` |
| `sars_2026` | 9,235 | 2026-08-02 | SARS base query, `2026/01/01` to `2026/12/31` |

Base queries, verbatim (they are also in `fetch_corpus.py`, so the two cannot drift):

```
mito:  "Homo sapiens"[Organism] AND mitochondrion[Filter]
       AND "complete genome"[Title] AND 16000:17000[SLEN]

SARS:  "Severe acute respiratory syndrome coronavirus 2"[Organism]
       AND "complete genome"[Title] AND 29000:30000[SLEN]
```

### The SARS-CoV-2 cap, and what it does and does not mean

Each year was capped at **50,000**, so the collection is a stratified sample by publication
year rather than a census. How large a share that is varies enormously by year, and anyone
reading the per-year compressed outputs as comparable samples should see the denominators first:

| year | records available | taken | share of the year |
|---|---|---|---|
| 2020 | 31,782 | 31,782 | **100%** |
| 2021 | 637,718 | 50,000 | 7.8% |
| 2022 | 1,003,674 | 50,000 | 5.0% |
| 2023 | 1,236,783 | 50,000 | **4.0%** |
| 2024 | 172,718 | 50,000 | 28.9% |
| 2025 | 70,838 | 50,000 | 70.6% |
| 2026 | 9,245 | 9,235 | 99.9% |
| **total** | **3,162,758** | **291,017** | **9.2%** |

*Availability counts are the `<Count>` the same `esearch` query returns, checked 2026-08-02.
Past years are effectively stable; 2026 was still growing at the fetch date, which is why
9,235 of 9,245 appear.*

So 2020 is a census of its year and 2023 is a 4% slice of its own. The years are not
equally represented relative to their size, and a difference between two per-year compressed outputs
is not by itself a difference between two years of the pandemic.

Stratifying by year was deliberate: an uncapped draw is dominated by whichever years
submitted most, which is not the same as covering the pandemic's clade history. But two
limits should be stated plainly:

- **Within a year the 50,000 are not a random sample.** They are the first 50,000 in
  the order `esearch` returned on the fetch date. Nothing about that order is
  guaranteed, documented, or stable.
- **The cap is why five rows of the per-year table read exactly 50,000.** That is the
  quota binding, not a property of those years.

The accession lists remove the reproducibility half of this problem: anyone can rebuild
the identical 291,017 records. They do not remove the sampling half, and a study that
needed a defensible random sample of each year would have to draw one.

## The assembled-genome corpora

`ecoli1` and `senterica` are assemblies, not `nuccore` records, and were selected from
NCBI Pathogen Detection rather than by an Entrez query, so for these the accession list
is the *only* definition of the corpus. `fetch_corpus.py` pulls them through the NCBI
Datasets API, which returns one multi-record FASTA per assembly; the contig structure is
kept, because separating contigs into columns is the first thing the method does.

| corpus | assemblies | note |
|---|---|---|
| `ecoli1` | 2,397 | *E. coli* / *Shigella* |
| `senterica` | 1,024 | *S. enterica* panel |

## Corpora not covered here

- **DNACorpus** (17 files) is distributed by its authors; use their release rather than
  a copy of ours, and check it against the sizes they publish.

## A caution about rebuilding later

GenBank records are revised, replaced and occasionally suppressed. A rebuild done long
after these dates may be short a few records, or may return a newer version of one, and
`--verify` will say so. Numbers computed on a drifted corpus will be close but need not
be identical, which is the reason the digests are here at all.

## Redistribution

NCBI places no restrictions on the use or redistribution of the data in GenBank, so the
sequences themselves may be mirrored. NCBI's own standard caveat applies and is worth
repeating: some submitters may claim patent, copyright or other rights in what they
submitted, and NCBI cannot warrant otherwise on their behalf.
