# DEDUCT

DEDUCT (Deep DNA Sequence Compression by Using Orthogonal Transform Network Layers) is a
reference-free, constructively lossless compressor for collections of related genomes.
It separates every assembly into contigs, groups the corresponding contigs across
assemblies into contig columns, clusters each column with a rotation- and
strand-invariant FFT correlation, aligns each cluster to a consensus, and codes the
aligned matrix with a per-cluster convolutional autoencoder whose bottleneck is a
vertical block DCT; this route is the learned path and the per-cluster network is the
trained coder. Sequences that do not cluster or do not align go to a pooled gzip
fallback, and every file is reconstructed byte-exactly because the residual is computed
against what the stored decoder will produce.

The method is described in "DEDUCT: Deep DNA Sequence Compression by Using Orthogonal
Transform Network Layers" (accepted for publication in IEEE Access, 2026). This repository holds the
implementation, the released configuration, the accession lists and fetch script for the
public corpora, and the test suite.

## Requirements and installation

| Requirement | Version (`pyproject.toml`) | Used for |
|---|---|---|
| Python | 3.10 or newer | |
| PyTorch | `torch>=2.0`, with CUDA | clustering FFTs, training, decoding |
| NumPy | `numpy>=1.24` | matrices and byte coders |
| edlib | `edlib>=1.3` | gap-aware alignment to the consensus |
| pytest | `pytest>=7` (optional extra `test`) | the test suite only |

gzip comes from the Python standard library; the package adds no other compression
dependency. `pip install -e .` installs NumPy, PyTorch and edlib. If the default PyTorch
wheel for your platform is CPU-only, install a CUDA build of PyTorch first; `--device cpu`
works but is much slower.

```sh
git clone https://github.com/Cetin-Lab/deduct.git deduct
cd deduct
pip install -e .            # numpy, torch, edlib
pip install -e ".[test]"    # adds pytest
deduct --help
```

## Usage

The command line has three subcommands. Every codec option has a default that does not
need to be given; the values used are written into `manifest.z`, so decompression never
needs them repeated.

### Compress

```sh
deduct compress /path/to/fasta_dir compressed/
deduct compress /path/to/fasta_dir compressed/ --checksums            # store a SHA-256 per file
deduct compress /path/to/fasta_dir compressed/ --ambiguity mask       # keep lengths under N runs
deduct compress /path/to/fasta_dir compressed/ --config configs/default.json
deduct compress /path/to/fasta_dir compressed/ --limit 100            # first 100 files only
```

The input is a directory of FASTA files, one file per assembly, with the suffixes `.fna`,
`.fa`, `.fasta` or their `.gz` forms. A file may hold several records: the n-th record of
every file forms contig column n. The output directory is created if it does not exist.
On success the command prints the manifest as JSON (byte counts per component, coverage
of the learned path, per-cluster statistics, timings, D-BPB, F-BPB, ratio) and exits 0.
An input the pipeline refuses (an empty record, a record made only of non-ACGT symbols
under `delete`, two files with the same stem, a file with no records) is reported on
stderr with exit code 1, and the output directory is left empty.

- `--checksums` stores a SHA-256 per input file in the side channel so that
  `deduct verify` can check the compress-decompress cycle. Reconstruction does not need
  it. On a collection of many short records the digests are the largest single
  side-channel term (6.03 MB with them against 3.31 MB without on the 67,190-record
  mitochondrial collection), so they are off by default.
- `--ambiguity delete|mask` decides what happens to non-ACGT symbols before clustering.
  `delete` (the default) removes them, which shifts every later coordinate and can take
  a record with long `N` runs out of rotation register with its relatives; `mask`
  substitutes a placeholder base so the length survives. Both are exactly invertible from
  the exceptions blob.
- `--config FILE` overlays a JSON file of option overrides. Keys are the field names of
  `deduct.encode.Config` (with underscores) or the flag names (with hyphens); an unknown
  key aborts the run before anything is compressed. The file is applied after the flags
  are parsed, so its values override flags given on the same command line. Three keys
  exist only here: `transform`, `soft_threshold` and `decision_margin`.
  `configs/default.json` is the released configuration; `device` is deliberately not in
  it, so `--device` keeps working alongside it.
- `--limit N` uses only the first N files in sorted name order.

### Decompress

```sh
deduct decompress compressed/ restored/
```

Rebuilds the normalised files (see Limitations) from the compressed output alone and
prints a JSON summary with `verified`, `failed` and `unchecked` counts. The exit code is
0, or 1 if a stored digest did not match. `--device cpu` selects the CPU.

### Verify

```sh
deduct verify compressed/ ; echo $?
```

Decompresses into a temporary directory and compares every file against the SHA-256
recorded when it was read. The exit code is the contract:

| Exit code | Meaning |
|---|---|
| 0 | every file was checked and every digest matched |
| 1 | at least one digest did not match |
| 3 | no digests are stored (compressed without `--checksums`); decompression completed but nothing was compared, and a message says so on stderr |

Exit code 2 is argparse's usage error, which is why the no-digest case uses 3. Run
`verify` as its own invocation, in an isolated process as the paper's protocol does, so
that the decoder does not inherit the arithmetic flags of the process that compressed.

### Options

Defaults are those of `deduct.encode.Config` and `deduct/cli.py`; they are the released
configuration and none of them needs to be given.

| Option | Default | Effect |
|---|---|---|
| `--kernel H,W` | `3,1` | Convolution kernel. `H` must be odd. `W = 1` keeps each column's latent a function of that column alone. |
| `--vstride N` | `4` | Row downsampling, a power of two. Columns are never downsampled. |
| `--block-height N` | `0` | Rows per block. `0` is one block per cluster, rounded up to a multiple of `vstride`, split only when one tile would exceed 5,000,000 cells (about 9,800 rows at tile width 512). |
| `--tile-width N` | `512` | Columns per tile. A cluster narrower than one tile goes to the fallback. |
| `--latent-channels N` | `1` | Channels at the bottleneck. |
| `--mid-channels N` | `32` | Width of the intermediate layers. |
| `--depth N` | none | Convolution layers. Unset means log2(`vstride`), which is 2 at the default stride. |
| `--activation` | `relu` | One of `relu`, `gelu`, `leaky`, `tanh`, `none`, applied between layers. |
| `--scale S` | `16` | Quantisation step for the transform coefficients. |
| `--penalty` | `deadzone` | `deadzone` charges only what survives quantisation; `l1` charges every coefficient. |
| `--penalty-weight F` | `0.01` | Weight of the sparsity penalty on the latent. |
| `--epochs N` | `60` | Maximum training epochs per cluster. |
| `--check-every N` | `5` | Write and measure the cluster's output every `N` epochs and keep the smallest. `0` measures once at the end. |
| `--patience N` | `3` | Stop after this many consecutive measurements without a smaller output. `0` always runs the full epoch budget. |
| `--learning-rate F` | `0.005` | Adam learning rate. The only hyperparameter the paper allows to vary per corpus; every reported corpus uses this value. |
| `--batch N` | `0` | Tiles per step. `0` chooses 4 for blocks taller than 400 rows and 16 otherwise. |
| `--seed N` | `0` | Random seed. Ten seeds on one 1,000 x 16,569 matrix gave a coefficient of variation of 12.6% in compressed size (range 47,727 to 68,182 B) at argmax accuracy 0.9965 to 0.9976. |
| `--latent-coder` | `dropk` | Drops frequency planes that are entirely zero. Also `dropk_best`, `dropk_bitpack`. |
| `--residual-coder` | `either` | Keeps the smaller of a dense symbol plane and a sparse list, per cluster. Also `dense`, `sparse`, `dense_best`. |
| `--weight-coder` | `raw16` | Float16 tensors. Exact, because the encoder predicts with the rounded weights. Also `raw32`, `torch`. |
| `--gzip-level N` | `9` | Level for every stored blob, on gzip's 1 to 9 scale. |
| `--measure-level N` | `1` | Level used only to rank epochs during training. The kept output is always written at `--gzip-level`. |
| `--accept F` | `0.6` | Match fraction at which a sequence joins a cluster. |
| `--max-clusters N` | `30` | Clusters per contig column. |
| `--min-members N` | `64` | Smallest cluster the learned path admits (the admission threshold). |
| `--identity-threshold F` | `0.35` | Edit distance over the consensus length (the cluster's most common member length) above which a member goes to the fallback. |
| `--workers N` | `24` | Parallel workers for parsing and alignment. |
| `--device` | `cuda` | `cpu` works and is much slower. |
| `transform` (config file only) | `"dct"` | `dct`, `identity` or `learned`; the latter two are the paper's ablation variants. |
| `soft_threshold` (config file only) | `true` | Learnable soft threshold on the coefficients. |
| `decision_margin` (config file only) | `0.1` | A residual entry is written wherever the top two logits are closer than this, even when the prediction is right, so no arithmetic difference below half the margin can change a decoded symbol. `0` disables the guard and makes decoding depend on reproducing the encoder's floating point exactly, which is why it is not a flag. |

Invalid geometry raises a `ValueError` when the first cluster's model is built, which is
after clustering: `vstride` must be a power of two, `block-height` a multiple of it, and
the kernel height odd.

### Output directory layout

`deduct compress` writes the compressed output directory; `deduct decompress` reads only
that directory.

```
compressed/
  manifest.z                gzip JSON: configuration, environment (torch, CUDA, cuDNN, GPU),
                            front-end settings, corpus counts, per-cluster statistics,
                            coverage, byte counts per component, timings
  sidechannel/
    headers.z               every record header, in file order then record order
    structure.z             per-file name, size and record count; per-column list of the
                            files that contribute a line; a SHA-256 per file under --checksums
    exceptions.z            position and identity of every non-ACGT run
  clusters/c<column>_g<index>/     one directory per cluster the learned path coded
    weights.z               decoder parameters (float16 under raw16)
    latent.z                quantised DCT coefficients
    residual.z              true symbol wherever the decoder's argmax is wrong or inside
                            the decision margin
    rows.z                  per-member rotation, strand, length and insertions
  fallback/
    pool.z                  every sequence the learned path did not take, length-ordered, gzip
    index.z                 the (column, row) slot of each pooled sequence
```

Only the decoder half of each network is stored; the DCT matrix is regenerated from the
block height. While compressing, a scratch directory `_work/` exists inside the output
directory and is removed when compression finishes. The exceptions blob is stored under
`sidechannel/` but is sequence, so it is charged inside F-BPB; the headers and record
structure are not.

## Source layout

```
deduct/
  common.py      alphabets, gzip and lzma wrappers, hashing, integer packing
  normalise.py   FASTA to the two-line indexed form
  separate.py    contig columns, the side channel and the exceptions blob
  group.py       rotation- and strand-invariant clustering
  align.py       gap-aware alignment to a consensus, and its inverse
  model.py       the codec network
  coders.py      byte coders for the latent, residual and weight blobs
  encode.py      Config, training, quantisation, residual, per-cluster blobs
  fallback.py    pooled gzip coder
  pipeline.py    the stages end to end: compress, decompress, verify
  cli.py         command line
configs/default.json   the released configuration, every Config field except device
tests/                 the test suite (see below)
scripts/
  fetch_corpus.py      rebuild any public corpus from its accession list, and verify it
  corpora/             one accession list per corpus, CHECKSUMS.txt, PROVENANCE.md
```

## Corpora

Every public corpus in the paper is defined by an accession list under
`scripts/corpora/`, not by a query: the databases grow, and a search capped at N records
returns a different N later. `scripts/fetch_corpus.py` rebuilds a corpus from its list
and checks the result against the per-corpus digest in `scripts/corpora/CHECKSUMS.txt`.

```sh
python scripts/fetch_corpus.py --list                                  # what is available
python scripts/fetch_corpus.py --corpus mito1000  --out ./mito1000     # rebuild one
python scripts/fetch_corpus.py --verify ./mito1000 --corpus mito1000   # check it matches
```

| Corpus | Records | Bytes as downloaded | Source |
|---|---|---|---|
| `ecoli1` | 2,397 assemblies | 11,207,086,531 (11.21 GB) | NCBI Datasets; selected from NCBI Pathogen Detection |
| `senterica` | 1,024 assemblies | 5,196,142,476 (5.20 GB) | NCBI Datasets; selected from NCBI Pathogen Detection |
| `mito1000` | 1,000 | 16,659,150 (16.7 MB) | NCBI nuccore |
| `mito_full` | 67,190 | 1,118,274,507 (1.12 GB) | NCBI nuccore |
| `sars_2020` | 31,782 | 951,400,872 (0.95 GB) | NCBI nuccore |
| `sars_2021` | 50,000 | 1,496,576,002 (1.50 GB) | NCBI nuccore |
| `sars_2022` | 50,000 | 1,495,436,765 (1.50 GB) | NCBI nuccore |
| `sars_2023` | 50,000 | 1,498,038,856 (1.50 GB) | NCBI nuccore |
| `sars_2024` | 50,000 | 1,492,932,264 (1.49 GB) | NCBI nuccore |
| `sars_2025` | 50,000 | 1,496,402,148 (1.50 GB) | NCBI nuccore |
| `sars_2026` | 9,235 | 275,905,030 (0.28 GB) | NCBI nuccore |

The whole set is 26,244,854,601 bytes (26.2 GB). The sizes are those of the downloaded
files; the paper reports the normalised size of each collection, which differs slightly
because normalisation joins wrapped sequence lines and appends the record index to each
header. `scripts/corpora/PROVENANCE.md` carries the queries, the fetch dates, the
per-year availability counts for SARS-CoV-2 and the sampling caveats that go with them.
DNACorpus (17 files, 685,597,124 bases) is distributed by its authors and is not
included here. `scripts/README.md` describes the fetch script in more detail.

## Tests

```sh
pip install -e ".[test]"
python -m pytest
```

`python -m pytest --collect-only -q` collects 52 tests in four files:

- `tests/test_coders.py` (11): every registered latent, residual and weight coder
  reconstructs its input exactly, including coefficients beyond int8.
- `tests/test_front_end.py` (7): the correlation frame length is 7-smooth and at least
  twice the sequence length; clustering is invariant to rotation and strand and
  unrelated sequences stay out; alignment is invertible under insertions, deletions,
  rotation and strand.
- `tests/test_end_to_end.py` (23): a small corpus compresses and decompresses to
  byte-identical files, with and without `--checksums`, under both `--ambiguity` modes
  and under alternative configurations; empty records, files without records and
  colliding input stems are refused, and a refused run does not poison a retry; `verify`
  on an output without digests exits 3; F-BPB charges the exceptions blob; the decoder
  pins its own arithmetic and a guarded output survives decoding under TF32. The two
  arithmetic tests need a CUDA device and are skipped without one.
- `tests/test_exceptions.py` (11): the run-length exceptions blob reconstructs exactly
  over its edge cases and refuses blobs it does not understand; an all-`N` record is
  refused under `delete` and reconstructs exactly under `mask`; a corpus with several
  ambiguity symbols is byte-exact end to end. The last test compresses a real `mito1000` corpus at the default
  configuration and is skipped unless `DEDUCT_MITO1000` names its directory.

The suite runs on the CPU when no CUDA device is present, more slowly.

## Limitations

- Reconstruction is exact with respect to the normalised form of the input, not the
  file as downloaded: one header line and one sequence line per record, upper-cased, with
  ` contig<k>` appended to the header. A wrapped FASTA is restored unwrapped.
- Upper-casing discards soft-masking. It changes nothing on the corpora measured, which
  are single-case, but a collection whose lower-case runs carry meaning needs different
  handling.
- Under the default `--ambiguity delete`, removing non-ACGT symbols shifts every later
  coordinate, so a record with long `N` runs can fail to rotation-align with its
  relatives and goes to the fallback; this is what limits coverage on the SARS-CoV-2
  years. A record consisting only of non-ACGT symbols is refused under `delete`; use
  `mask`.
- GPU memory peaks during clustering, which holds the transforms of every sequence in the
  largest contig column: 22.1 GB on `ecoli1` and 26.6 GB on `senterica`, measured on a
  32 GB card. Compression itself peaks near 11 GB, because only one cluster's model is in
  memory at a time. The clustering cache is budgeted against free VRAM and falls back to
  recomputation, so smaller cards work at lower speed.
- The fallback is gzip, whose 32 KiB window bounds what it can match. On a collection
  where nothing clusters (DNACorpus), the whole output is at that rate.
- Compression trains a model per cluster and is much slower than decompression: 7,811.5 s
  to compress `ecoli1` against 210.2 s to decompress and verify it.

## Results

All numbers below come from Table 2 of the paper, under the single released configuration
(`configs/default.json`), with every file decoded and checked byte-exact. D-BPB is the
trained coder's bits per base over the bases it received, weights included; F-BPB is the
whole pipeline's bits per input symbol, fallback and exceptions included; the ratio is
the normalised input over the output on disk, side channel and manifest included.

| Collection | D-BPB | F-BPB | Ratio |
|---|---|---|---|
| `ecoli1` (2,397 assemblies, 11.22 GB) | 0.0259 | 0.3755 | 21.2x |
| Mitochondrial (67,190 records) | 0.0163 | 0.0180 | 337.9x |
| SARS-CoV-2 (291,017 records) | 0.0214 | 0.0592 | 129.0x |
| *S. enterica* (1,024 assemblies) | 0.6427 | 1.7961 | 4.5x |
| DNACorpus (17 unrelated sequences) | -- | 2.1724 | 3.7x |

The component sizes behind these rates, the per-year SARS-CoV-2 results, the ablation and
the timing tables are in the paper and its supplement. The companion page at
https://cetin-lab.github.io/deduct/ and its notes page at https://cetin-lab.github.io/deduct/notes.html carry the material that was
condensed out of the supplement.

## Citation

See `CITATION.cff`. The article has been accepted for publication in IEEE Access (2026); the software
release is archived at https://doi.org/10.5281/zenodo.24122005.

## License

PolyForm Noncommercial License 1.0.0 (`LICENSE.md`): noncommercial use, modification and
redistribution are permitted; the license is not OSI-approved. The authors and their
affiliations are listed in `AUTHORS`.
