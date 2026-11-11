"""Command line: `deduct compress`, `deduct decompress`, `deduct verify`.

    deduct compress   SRC OUTPUT [options]
    deduct decompress OUTPUT DIR
    deduct verify     OUTPUT

`compress` reads a directory of FASTA files and writes a compressed output directory.
`decompress` rebuilds the normalised files from the compressed output alone. `verify` does
both and compares each file's SHA-256 against the one recorded when it was read. Digests are
only recorded when compression ran with `--checksums`; on a compressed output without them
`verify` has nothing to compare, which it reports on stderr and with exit status 3 rather
than claiming success.

Every codec option below has a default that does not need to be given; the values chosen are
written into the manifest of the compressed output, so a compressed output made with
non-default options decompresses without them being repeated.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .encode import Config


def _kernel(text: str) -> tuple[int, int]:
    parts = text.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"kernel must be 'height,width', got {text!r}")
    return int(parts[0]), int(parts[1])


def _add_codec_options(parser: argparse.ArgumentParser) -> None:
    defaults = Config()
    group = parser.add_argument_group("codec geometry")
    group.add_argument("--kernel", type=_kernel, default=defaults.kernel, metavar="H,W",
                       help="convolution kernel; height must be odd (default 3,1)")
    group.add_argument("--vstride", type=int, default=defaults.vstride,
                       help="row downsampling, a power of two (default 4)")
    group.add_argument("--block-height", type=int, default=defaults.block_height,
                       help="rows per block; 0 means one block per cluster (default 0)")
    group.add_argument("--tile-width", type=int, default=defaults.tile_width,
                       help="columns per tile (default 512)")
    group.add_argument("--latent-channels", type=int, default=defaults.latent_channels,
                       help="channels at the bottleneck (default 1)")
    group.add_argument("--mid-channels", type=int, default=defaults.mid_channels,
                       help="width of the intermediate layers (default 32)")
    group.add_argument("--depth", type=int, default=defaults.depth,
                       help="convolution layers; default log2(vstride)")
    group.add_argument("--activation", default=defaults.activation,
                       choices=["relu", "gelu", "leaky", "tanh", "none"])

    group = parser.add_argument_group("training")
    group.add_argument("--scale", type=float, default=defaults.scale,
                       help="quantisation step (default 16)")
    group.add_argument("--penalty", default=defaults.penalty, choices=["deadzone", "l1"])
    group.add_argument("--penalty-weight", type=float, default=defaults.penalty_weight)
    group.add_argument("--epochs", type=int, default=defaults.epochs)
    group.add_argument("--check-every", type=int, default=defaults.check_every,
                       help="measure the compressed output every N epochs and keep the smallest; "
                            "0 measures once at the end (default 5)")
    group.add_argument("--patience", type=int, default=defaults.patience,
                       help="stop after this many measurements without improvement; "
                            "0 always runs the full epoch budget (default 3)")
    group.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    group.add_argument("--batch", type=int, default=defaults.batch,
                       help="tiles per step; 0 chooses by matrix height (default 0)")
    group.add_argument("--seed", type=int, default=defaults.seed)

    group = parser.add_argument_group("byte coders")
    group.add_argument("--latent-coder", default=defaults.latent_coder,
                       choices=["dropk", "dropk_best", "dropk_bitpack"])
    group.add_argument("--residual-coder", default=defaults.residual_coder,
                       choices=["either", "dense", "sparse", "dense_best"])
    group.add_argument("--weight-coder", default=defaults.weight_coder,
                       choices=["raw16", "raw32", "torch"])
    group.add_argument("--gzip-level", type=int, default=defaults.gzip_level)
    group.add_argument("--measure-level", type=int, default=defaults.measure_level,
                       help="level used to rank epochs during training; the compressed output "
                            "that is kept is always written at --gzip-level (default 1)")

    group = parser.add_argument_group("front end")
    group.add_argument("--accept", type=float, default=0.6,
                       help="match fraction to join a cluster (default 0.6)")
    group.add_argument("--max-clusters", type=int, default=30,
                       help="clusters per contig column (default 30)")
    group.add_argument("--min-members", type=int, default=64,
                       help="smallest cluster the learned path admits, the admission threshold "
                            "(default 64)")
    group.add_argument("--identity-threshold", type=float, default=0.35,
                       help="edit distance over modal length above which a member goes to the "
                            "fallback (default 0.35)")
    group.add_argument("--workers", type=int, default=24)
    group.add_argument("--device", default=defaults.device)


def _config_from(args) -> Config:
    """Build a `Config` from the parsed arguments.

    Every `Config` field is taken from `args` when it is present there, so a field reachable
    only through `--config` (`transform`, `soft_threshold`, `decision_margin`) is honoured
    rather than silently dropped.
    """
    values = {}
    for field in dataclasses.fields(Config):
        if hasattr(args, field.name):
            values[field.name] = getattr(args, field.name)
    values["kernel"] = tuple(values["kernel"])
    return Config(**values)


def _apply_config_file(args, path: str) -> None:
    """Overlay a JSON file of option overrides, rejecting anything that would be ignored.

    A parser that accepts a key and drops it is worse than one that refuses it: the run
    succeeds and reports a configuration it did not use.
    """
    with open(path) as handle:
        overrides = json.load(handle)
    known = {f.name for f in dataclasses.fields(Config)} | set(vars(args))
    unknown = sorted(k for k in (key.replace("-", "_") for key in overrides) if k not in known)
    if unknown:
        raise SystemExit(f"{path}: unknown option(s) {', '.join(unknown)}; "
                         f"nothing was compressed")
    for key, value in overrides.items():
        setattr(args, key.replace("-", "_"), value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deduct", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    compress = sub.add_parser("compress", help="compress a directory of FASTA files")
    compress.add_argument("src", help="directory of FASTA files")
    compress.add_argument("archive", metavar="output",
                          help="compressed output directory to create")
    compress.add_argument("--limit", type=int, help="use only the first N files")
    compress.add_argument("--config", help="JSON file of option overrides")
    compress.add_argument("--ambiguity", choices=("delete", "mask"), default="delete",
                          help="what to do with non-ACGT symbols before clustering. `delete` "
                               "removes them, which shortens the sequence and shifts every "
                               "coordinate after it; `mask` substitutes a placeholder base so "
                               "the rotation correlation stays in register. Both are exactly "
                               "invertible.")
    compress.add_argument("--checksums", action="store_true",
                          help="store a SHA-256 per input file so that `deduct verify` can "
                               "check the compress-decompress cycle. Reconstruction does not "
                               "need it and is constructive either way; on a collection of "
                               "many short records the digests are the largest single "
                               "side-channel term, so they are off by default.")
    _add_codec_options(compress)

    decompress = sub.add_parser("decompress",
                                help="rebuild the normalised files from a compressed output")
    decompress.add_argument("archive", metavar="output",
                            help="compressed output directory to read")
    decompress.add_argument("out", metavar="dir",
                            help="directory to write the reconstructed files into")
    decompress.add_argument("--device", default="cuda")

    verify = sub.add_parser("verify",
                            help="decompress and check every file's checksum; a compressed output "
                                 "made without --checksums has nothing to check, which "
                                 "exits 3 instead of claiming success")
    verify.add_argument("archive", metavar="output",
                        help="compressed output directory to check")
    verify.add_argument("--device", default="cuda")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from . import pipeline

    if args.command == "compress":
        if args.config:
            _apply_config_file(args, args.config)
        try:
            stats = pipeline.compress(args.src, args.archive, _config_from(args),
                                      limit=args.limit, accept=args.accept,
                                      max_clusters=args.max_clusters,
                                      min_members=args.min_members,
                                      identity_threshold=args.identity_threshold,
                                      workers=args.workers, checksums=args.checksums,
                                      ambiguity=args.ambiguity)
        except ValueError as error:
            # A refusal at read time (empty record, colliding stems) is a statement about the
            # input, not a defect in the program; the user needs the sentence, not the
            # traceback it rode in on.
            print(f"deduct compress: {error}", file=sys.stderr)
            return 1
        json.dump(stats, sys.stdout, indent=1)
        print()
        return 0

    if args.command == "decompress":
        stats = pipeline.decompress(args.archive, args.out, device=args.device)
        json.dump(stats, sys.stdout, indent=1)
        print()
        return 0 if not stats["failed"] else 1

    stats = pipeline.verify(args.archive, device=args.device)
    json.dump(stats, sys.stdout, indent=1)
    print()
    if stats["failed"]:
        return 1
    if not stats["verified"]:
        # Exit 0 here would let "verify passed" mean "nothing was checked". A distinct code
        # keeps 1 meaning what it always has: a digest that was compared and did not match.
        # 3 rather than 2 because argparse already exits 2 on a usage error, and a harness
        # treating "no digests" as benign must not swallow a mistyped flag with it.
        print("deduct verify: the compressed output stores no checksums, so no digest was "
              "compared; this run only confirmed that decompression completes. Compress with "
              "--checksums for a checked compress-decompress cycle.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
