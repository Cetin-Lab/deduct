"""Train a codec on one aligned matrix and write the compressed output for it.

The compressed output of a cluster is three blobs and a header:

    weights     the decoder's parameters
    latent      the quantised transform coefficients
    residual    the true symbol wherever the decoder's argmax is wrong, zero elsewhere

Losslessness is constructive rather than earned. The residual is computed against exactly what
the decoder will produce (the same weights, the same dequantised latent, the same argmax), so
the reconstruction is correct whatever the network learned. A poorly trained model costs bytes
in the residual and nothing else, and the compressed output is verified by decoding it before
its size is reported.

Two consequences follow and both are load-bearing. The weights must be serialised *before* the
encoder predicts with them, otherwise a lossy weight format would leave the decoder holding
different weights from the ones the residual was taken against. And the epoch to keep is chosen
by writing the compressed output and measuring it, not by the training loss, because the loss
is only a proxy for the quantity being minimised.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import coders
from .model import Codec, one_hot, tile_origins


MAX_TILE_CELLS = 5_000_000
"""Largest tile a single block may produce, in cells.

A tile becomes five one-hot channels and then `mid_channels` of activations, so what it needs
on the device is a multiple of this figure. Five million is the size at which a batch of four
is known to run on a 32 GB card; a cluster tall enough to exceed it is split into blocks rather
than failing, which is what a 67,190-member collection does at a tile width of 512.
"""


@dataclass
class Config:
    """Codec configuration. Every field is written into the manifest of the compressed output.

    `block_height = 0` means one block per cluster: the whole matrix height, rounded up to a
    multiple of `vstride`. Splitting the rows into shorter blocks makes each block carry its own
    copy of the cluster's shared signal, so the default is not to split.

    `patience` counts consecutive measurements without improvement, after which training stops.
    `0` disables it and always runs the full `epochs`.

    `measure_level` is the entropy-coding level used when the compressed output is written *to
    measure it* during training. Writing it dominates the cost of the stopping rule, and what
    the measurement needs is a ranking rather than a size, so it runs at level 1 and the
    compressed output that is kept is written at `gzip_level`. The two rankings are not
    identical; how far apart they sit on DEFLATE's 1..9 scale has not been measured, and the
    levels are close enough together that the gap is expected to be small.

    `learning_rate` is 5e-3 rather than the more usual 1e-3 because 1e-3 does not escape a
    plateau this objective has: on a 10,000-member mitochondrial collection it leaves argmax
    accuracy at 0.43 where a per-column majority vote reaches 0.998, and at 5e-3 the same
    configuration reaches 0.9977. Neither the sparsity penalty nor the patience setting moves
    that plateau; the step size is what does.

    `decision_margin` is how much room the decoder's argmax must win by before the encoder is
    willing to leave that cell uncorrected. A cell whose top two logits are closer than this
    gets a residual entry even though the prediction is right, which pins the symbol against
    any arithmetic that moves a logit by less than half the margin. `0` disables the guard and
    makes correctness depend on the decoder reproducing the encoder's floating point exactly.
    It is deliberately not a command-line option, unlike every other field here: turning it off
    trades a fraction of a percent of compressed output size for a compressed output that may
    not decode on other hardware, and that is not a trade to offer behind a flag.
    """

    block_height: int = 0
    tile_width: int = 512
    latent_channels: int = 1
    mid_channels: int = 32
    vstride: int = 4
    kernel: tuple[int, int] = (3, 1)
    activation: str = "relu"
    depth: int | None = None
    scale: float = 16.0
    penalty: str = "deadzone"
    penalty_weight: float = 1e-2
    epochs: int = 60
    check_every: int = 5
    patience: int = 3
    learning_rate: float = 5e-3
    batch: int = 0
    seed: int = 0
    latent_coder: str = "dropk"
    residual_coder: str = "either"
    weight_coder: str = "raw16"
    gzip_level: int = 9
    measure_level: int = 1
    decision_margin: float = 0.1
    transform: str = "dct"
    soft_threshold: bool = True
    device: str = "cuda"

    def resolved_block_height(self, height: int) -> int:
        """Rows per block: the whole matrix by default, capped so one tile fits in memory.

        A tile is `block_height * tile_width` cells and is expanded to five one-hot channels
        and then to `mid_channels`, so its footprint grows linearly with the block height. Above
        `MAX_TILE_CELLS` a single tile no longer fits on the device, and the matrix has to be
        split into blocks even though splitting costs: each block carries its own copy of the
        signal the cluster shares.
        """
        if self.block_height:
            return self.block_height
        blocks = max(1, math.ceil(height * self.tile_width / MAX_TILE_CELLS))
        per_block = math.ceil(height / blocks)
        return int(math.ceil(per_block / self.vstride) * self.vstride)

    def resolved_batch(self, height: int) -> int:
        if self.batch:
            return self.batch
        # Tiles of a tall matrix are large; a smaller batch keeps activation memory bounded.
        # The block-height cap above is what keeps a single tile from being too large at all,
        # so this only has to distinguish short blocks from long ones.
        return 4 if self.resolved_block_height(height) > 400 else 16


@dataclass
class Result:
    """What one cluster's compressed output cost and how well the model did."""

    weights: int = 0
    latent: int = 0
    residual: int = 0
    total: int = 0
    accuracy: float = 0.0
    guarded: int = 0
    """Cells the prediction got right that were corrected anyway, for lack of a clear margin."""
    epoch: int = 0
    verified: bool = False
    stopped_early: bool = False
    curve: list = field(default_factory=list, repr=False)
    train_seconds: float = 0.0
    height: int = 0
    width: int = 0
    bases: int = 0
    gaps: int = 0
    frequencies_used: int = 0
    frequencies: int = 0
    blobs: dict = field(default_factory=dict, repr=False)


def _set_arithmetic() -> None:
    """Pin the floating-point paths the codec's correctness rests on.

    The residual records only the cells the encoder's prediction got wrong, so a cell the
    encoder predicted correctly carries no correction and the decoder has to arrive at the same
    symbol on its own. That holds only while both sides evaluate the decoder in the same
    arithmetic. cuDNN's TF32 path carries a 10-bit mantissa against float32's 24, which is
    enough to move an argmax wherever the top two logits are close, and it is enabled by
    default in a process that has not said otherwise.

    Both `encode_matrix` and `decode_matrix` call this before touching the model, so a
    decompression run in an isolated process reproduces the arithmetic the compressed output
    was written under. Every coded cluster depends on it; it is not a tuning knob.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    _set_arithmetic()


def _tiles(matrix: np.ndarray, block_height: int, tile_width: int):
    """Pad the matrix up to whole tiles and return the tile stack and the padded shape."""
    height, width = matrix.shape
    padded_h = int(math.ceil(height / block_height) * block_height)
    padded_w = int(math.ceil(width / tile_width) * tile_width)
    padded = np.zeros((padded_h, padded_w), np.uint8)
    padded[:height, :width] = matrix
    origins = tile_origins(padded_h, padded_w, block_height, tile_width)
    stack = np.stack([padded[i:i + block_height, j:j + tile_width] for i, j in origins])
    # Kept at one byte per cell. The training loss needs int64 targets, but only for the batch
    # on the device: widening the whole stack costs eight bytes a cell, which on a cluster of a
    # quarter of a million near-identical members is tens of gigabytes of host memory for a
    # tensor that is only ever read a batch at a time.
    return torch.from_numpy(stack), origins, (padded_h, padded_w)


def _penalty(coefficients: torch.Tensor, config: Config) -> torch.Tensor:
    """Sparsity term.

    `deadzone` charges only for the part of a coefficient that survives quantisation, which is
    what actually costs bytes; `l1` charges every coefficient including those the quantiser
    would send to zero for free.
    """
    if config.penalty == "deadzone":
        return F.relu(coefficients.abs() - config.scale / 2).mean()
    if config.penalty == "l1":
        return coefficients.abs().mean()
    raise ValueError(f"unknown penalty {config.penalty!r}; have 'deadzone' and 'l1'")


def _write(model: Codec, tiles, origins, matrix, padded_shape, config: Config,
           level: int | None = None) -> Result:
    """Serialise the model, predict with what came back, and build the three blobs.

    `level` overrides the entropy-coding level; the caller passes the cheap `measure_level`
    while ranking epochs and the full `gzip_level` for the compressed output that is kept.
    """
    level = config.gzip_level if level is None else level
    height, width = matrix.shape
    block_height = config.resolved_block_height(height)
    n_freq = block_height // config.vstride
    device = config.device

    weight_encode, weight_decode = coders.WEIGHT_CODERS[config.weight_coder]
    template = model.decoder_state()
    weight_blob = weight_encode(template, level=level)

    # Predict with the weights the compressed output holds, not the ones training left in
    # memory. The copy keeps a mid-training measurement from perturbing the run it is measuring.
    predictor = copy.deepcopy(model)
    predictor.load_state_dict({k: v.to(device)
                               for k, v in weight_decode(weight_blob, template).items()},
                              strict=False)
    predictor.eval()

    # A cell is corrected when the prediction is wrong, and also when it is right but only just:
    # a residual entry that agrees with the prediction costs bytes and buys immunity to any
    # arithmetic that moves a logit by less than half the margin. The symbol written is the true
    # one either way, so the decoder is unchanged and older compressed outputs still decode.
    #
    # This is done a tile at a time, into the residual and nothing else. Holding the prediction,
    # the too-close mask and the wrong mask as whole planes alongside it costs four arrays the
    # size of the matrix, which on a cluster of 148,868 members is 18 GB of host memory for
    # quantities that are only ever read within the tile that produced them.
    Q = np.zeros((len(origins), config.latent_channels, n_freq, config.tile_width), np.int16)
    residual = np.zeros((height, width), np.uint8)
    correct = 0
    guarded = 0
    step = max(1, config.resolved_batch(height) // 2)
    with torch.no_grad():
        for start in range(0, len(origins), step):
            batch = tiles[start:start + step].to(device).long()
            coefficients = predictor.encode(one_hot(batch))
            quantised = torch.round(coefficients / config.scale).to(torch.int16)
            Q[start:start + step] = quantised.cpu().numpy()
            logits = predictor.decode(quantised.to(coefficients.dtype) * config.scale)
            argmax = logits.argmax(1).cpu().numpy().astype(np.uint8) + 1
            narrow = None
            if config.decision_margin:
                top2 = logits.topk(2, dim=1).values
                narrow = ((top2[:, 0] - top2[:, 1]) < config.decision_margin).cpu().numpy()
                del top2
            for k, (i, j) in enumerate(origins[start:start + step]):
                # Tiles cover the padded shape; only the part inside the matrix is real.
                rows, cols = min(i + block_height, height) - i, min(j + config.tile_width,
                                                                    width) - j
                if rows <= 0 or cols <= 0:
                    continue
                truth = matrix[i:i + rows, j:j + cols]
                missed = argmax[k][:rows, :cols] != truth
                correct += int(rows * cols - missed.sum())
                take = missed
                if narrow is not None:
                    close = narrow[k][:rows, :cols]
                    guarded += int((close & ~missed).sum())
                    take = missed | close
                residual[i:i + rows, j:j + cols] = np.where(take, truth, 0)
            del batch, coefficients, quantised, logits, argmax, narrow

    latent_encode, latent_decode = coders.LATENT_CODERS[config.latent_coder]
    residual_encode, residual_decode = coders.RESIDUAL_CODERS[config.residual_coder]
    latent_blob = latent_encode(Q, level=level)
    residual_blob = residual_encode(residual, level=level)
    if not np.array_equal(latent_decode(latent_blob), Q):
        raise AssertionError(
            f"latent coder {config.latent_coder!r} does not reconstruct its input")
    if not np.array_equal(residual_decode(residual_blob), residual):
        raise AssertionError(
            f"residual coder {config.residual_coder!r} does not reconstruct its input")

    result = Result(
        weights=len(weight_blob), latent=len(latent_blob), residual=len(residual_blob),
        accuracy=correct / (height * width),
        guarded=guarded,
        height=height, width=width,
        bases=int((matrix != 5).sum()), gaps=int((matrix == 5).sum()),
        frequencies=n_freq,
        frequencies_used=int(sum(1 for k in range(n_freq) if Q[:, :, k, :].any())),
        blobs={"weights": weight_blob, "latent": latent_blob, "residual": residual_blob},
    )
    result.total = result.weights + result.latent + result.residual
    del Q
    return result


def encode_matrix(matrix: np.ndarray, config: Config, verify: bool = True,
                  weights: bytes | None = None, train: bool = True) -> Result:
    """Train on one aligned matrix and return its compressed output.

    The epoch kept is the one whose compressed output is smallest, measured by writing it. With
    `check_every = 0` the model is measured once, after the last epoch.

    Parameters
    ----------
    weights
        A weight blob from an earlier compressed output, in the format `config.weight_coder`
        writes. The model starts from it instead of being initialised from scratch.
    train
        When false, no optimiser step is taken and the supplied weights are used as they are.
        This is what makes a model transferable: the compressed output of the second cluster
        then carries a latent and a residual but no weights of its own, and whether that is a
        saving depends on whether the borrowed model predicts the second cluster well enough
        that its residual does not grow by more than the weights cost.

    A model can only be applied to a matrix whose block height matches the one it was trained
    at, because the transform is sized by it. Two collections therefore have to be given the
    same `block_height` for a model to move between them.
    """
    if weights is None and not train:
        raise ValueError("train=False needs weights to encode with")
    height, width = matrix.shape
    block_height = config.resolved_block_height(height)
    batch = config.resolved_batch(height)
    _set_determinism(config.seed)

    tiles, origins, padded_shape = _tiles(matrix, block_height, config.tile_width)
    model = Codec(block_height, config.latent_channels, config.vstride, tuple(config.kernel),
                  config.activation, config.depth, config.mid_channels,
                  config.transform, config.soft_threshold).to(config.device)
    if weights is not None:
        decode_weights = coders.WEIGHT_CODERS[config.weight_coder][1]
        model.load_state_dict({k: v.to(config.device) for k, v in
                               decode_weights(weights, model.decoder_state()).items()},
                              strict=False)
    if not train:
        # Nothing is learned, so there is no epoch to choose: write the compressed output once
        # with the weights as given.
        model.eval()
        result = _write(model, tiles, origins, matrix, padded_shape, config)
        result.epoch = 0
        result.train_seconds = 0.0
        if verify:
            result.verified = bool(np.array_equal(
                decode_matrix(result.blobs, config, height, width), matrix))
        del tiles, model
        if config.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return result

    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best, best_state, stale = None, None, 0
    curve = []
    started = time.time()
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(tiles))
        for start in range(0, len(tiles), batch):
            chunk = tiles[order[start:start + batch]].to(config.device).long()
            logits, coefficients = model(one_hot(chunk))
            loss = (F.cross_entropy(logits, (chunk - 1).clamp(0, 4))
                    + config.penalty_weight * _penalty(coefficients, config))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            del logits, coefficients, chunk
        last = epoch == config.epochs - 1
        if config.check_every and ((epoch + 1) % config.check_every == 0 or last):
            candidate = _write(model, tiles, origins, matrix, padded_shape, config,
                               level=config.measure_level)
            candidate.epoch = epoch + 1
            curve.append((epoch + 1, candidate.total))
            if best is None or candidate.total < best.total:
                best = candidate
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                # The measured compressed output is what is being minimised, so "no
                # improvement" is counted on it rather than on the loss. The full curve is
                # recorded so the patience value can be checked against what the runs did.
                stale += 1
                if config.patience and stale >= config.patience:
                    break
    if best is None:
        best = _write(model, tiles, origins, matrix, padded_shape, config,
                      level=config.measure_level)
        best.epoch = config.epochs
        best_state = copy.deepcopy(model.state_dict())

    # Everything above ranked epochs at the cheap level. The compressed output that is kept,
    # reported and verified is written once, here, at the full `gzip_level`.
    epoch, curve_so_far = best.epoch, list(curve)
    model.load_state_dict(best_state)
    best = _write(model, tiles, origins, matrix, padded_shape, config)
    best.epoch, curve = epoch, curve_so_far
    best.train_seconds = round(time.time() - started, 2)
    best.curve = curve
    best.stopped_early = bool(curve) and curve[-1][0] < config.epochs

    if verify:
        model.load_state_dict(best_state)
        rebuilt = decode_matrix(best.blobs, config, height, width)
        best.verified = bool(np.array_equal(rebuilt, matrix))

    del tiles, model
    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return best


def decode_matrix(blobs: dict, config: Config, height: int, width: int) -> np.ndarray:
    """Rebuild an aligned matrix from its three blobs and the configuration.

    The model is constructed from `config` alone and filled from the stored weights; nothing
    of the training run is consulted, which is what makes this a decode rather than a check.
    """
    _set_arithmetic()
    block_height = config.resolved_block_height(height)
    padded_h = int(math.ceil(height / block_height) * block_height)
    padded_w = int(math.ceil(width / config.tile_width) * config.tile_width)
    origins = tile_origins(padded_h, padded_w, block_height, config.tile_width)

    model = Codec(block_height, config.latent_channels, config.vstride, tuple(config.kernel),
                  config.activation, config.depth, config.mid_channels,
                  config.transform, config.soft_threshold).to(config.device)
    weight_decode = coders.WEIGHT_CODERS[config.weight_coder][1]
    model.load_state_dict({k: v.to(config.device) for k, v in
                           weight_decode(blobs["weights"], model.decoder_state()).items()},
                          strict=False)
    model.eval()

    Q = coders.LATENT_CODERS[config.latent_coder][1](blobs["latent"])
    residual = coders.RESIDUAL_CODERS[config.residual_coder][1](blobs["residual"])

    out = np.zeros((padded_h, padded_w), np.uint8)
    step = max(1, config.resolved_batch(height) // 2)
    with torch.no_grad():
        for start in range(0, len(origins), step):
            block = torch.tensor(Q[start:start + step].astype(np.float32)).to(config.device)
            argmax = model.decode(block * config.scale)
            argmax = argmax.argmax(1).cpu().numpy().astype(np.uint8) + 1
            for k, (i, j) in enumerate(origins[start:start + step]):
                out[i:i + block_height, j:j + config.tile_width] = argmax[k]

    matrix = out[:height, :width].copy()
    matrix[residual != 0] = residual[residual != 0]
    return matrix
