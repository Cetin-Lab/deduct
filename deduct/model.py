"""The codec network: a convolutional encoder, an orthonormal vertical DCT, and a decoder.

The input is one tile of an aligned matrix (rows are cluster members, columns are aligned
positions), one-hot encoded over the five symbols {A, C, G, T, gap}. The two axes are treated
differently throughout:

    rows      members of a cluster, nearly identical to one another. Redundant, and the only
              axis the encoder downsamples.
    columns   aligned positions. Adjacent columns are independent, so they are never strided
              and the convolution kernel is one column wide; widening it or striding it makes
              the latent transport the sequence instead of compressing it.

Only the decoder is written to the compressed output. The DCT matrix is regenerated from the
block height rather than stored, and the encoder is never evaluated during decompression.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky": lambda: nn.LeakyReLU(0.1),
    "tanh": nn.Tanh,
    "none": nn.Identity,
}


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of size `n`, as float32.

    Row `k` is the `k`-th basis vector, so a forward transform is `D @ x` and its inverse is
    `D.T @ y`.
    """
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2.0 / n)
    m[0] *= 1 / np.sqrt(2)
    return m.astype(np.float32)


class Codec(nn.Module):
    """Convolutional autoencoder with a vertical block-DCT bottleneck.

    Parameters
    ----------
    block_height
        Number of rows in a tile. Must be divisible by `vstride`.
    latent_channels
        Channels at the bottleneck, after the encoder and before the transform.
    vstride
        Total downsampling along the row axis, spread over the encoder's layers. Must be a
        power of two. The column axis is never downsampled.
    kernel
        `(height, width)` of every convolution. The height must be odd, so that a padding of
        `height // 2` keeps the output height exactly `input // stride`. The width is normally
        1, which keeps each column's latent a function of that column alone; any width is
        accepted, including even ones, which need asymmetric padding to leave the latent the
        same width as the tile.
    activation
        Name from `ACTIVATIONS`, applied between layers but not before the transform.
    depth
        Number of convolution layers. Defaults to `log2(vstride)`, which gives every layer a
        stride of 2. Setting it lower concentrates the stride into fewer layers and keeps the
        parameter count from growing with `vstride`; setting it higher adds layers at stride 1.
    mid_channels
        Width of the intermediate layers.

    Raises
    ------
    ValueError
        If the geometry is inconsistent: a non-power-of-two `vstride`, a `block_height` that
        is not a multiple of it, or an even kernel height.
    """

    def __init__(self, block_height: int, latent_channels: int = 1, vstride: int = 4,
                 kernel: tuple[int, int] = (3, 1), activation: str = "relu",
                 depth: int | None = None, mid_channels: int = 32,
                 transform: str = "dct", soft_threshold: bool = True):
        super().__init__()
        if transform not in ("dct", "identity", "learned"):
            raise ValueError(f"unknown transform {transform!r}")
        if vstride < 1 or vstride & (vstride - 1):
            raise ValueError(f"vstride must be a power of two, got {vstride}")
        if block_height % vstride:
            raise ValueError(f"block_height {block_height} is not a multiple of vstride {vstride}")
        kh, kw = kernel
        if kh % 2 == 0:
            raise ValueError(f"kernel height must be odd, got {kh}")
        if kw < 1:
            raise ValueError(f"kernel width must be at least 1, got {kw}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; have {sorted(ACTIVATIONS)}")

        n_layers = depth or max(1, int(math.log2(vstride)))
        strides = _spread_stride(vstride, n_layers)
        # The column axis is padded outside the convolution so that even kernel widths work.
        # A width-w kernel at stride 1 needs w-1 columns of padding to leave the width
        # unchanged, and w-1 is odd when w is even, so it cannot be split evenly the way
        # Conv2d's `padding` requires. Splitting it explicitly as (w//2, w-1-w//2) reproduces
        # symmetric padding exactly for odd widths and makes even ones representable.
        self.pad_left = kw // 2
        self.pad_right = kw - 1 - kw // 2
        pad = (kh // 2, 0)
        act = ACTIVATIONS[activation]

        widths = [5] + [mid_channels] * (n_layers - 1) + [latent_channels]
        encoder = []
        for i in range(n_layers):
            encoder.append(nn.Conv2d(widths[i], widths[i + 1], (kh, kw),
                                     stride=(strides[i], 1), padding=pad))
            if i < n_layers - 1:
                encoder.append(act())
        self.encoder = nn.Sequential(*encoder)

        widths = [latent_channels] + [mid_channels] * (n_layers - 1) + [5]
        decoder = []
        for i in range(n_layers):
            s = strides[n_layers - 1 - i]
            decoder.append(nn.ConvTranspose2d(widths[i], widths[i + 1], (kh, kw),
                                              stride=(s, 1), padding=pad,
                                              output_padding=(s - 1, 0)))
            if i < n_layers - 1:
                decoder.append(act())
        self.decoder = nn.Sequential(*decoder)

        # The transform is a square matrix over the latent's row axis. `dct` is fixed and
        # regenerated by the decoder, so it costs nothing in the compressed output; `identity`
        # is the ablation that removes the transform while leaving the geometry alone; `learned`
        # starts at the DCT and is trained, so it is the only one whose entries have to be
        # stored, and it is counted like any other parameter.
        self.transform = transform
        n_freq = block_height // vstride
        basis = torch.tensor(dct_matrix(n_freq)) if transform != "identity" else torch.eye(n_freq)
        if transform == "learned":
            self.dct = nn.Parameter(basis)
        else:
            self.register_buffer("dct", basis)
        self.use_threshold = soft_threshold
        self.log_threshold = nn.Parameter(torch.tensor(-3.0))

    # ------------------------------------------------------------------ transform

    def forward_dct(self, x: torch.Tensor) -> torch.Tensor:
        """Transform along the row axis of the latent."""
        return torch.einsum("ij,bcjw->bciw", self.dct, x)

    def inverse_dct(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ji,bcjw->bciw", self.dct, x)

    def soft_threshold(self, c: torch.Tensor) -> torch.Tensor:
        """Shrink coefficients toward zero by a learned amount, or pass them through."""
        if not self.use_threshold:
            return c
        return torch.sign(c) * F.relu(c.abs() - torch.exp(self.log_threshold))

    # ------------------------------------------------------------------ halves

    def _encode_columns(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder with the column padding applied around each convolution."""
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                x = layer(F.pad(x, (self.pad_left, self.pad_right)))
            else:
                x = layer(x)
        return x

    def _decode_columns(self, x: torch.Tensor) -> torch.Tensor:
        """Run the decoder, cropping each transposed convolution back to the input width."""
        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose2d):
                width = x.shape[-1]
                x = layer(x)[..., self.pad_left:self.pad_left + width]
            else:
                x = layer(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """One-hot tile `(B, 5, H, W)` to coefficients `(B, C, H/vstride, W)`."""
        return self.soft_threshold(self.forward_dct(self._encode_columns(x)))

    def decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Coefficients back to per-symbol logits `(B, 5, H, W)`."""
        return self._decode_columns(self.inverse_dct(coefficients))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(logits, coefficients)`. The coefficients are needed for the loss penalty."""
        c = self.encode(x)
        return self.decode(c), c

    # ------------------------------------------------------------------ compressed output

    def decoder_state(self) -> dict:
        """The parameters the compressed output must carry.

        Only the decoder half: decompression never evaluates the encoder. The DCT buffer is
        excluded because an identically configured model regenerates it.
        """
        keep = {k: v for k, v in self.state_dict().items() if k.startswith("decoder")}
        if self.transform == "learned":
            # A trained transform is not regenerable, so the decoder needs it in the
            # compressed output.
            keep["dct"] = self.state_dict()["dct"]
        return keep


def _spread_stride(total: int, n_layers: int) -> list[int]:
    """Factor `total` into `n_layers` per-layer strides, twos first and the remainder last."""
    strides = [1] * n_layers
    remaining = total
    for i in range(n_layers):
        take = 2 if (remaining > 1 and i < n_layers - 1) else remaining
        strides[i] = take
        remaining //= take
    return strides


def one_hot(tile: torch.Tensor) -> torch.Tensor:
    """`(B, H, W)` of symbols 1..5 to `(B, 5, H, W)` of indicators, on the tile's device."""
    out = torch.zeros((tile.shape[0], 5) + tile.shape[1:], dtype=torch.float32,
                      device=tile.device)
    for v in range(1, 6):
        out[:, v - 1] = tile == v
    return out


def tile_origins(height: int, width: int, block_height: int, tile_width: int):
    """Top-left corner of every whole tile that fits in a `height` by `width` array."""
    return [(i, j)
            for i in range(0, height - block_height + 1, block_height)
            for j in range(0, width - tile_width + 1, tile_width)]
