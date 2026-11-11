"""Every registered byte coder must reproduce its input exactly."""
import numpy as np
import pytest
import torch

from deduct import coders
from deduct.model import Codec


def _latent(rng, spread=3):
    """A latent shaped like a real one: one populated frequency plane, the rest empty."""
    Q = np.zeros((6, 1, 24, 512), np.int16)
    Q[:, :, 0, :] = rng.integers(-spread, spread + 1, (6, 1, 512))
    return Q


def _residual(rng, density=0.02):
    R = np.zeros((300, 4096), np.uint8)
    mask = rng.random(R.shape) < density
    R[mask] = rng.integers(1, 6, int(mask.sum()))
    return R


@pytest.mark.parametrize("name", sorted(coders.LATENT_CODERS))
def test_latent_round_trip(name):
    rng = np.random.default_rng(0)
    encode, decode = coders.LATENT_CODERS[name]
    for Q in (_latent(rng), _latent(rng, spread=200), np.zeros((2, 1, 8, 512), np.int16)):
        assert np.array_equal(decode(encode(Q)), Q)


@pytest.mark.parametrize("name", sorted(coders.RESIDUAL_CODERS))
def test_residual_round_trip(name):
    rng = np.random.default_rng(1)
    encode, decode = coders.RESIDUAL_CODERS[name]
    for R in (_residual(rng), _residual(rng, density=0.5), np.zeros((16, 512), np.uint8)):
        assert np.array_equal(decode(encode(R)), R)


@pytest.mark.parametrize("name", sorted(coders.WEIGHT_CODERS))
def test_weight_round_trip(name):
    encode, decode = coders.WEIGHT_CODERS[name]
    model = Codec(64, 1, 4, (3, 1), "relu")
    state = model.decoder_state()
    restored = decode(encode(state), state)
    assert set(restored) == set(state)
    for key, value in state.items():
        # raw16 rounds, so the check is against the value the decoder will actually hold
        expected = value.detach().cpu()
        if name == "raw16":
            expected = expected.to(torch.float16).to(torch.float32)
        assert torch.equal(restored[key].cpu(), expected)


def test_latent_survives_values_beyond_int8():
    """A quantised coefficient outside [-128, 127] must not wrap."""
    Q = np.zeros((2, 1, 4, 512), np.int16)
    Q[:, :, 0, :] = 300
    Q[0, 0, 1, 0] = -400
    for encode, decode in coders.LATENT_CODERS.values():
        assert np.array_equal(decode(encode(Q)), Q)
