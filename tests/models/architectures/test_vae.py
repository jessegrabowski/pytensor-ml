import json
import os

from pathlib import Path

import numpy as np
import pytensor
import pytest

from safetensors import safe_open

from pytensor_ml.models import build_from_config
from pytensor_ml.pretrained import from_pretrained
from tests.conftest import REFERENCE_DATA

TINY_VAE = json.loads((REFERENCE_DATA / "tiny_vae" / "config.json").read_text())

SDXL_VAE = os.environ.get("PYTENSOR_ML_SDXL_VAE")


@pytest.mark.parametrize("direction", ["encode", "decode"])
def test_a_loaded_autoencoder_computes_what_diffusers_computes(direction):
    """Both halves are stacks of blocks whose every wiring mistake -- a resnet reading the wrong
    stage's width, a downsample padded symmetrically instead of on two sides -- still produces an
    output of the right shape. See tests/data for the generator."""
    expected = np.load(REFERENCE_DATA / "tiny_vae" / "expected_outputs.npz")
    (image, latent), (mean, log_variance, sample) = from_pretrained(REFERENCE_DATA / "tiny_vae")

    if direction == "encode":
        wanted, computed = (
            ["mean", "log_variance"],
            pytensor.function([image], [mean, log_variance])(expected["image"]),
        )
    else:
        wanted, computed = ["sample"], [pytensor.function([latent], sample)(expected["latent"])]

    for name, got in zip(wanted, computed, strict=True):
        np.testing.assert_allclose(got, expected[name], rtol=1e-4, atol=1e-5, err_msg=name)


def test_the_builder_claims_every_tensor_the_checkpoint_holds():
    """A parameter with no key keeps its initialization, which is a wrong model that runs, and a
    tensor with no parameter means a whole module went unbuilt."""
    _, _, keys = build_from_config(TINY_VAE)
    with safe_open(
        REFERENCE_DATA / "tiny_vae" / "diffusion_pytorch_model.safetensors", framework="numpy"
    ) as checkpoint:
        held = set(checkpoint.keys())

    assert keys.keys() == held


def test_each_direction_reads_only_its_own_input():
    """The two halves share weights and nothing else, so compiling one must not demand the other's
    input. Nothing fixes the resolution either, so one loaded autoencoder serves every size."""
    (image, latent), (mean, log_variance, sample) = build_from_config(TINY_VAE)[:2]

    assert pytensor.function([image], [mean, log_variance]).maker.fgraph.inputs
    assert pytensor.function([latent], sample).maker.fgraph.inputs
    assert sample.type.shape == (None, None, None, TINY_VAE["out_channels"])
    assert mean.type.shape == (None, None, None, TINY_VAE["latent_channels"])


@pytest.mark.parametrize(
    "override, match",
    [
        ({"up_block_types": ["UpDecoderBlock2D", "AttnUpDecoderBlock2D"]}, "up_block_types"),
        ({"act_fn": "mish"}, "act_fn"),
        ({"mid_block_add_attention": False}, "mid_block_add_attention"),
    ],
    ids=["attention_up_block", "activation", "no_mid_attention"],
)
def test_a_config_describing_a_different_autoencoder_raises(override, match):
    """Each of these loads cleanly against a subset of the keys and returns a wrong image."""
    with pytest.raises(ValueError, match=match):
        build_from_config({**TINY_VAE, **override})


@pytest.mark.checkpoint
@pytest.mark.skipif(SDXL_VAE is None, reason="PYTENSOR_ML_SDXL_VAE is not set")
@pytest.mark.parametrize("name", ["mean", "log_variance", "sample"])
def test_the_real_sdxl_autoencoder_computes_what_diffusers_computes(name):
    """The tiny fixture pins the wiring at two stages; this pins it at the widths, depth and group
    count SDXL actually ships, against weights nobody here chose. Point PYTENSOR_ML_SDXL_VAE at a
    downloaded ``vae`` component directory to run it."""
    expected = np.load(REFERENCE_DATA / "sdxl_vae.npz")
    (image, latent), outputs = from_pretrained(Path(SDXL_VAE), variant="fp16")
    mean, log_variance, sample = outputs

    if name == "sample":
        computed = pytensor.function([latent], sample)(expected["latent"])
    else:
        computed = pytensor.function([image], {"mean": mean, "log_variance": log_variance}[name])(
            expected["image"]
        )

    np.testing.assert_allclose(computed, expected[name], rtol=1e-3, atol=1e-3)


@pytest.mark.checkpoint
@pytest.mark.skipif(SDXL_VAE is None, reason="PYTENSOR_ML_SDXL_VAE is not set")
def test_the_real_sdxl_autoencoder_reconstructs_an_image_it_encoded():
    """Agreement with diffusers holds even for a pipeline that is wrong in the same way at both
    ends. Reconstructing a smooth image is the check that the pair means something: the error has
    to beat what predicting a flat grey scores."""
    expected = np.load(REFERENCE_DATA / "sdxl_vae.npz")
    (image, latent), (mean, _, sample) = from_pretrained(Path(SDXL_VAE), variant="fp16")

    encoded = pytensor.function([image], mean)(expected["image"])
    reconstruction = pytensor.function([latent], sample)(encoded)

    error = np.abs(reconstruction - expected["image"]).mean()
    flat_grey = np.abs(expected["image"]).mean()
    assert error < flat_grey / 2, (
        f"reconstruction error {error:.3f} against baseline {flat_grey:.3f}"
    )
