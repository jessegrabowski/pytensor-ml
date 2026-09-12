from pathlib import Path

import numpy as np
import pytensor
import pytest

from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.op import RandomVariable

from pytensor_ml.layers import Input
from pytensor_ml.models import ResnetBlock2D, channels_last

REFERENCE = np.load(Path(__file__).parents[2] / "data" / "resnet_block_2d.npz")


def load_diffusers_weights(block, weights):
    """Fill a block from the tensors diffusers names in its state_dict."""
    for name, norm in [("norm1", block.norm1), ("norm2", block.norm2)]:
        norm.scale.set_value(weights[f"{name}.weight"])
        norm.loc.set_value(weights[f"{name}.bias"])

    for name, layer in [
        ("conv1", block.conv1),
        ("conv2", block.conv2),
        ("conv_shortcut", block.conv_shortcut),
        ("time_emb_proj", block.time_emb_proj),
    ]:
        if layer is None:
            continue
        layer.W.set_value(channels_last(weights[f"{name}.weight"]))
        layer.b.set_value(weights[f"{name}.bias"])


@pytest.mark.parametrize(
    "case, out_channels, temb_channels",
    [
        ("same_width", None, None),
        ("wider", 6, None),
        ("conditioned", 6, 5),
    ],
)
def test_a_resnet_block_computes_what_diffusers_computes(case, out_channels, temb_channels):
    """The block is the VAE's and the U-Net's whole body, and every part of it -- where the norms
    sit relative to the convolutions, whether the shortcut is learned, where the timestep embedding
    is added -- gives a plausible wrong answer if it is off. See tests/data for the generator."""
    weights = {
        key.split("/")[-1]: REFERENCE[key]
        for key in REFERENCE
        if key.startswith(f"{case}/weights/")
    }

    X = Input("X", shape=(None, 8, 8, 4))
    inputs, values = [X], [REFERENCE[f"{case}/X"]]
    if temb_channels:
        inputs.append(Input("temb", shape=(None, temb_channels)))
        values.append(REFERENCE[f"{case}/temb"])

    block = ResnetBlock2D(
        "block",
        in_channels=4,
        out_channels=out_channels,
        temb_channels=temb_channels,
        n_groups=2,
    )
    output = block(*inputs)
    load_diffusers_weights(block, weights)

    computed = pytensor.function(inputs, output)(*values)

    np.testing.assert_allclose(computed, REFERENCE[f"{case}/output"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, False),
        ({"out_channels": 16}, True),
        ({"conv_shortcut": True}, True),
    ],
    ids=["same_width", "wider", "asked_for"],
)
def test_the_shortcut_is_built_only_where_the_addition_needs_it(kwargs, expected):
    """A block that changes nothing should add the input straight back rather than carry a 1x1
    convolution's worth of weights, and asking for one anyway should still get one."""
    block = ResnetBlock2D("block", in_channels=8, n_groups=2, **kwargs)

    assert (block.conv_shortcut is not None) == expected


def test_a_widening_block_cannot_refuse_its_shortcut():
    """Without the shortcut the addition has nothing to add, and the shape error it reaches instead
    names neither the block nor the argument that caused it."""
    with pytest.raises(ValueError, match="conv_shortcut=False"):
        ResnetBlock2D("wider", in_channels=4, out_channels=8, n_groups=2, conv_shortcut=False)


@pytest.mark.parametrize("dropout", [0.0, 0.1], ids=["no_dropout", "dropout"])
def test_dropout_adds_rng_only_when_enabled(dropout):
    out = ResnetBlock2D("block", in_channels=4, n_groups=2, dropout=dropout)(
        Input("X", shape=(2, 8, 8, 4))
    )
    has_rng = any(
        node.owner and isinstance(node.owner.op, RandomVariable) for node in ancestors([out])
    )

    assert has_rng == (dropout > 0)


def test_a_block_and_its_timestep_embedding_have_to_agree():
    X = Input("X", shape=(None, 8, 8, 4))
    temb = Input("temb", shape=(None, 5))

    with pytest.raises(ValueError, match="did not pass"):
        ResnetBlock2D("conditioned", in_channels=4, n_groups=2, temb_channels=5)(X)

    with pytest.raises(ValueError, match="no projection"):
        ResnetBlock2D("plain", in_channels=4, n_groups=2)(X, temb)
