from pathlib import Path

import numpy as np
import pytensor
import pytest

from pytensor_ml.layers import Input
from pytensor_ml.models import AttentionBlock2D, channels_last

REFERENCE = np.load(Path(__file__).parents[2] / "data" / "attention_block_2d.npz")


def load_diffusers_weights(block, weights):
    """Fill a block from the tensors diffusers names in its state_dict."""
    block.norm.scale.set_value(weights["group_norm.weight"])
    block.norm.loc.set_value(weights["group_norm.bias"])

    for name, projection in [
        ("to_q", block.attn.q_proj),
        ("to_k", block.attn.k_proj),
        ("to_v", block.attn.v_proj),
        ("to_out.0", block.attn.out_proj),
    ]:
        projection.W.set_value(channels_last(weights[f"{name}.weight"]))
        projection.b.set_value(weights[f"{name}.bias"])


@pytest.mark.parametrize("case, n_head", [("one_head", 1), ("four_heads", 4)])
def test_an_attention_block_computes_what_diffusers_computes(case, n_head):
    """Flattening the map into a sequence and restoring it afterwards has to survive the round trip
    in the same pixel order, and the heads have to split the channels the way diffusers splits them.
    Either mistake gives a plausible feature map. See tests/data for the generator."""
    weights = {
        key.split("/")[-1]: REFERENCE[key]
        for key in REFERENCE
        if key.startswith(f"{case}/weights/")
    }

    X = Input("X", shape=(None, 4, 4, 8))
    block = AttentionBlock2D("block", channels=8, n_head=n_head, n_groups=2)
    output = block(X)
    load_diffusers_weights(block, weights)

    computed = pytensor.function([X], output)(REFERENCE[f"{case}/X"])

    np.testing.assert_allclose(computed, REFERENCE[f"{case}/output"], rtol=1e-5, atol=1e-5)


def test_the_block_leaves_the_feature_map_shaped_as_it_found_it():
    """The residual addition is what forces this, so a reshape that transposed height and width
    would still add cleanly on a square map and silently scramble a rectangular one."""
    output = AttentionBlock2D("block", channels=8, n_groups=2)(Input("X", shape=(None, 6, 4, 8)))

    assert output.type.shape == (None, 6, 4, 8)
