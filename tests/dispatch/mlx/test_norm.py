import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytest.importorskip("mlx.core")

from pytensor.compile.mode import MLX

from pytensor_ml.layers import GroupNorm, LayerNorm


@pytest.mark.parametrize(
    "build",
    [
        lambda: GroupNorm("group", n_groups=4, n_in=64, epsilon=1e-6),
        lambda: LayerNorm("layer", n_in=64),
    ],
    ids=["group", "layer"],
)
@pytest.mark.parametrize("scale", [1.0, 3000.0], ids=["small", "past_the_square_root"])
def test_a_norm_survives_float16_activations_it_cannot_square(build, scale):
    """mlx executes float16 natively, so this is where the overflow actually happens: an activation
    of 3000 squares to 9e6 against float16's 65504 ceiling. A diffusion decoder reaches those
    magnitudes on its way out, so the statistics have to accumulate wider than the input."""
    values = (np.random.default_rng(0).normal(size=(2, 16, 64)) * scale).astype("float16")

    X = pt.tensor("X", shape=values.shape, dtype="float16")
    computed = np.asarray(pytensor.function([X], build()(X), mode=MLX)(values))

    assert not np.isnan(computed).any()
    np.testing.assert_allclose(computed.astype("float64").std(), 1.0, rtol=0.05)
