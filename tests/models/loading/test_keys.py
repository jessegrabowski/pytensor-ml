import numpy as np
import pytensor
import pytest

from pytensor_ml.layers import Linear
from pytensor_ml.models import KeyMap, channels_last


def test_key_map_builds_paths_from_nested_scopes():
    """A builder records the module path it is already inside, rather than a loader rediscovering it."""
    keys = KeyMap()
    with keys.scope("text_model", "encoder"):
        for i in range(2):
            layer = Linear(f"fc_{i}", n_in=4, n_out=4)
            with keys.scope("layers", str(i), "mlp", "fc1"):
                keys.bind(layer.W, "weight")
                keys.bind(layer.b, "bias")

    assert keys.keys() == {
        "text_model.encoder.layers.0.mlp.fc1.weight",
        "text_model.encoder.layers.0.mlp.fc1.bias",
        "text_model.encoder.layers.1.mlp.fc1.weight",
        "text_model.encoder.layers.1.mlp.fc1.bias",
    }
    assert len(keys) == 4


def test_key_map_holds_parameters_by_identity():
    """Two layers built with the same name own identically-named parameters. Keying on the object is
    what keeps them apart, and is why the map holds parameters rather than strings."""
    first = Linear("fc", n_in=4, n_out=4)
    second = Linear("fc", n_in=4, n_out=4)
    assert first.W.name == second.W.name

    keys = KeyMap()
    keys.bind(first.W, "first.weight")
    keys.bind(second.W, "second.weight")

    assert keys.key_for(first.W) == "first.weight"
    assert keys.key_for(second.W) == "second.weight"


def test_key_map_rejects_binding_one_parameter_twice():
    keys = KeyMap()
    layer = Linear("fc", n_in=4, n_out=4)
    keys.bind(layer.W, "encoder.weight")

    with pytest.raises(ValueError, match=r"already bound to 'encoder\.weight'"):
        keys.bind(layer.W, "decoder.weight")


def test_key_map_rejects_binding_one_key_twice():
    keys = KeyMap()
    keys.bind(Linear("a", n_in=4, n_out=4).W, "shared.weight")

    with pytest.raises(ValueError, match="cannot load into two parameters"):
        keys.bind(Linear("b", n_in=4, n_out=4).W, "shared.weight")


def test_key_map_scope_unwinds_when_the_body_raises():
    """A leaked prefix would silently misname every key bound after it."""
    keys = KeyMap()
    with pytest.raises(RuntimeError):
        with keys.scope("encoder"):
            raise RuntimeError("builder blew up")

    keys.bind(Linear("fc", n_in=4, n_out=4).W, "decoder.weight")
    assert keys.keys() == {"decoder.weight"}


@pytest.mark.parametrize(
    "checkpoint_shape, expected",
    [((16, 3), (3, 16)), ((16, 3, 5), (5, 3, 16)), ((16, 3, 3, 5), (3, 5, 3, 16))],
    ids=["dense", "conv1d", "conv2d"],
)
def test_channels_last_moves_the_checkpoint_axes(checkpoint_shape, expected):
    """HF stores (out, in, *kernel) and this library stores (*kernel, in, out). One move covers every
    rank, and the element-wise check is what a shape assertion alone would miss on a square kernel."""
    checkpoint = np.arange(np.prod(checkpoint_shape)).reshape(checkpoint_shape)

    moved = channels_last(checkpoint)

    assert moved.shape == expected
    for index in np.ndindex(*checkpoint_shape):
        out_axis, in_axis, *spatial = index
        assert moved[(*spatial, in_axis, out_axis)] == checkpoint[index]


@pytest.mark.parametrize("floatX", ["float32", "float16"])
def test_load_casts_to_the_parameter_dtype(floatX):
    """A parameter's dtype is fixed by floatX when the layer builds it; loading cannot change it.
    Building at float16 is what lets a jax or mlx graph keep the checkpoint's own fp16."""
    with pytensor.config.change_flags(floatX=floatX):
        layer = Linear("fc", n_in=4, n_out=3)
    keys = KeyMap()
    keys.bind(layer.W, "fc.weight", transform=channels_last)
    keys.bind(layer.b, "fc.bias")

    checkpoint = {
        "fc.weight": np.ones((3, 4), dtype="float16"),
        "fc.bias": np.ones(3, dtype="float16"),
    }
    keys.load(checkpoint.__getitem__, checkpoint)

    assert layer.W.get_value().dtype == floatX
    assert layer.b.get_value().dtype == floatX


def test_load_applies_the_bound_transform():
    layer = Linear("fc", n_in=4, n_out=3)
    keys = KeyMap()
    keys.bind(layer.W, "fc.weight", transform=channels_last)

    checkpoint_weight = np.arange(12, dtype="float32").reshape(3, 4)
    keys.load({"fc.weight": checkpoint_weight}.__getitem__, ["fc.weight"])

    np.testing.assert_array_equal(layer.W.get_value(), checkpoint_weight.T)


def test_load_rejects_a_shape_mismatch():
    """The transform is the thing most likely to be wrong, and on a square kernel a wrong one keeps
    the right shape -- so this fires on everything else."""
    layer = Linear("fc", n_in=4, n_out=3)
    keys = KeyMap()
    keys.bind(layer.W, "fc.weight")

    with pytest.raises(ValueError, match=r"'fc.weight' holds \(3, 4\) but fc_W needs \(4, 3\)"):
        keys.load({"fc.weight": np.ones((3, 4), dtype="float32")}.__getitem__, ["fc.weight"])


def test_load_raises_when_the_checkpoint_cannot_fill_a_parameter():
    """A parameter left at its initialization is a wrong model that runs, so this is the one
    direction that must be fatal -- and fatal before anything is stored."""
    layer = Linear("fc", n_in=4, n_out=3)
    keys = KeyMap()
    keys.bind(layer.W, "fc.weight", transform=channels_last)
    keys.bind(layer.b, "fc.bias")
    before = layer.W.get_value().copy()

    with pytest.raises(ValueError, match=r"no tensor for 1 bound parameter\(s\): 'fc.bias'"):
        keys.load({"fc.weight": np.ones((3, 4), dtype="float32")}.__getitem__, ["fc.weight"])

    np.testing.assert_array_equal(layer.W.get_value(), before)


def test_load_reports_a_surplus_tensor_and_proceeds():
    """Every parameter still got a value, so a spare tensor is reported rather than fatal. Older CLIP
    checkpoints carry a serialized position_ids that nothing needs."""
    layer = Linear("fc", n_in=4, n_out=3)
    keys = KeyMap()
    keys.bind(layer.W, "fc.weight", transform=channels_last)
    keys.bind(layer.b, "fc.bias")

    checkpoint = {
        "fc.weight": np.ones((3, 4), dtype="float32"),
        "fc.bias": np.zeros(3, dtype="float32"),
        "fc.position_ids": np.arange(77),
    }
    surplus = keys.load(checkpoint.__getitem__, checkpoint)

    assert surplus == ["fc.position_ids"]
    np.testing.assert_array_equal(layer.W.get_value(), np.ones((4, 3)))


def test_load_reports_a_dtype_numpy_cannot_read():
    """safetensors raises a bare TypeError for bfloat16, which says nothing about what to do next."""
    keys = KeyMap()
    linear = Linear("fc", n_in=2, n_out=2)
    keys.bind(linear.W, "fc.weight")
    keys.bind(linear.b, "fc.bias")

    def read(key):
        raise TypeError("unsupported dtype BF16")

    with pytest.raises(TypeError, match="Re-save the checkpoint"):
        keys.load(read, ["fc.weight", "fc.bias"])


@pytest.mark.parametrize(
    "checkpoint_prefix",
    ["", "transformer."],
    ids=["published", "saved_from_the_head_class"],
)
def test_load_matches_a_checkpoint_that_adds_or_drops_the_model_prefix(checkpoint_prefix):
    """Whether the leading segment is there depends on the class the checkpoint was saved from, and
    the published files invert what transformers writes today."""
    keys = KeyMap()
    linear = Linear("fc", n_in=2, n_out=2)
    keys.bind(linear.W, "h.0.weight")
    keys.bind(linear.b, "h.0.bias")
    checkpoint = {
        f"{checkpoint_prefix}h.0.weight": np.ones((2, 2)),
        f"{checkpoint_prefix}h.0.bias": np.full(2, 3.0),
    }

    assert keys.load(checkpoint.__getitem__, checkpoint) == []
    np.testing.assert_array_equal(linear.b.get_value(), 3.0)
