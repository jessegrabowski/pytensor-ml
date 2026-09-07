import pytensor.tensor as pt
import pytest

from pytensor_ml.layers import Linear
from pytensor_ml.models import (
    architecture_name,
    build_from_config,
    channels_last,
    register_builder,
)
from pytensor_ml.models.registry import _BUILDERS


@pytest.fixture
def isolated_builder_registry():
    """Undo any builder registration a test performs. The registry is module-level, so a leaked entry
    would answer for every test that ran afterwards."""
    registered = dict(_BUILDERS)
    yield
    _BUILDERS.clear()
    _BUILDERS.update(registered)


@pytest.mark.parametrize(
    "config",
    [
        {"_class_name": "ToyEncoder", "n_in": 4, "n_out": 3},
        {"model_type": "toy_encoder", "architectures": ["ToyEncoder"], "n_in": 4, "n_out": 3},
    ],
    ids=["diffusers", "transformers"],
)
def test_registry_dispatches_on_the_declared_class(config, isolated_builder_registry):
    """Diffusers and transformers spell the architecture differently, and both resolve to the class
    name a builder registers under."""

    @register_builder("ToyEncoder")
    def build_toy_encoder(cfg, keys):
        X = pt.tensor("X", shape=(None, cfg["n_in"]))
        fc = Linear("fc", n_in=cfg["n_in"], n_out=cfg["n_out"])
        keys.bind(fc.W, "fc.weight", transform=channels_last)
        return [X], fc(X)

    data_inputs, outputs, keys = build_from_config(config)

    assert architecture_name(config) == "ToyEncoder"
    assert [variable.name for variable in data_inputs] == ["X"]
    assert outputs.type.shape == (None, 3)
    assert keys.keys() == {"fc.weight"}


def test_a_config_naming_no_architecture_raises():
    with pytest.raises(ValueError, match="names no architecture"):
        build_from_config({"hidden_size": 8})


def test_an_unregistered_architecture_raises():
    with pytest.raises(ValueError, match="No builder is registered for 'NotARealModel'"):
        build_from_config({"_class_name": "NotARealModel"})


def test_registering_an_architecture_twice_raises(isolated_builder_registry):
    """Import order would otherwise decide which builder answers, silently."""

    @register_builder("ToyEncoder")
    def build_toy_encoder(cfg, keys):
        raise AssertionError("not called")

    with pytest.raises(ValueError, match="already has a builder"):

        @register_builder("ToyEncoder")
        def build_toy_encoder_again(cfg, keys):
            raise AssertionError("not called")
