import json

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.type import RandomGeneratorType, random_generator_type
from safetensors.numpy import save_file

from pytensor_ml.activations import ReLU
from pytensor_ml.checkpoint import jsonable_rng_state
from pytensor_ml.layers import BatchNorm, Dropout, Embedding, Linear, Sequential
from pytensor_ml.models import (
    KeyMap,
    architecture_name,
    bind_linear,
    build_from_config,
    channels_last,
    register_builder,
)
from pytensor_ml.models.registry import _BUILDERS
from pytensor_ml.params import NonTrainableParameter, TrainableParameter
from pytensor_ml.pretrained import (
    _detect_format,
    from_pretrained,
    load_network,
    save_network,
    save_pretrained,
)
from pytensor_ml.pytensorf import (
    collect_shared_variables,
    collect_trainable_params,
    function,
)
from pytensor_ml.state import (
    NormalInitializer,
    UnrecordedInitializer,
    initialize_params,
    initializer,
)
from tests.conftest import constant, he_normal

floatX = pytensor.config.floatX


def build_initialized_network(seed=0):
    rng = np.random.default_rng(seed)
    X = pt.matrix("X")
    output = Sequential(Linear("fc1", n_in=4, n_out=8), ReLU(), Linear("fc2", n_in=8, n_out=2))(X)
    for parameter in collect_trainable_params(output):
        value = rng.normal(size=parameter.get_value().shape)
        parameter.set_value(value.astype(parameter.type.dtype))
    return X, output


def predict(inputs, output, x_value):
    # Plain literals default to float64, which does not fit a float32 graph.
    return pytensor.function(inputs, output)(np.asarray(x_value, dtype=floatX))


def test_from_pretrained_restores_architecture_and_weights(tmp_path):
    X, output = build_initialized_network()
    x_value = np.random.default_rng(1).normal(size=(5, 4))
    expected = predict([X], output, x_value)

    save_pretrained(output, tmp_path)
    restored_inputs, restored_output = from_pretrained(tmp_path)

    np.testing.assert_allclose(
        predict(restored_inputs, restored_output, x_value), expected, rtol=1e-6
    )


def test_save_pretrained_writes_config_and_weights(tmp_path):
    X, output = build_initialized_network()
    save_pretrained(output, tmp_path)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()


def test_load_network_restores_trainable_params_holding_a_draw(tmp_path):
    """A rebuilt parameter holds a value for the same reason a freshly constructed one does, and holds it
    under the same law: a weight drawn, a bias at the zero it declares. The values are not the saved ones --
    this restores architecture, and `load_state` fills them by name."""
    X, output = build_initialized_network()
    save_network(output, tmp_path / "config.json")

    _, restored_output = load_network(tmp_path / "config.json")
    params = {p.name: p for p in collect_trainable_params(restored_output)}

    assert set(params) == {"fc1_W", "fc1_b", "fc2_W", "fc2_b"}
    assert all(isinstance(p, TrainableParameter) for p in params.values())
    for name in ("fc1_W", "fc2_W"):
        assert len(np.unique(params[name].get_value())) > 1, name
    for name in ("fc1_b", "fc2_b"):
        np.testing.assert_array_equal(params[name].get_value(), 0)


def test_from_pretrained_rejects_an_architecture_with_no_builder(tmp_path):
    # A HuggingFace config shares our filenames but is a hyperparameter sheet; auto-detect must not misparse.
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "bert", "architectures": ["BertModel"], "hidden_size": 768})
    )
    with pytest.raises(ValueError, match="No builder is registered for 'BertModel'"):
        from_pretrained(tmp_path)


def test_from_pretrained_rejects_unrecognized_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"foo": 1}))
    with pytest.raises(ValueError, match="Unrecognized config"):
        from_pretrained(tmp_path)


def test_load_network_rejects_unstamped_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    with pytest.raises(ValueError, match="HuggingFace config"):
        load_network(tmp_path / "config.json")


def test_dropout_network_roundtrips_with_fresh_rng(tmp_path):
    X = pt.matrix("X")
    output = Sequential(Linear("fc", n_in=4, n_out=4), Dropout(p=0.5, random_state=0))(X)
    for parameter in collect_trainable_params(output):
        value = np.random.default_rng(0).normal(size=parameter.get_value().shape)
        parameter.set_value(value.astype(parameter.type.dtype))
    fc_weight = (
        next(v for v in collect_shared_variables(output) if v.name == "fc_W").get_value().copy()
    )

    save_pretrained(output, tmp_path)
    restored_inputs, restored_output = from_pretrained(tmp_path)  # fresh RNG by default

    # The weights round-trip through safetensors even with an RNG in the graph; the RNG itself is fresh.
    restored_weight = next(v for v in collect_shared_variables(restored_output) if v.name == "fc_W")
    np.testing.assert_array_equal(restored_weight.get_value(), fc_weight)
    predict(restored_inputs, restored_output, np.zeros((3, 4)))  # the rebuilt graph runs


def test_restore_rng_reproduces_dropout_draws(tmp_path):
    X = pt.matrix("X")
    output = Dropout(p=0.5, random_state=0)(X)
    save_pretrained(output, tmp_path)
    x_value = np.random.default_rng(1).normal(size=(6, 4))

    original = predict([X], output, x_value)
    fresh_inputs, fresh_output = from_pretrained(tmp_path)  # default: fresh RNG
    restored_inputs, restored_output = from_pretrained(tmp_path, restore_rng=True)

    np.testing.assert_array_equal(predict(restored_inputs, restored_output, x_value), original)
    assert not np.array_equal(predict(fresh_inputs, fresh_output, x_value), original)


def test_batchnorm_non_trainable_state_survives_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = pt.matrix("X")
    output = Sequential(Linear("fc", n_in=4, n_out=4), BatchNorm("bn", n_in=4))(X)
    running_mean = next(v for v in collect_shared_variables(output) if v.name == "bn_running_mean")
    running_mean.set_value(rng.normal(size=4).astype(running_mean.type.dtype))

    save_pretrained(output, tmp_path)
    _, restored_output = from_pretrained(tmp_path)

    restored_mean = next(
        v for v in collect_shared_variables(restored_output) if v.name == "bn_running_mean"
    )
    assert isinstance(restored_mean, NonTrainableParameter)
    np.testing.assert_array_equal(restored_mean.get_value(), running_mean.get_value())


def test_load_network_rejects_an_older_format_version(tmp_path):
    # Op classes are recorded by import path, so a config from another layout must fail here rather than
    # later inside class resolution.
    X = pt.tensor("X", shape=(None, 4))
    path = tmp_path / "config.json"
    save_network(Linear("fc", n_in=4, n_out=2)(X), path, inputs=[X])

    config = json.loads(path.read_text())
    config["format_version"] = 1
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="graph format version 1"):
        load_network(path)


def test_a_loaded_batch_norm_returns_to_its_identity_transform(tmp_path):
    """The regression the whole exercise is for. A batch norm scale is ones because the layer declares it,
    and a config that dropped the declaration gave the scale a fan-scaled draw -- and once `fans` started
    refusing 1-D shapes, an outright error."""
    X = pt.matrix("X")
    output = Sequential(Linear("fc", n_in=4, n_out=4), BatchNorm("norm", n_in=4))(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    parameters = collect_trainable_params(restored)
    for parameter, value in zip(parameters, initialize_params(parameters, rng=0)):
        parameter.set_value(value)

    by_name = {p.name: p for p in parameters}
    np.testing.assert_array_equal(by_name["norm_scale"].get_value(), 1)
    np.testing.assert_array_equal(by_name["norm_loc"].get_value(), 0)


def test_a_parameterized_initializer_keeps_its_arguments(tmp_path):
    """Recording the registry name alone would be lossy: 'normal' rebuilds at the default spread, so a table
    built for GPT-2 at 0.02 would come back at 0.01 and nothing would say so."""
    X = pt.imatrix("ids")
    output = Embedding(
        "tok", n_embeddings=32, n_features=8, weight_initializer=NormalInitializer(0.0, 0.02)
    )(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    [table] = collect_trainable_params(restored)

    assert isinstance(table.initializer, NormalInitializer)
    assert (table.initializer.mean, table.initializer.std) == (0.0, 0.02)


def test_a_decorated_initializer_round_trips_with_its_parameters(tmp_path):
    """The reason `@initializer` exists: the parameters a closure would have captured are declared instead,
    so they can be written down. `constant` lives in conftest, at module level, which is what lets the
    config find the class again."""
    X = pt.matrix("X")
    output = Linear("fc", n_in=4, n_out=4, weight_initializer=constant(value=7.0))(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    weight = next(p for p in collect_trainable_params(restored) if p.name == "fc_W")

    assert isinstance(weight.initializer, constant)
    assert weight.initializer.value == 7.0
    np.testing.assert_array_equal(weight.get_value(), 7.0)  # and the rebuilt parameter drew from it


def test_an_initializer_defined_locally_reports_what_was_lost(tmp_path):
    """An import cannot reach a class defined inside a function, so the config records only its name. Saving
    and loading still work, since restoring saved values needs no law; the redraw is what needs it back."""

    @initializer
    def local_constant(rng, shape, value):
        return np.full(shape, value)

    X = pt.matrix("X")
    output = Linear("fc", n_in=4, n_out=4, weight_initializer=local_constant(value=3.0))(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    weight = next(p for p in collect_trainable_params(restored) if p.name == "fc_W")

    assert isinstance(weight.initializer, UnrecordedInitializer)
    with pytest.raises(ValueError, match="local_constant, which the saved config could not record"):
        initialize_params([weight], rng=0)


def test_a_loaded_network_initializes_exactly_like_the_one_it_was_saved_from(tmp_path):
    """The whole point, stated as one equality. Same seed, same values, parameter for parameter -- which also
    pins that the rebuilt parameters come back in the saved order, since one generator draws them in
    sequence and a permutation would hand each the wrong draw."""
    X = pt.matrix("X")
    original = Sequential(
        Linear("fc1", n_in=4, n_out=8),
        ReLU(),
        BatchNorm("norm", n_in=8),
        Linear("fc2", n_in=8, n_out=2, weight_initializer=NormalInitializer(0.0, 0.02)),
    )(X)
    save_network(original, tmp_path / "config.json")
    _, restored = load_network(tmp_path / "config.json")

    def seeded(output):
        parameters = collect_trainable_params(output)
        values = initialize_params(parameters, rng=1234)
        return {p.name: value for p, value in zip(parameters, values)}

    from_original, from_loaded = seeded(original), seeded(restored)

    assert list(from_original) == list(from_loaded)  # same order, not merely the same names
    for name, value in from_original.items():
        np.testing.assert_array_equal(value, from_loaded[name], err_msg=name)


@initializer
def arange_fill(rng, shape, start):
    """A second decorated initializer, distinguishable from `constant` at a glance."""
    return np.arange(start, start + int(np.prod(shape))).reshape(shape)


def test_several_decorated_initializers_keep_their_own_class_and_parameters(tmp_path):
    """Props live on the instance and the class is shared, so two instances of one decorated initializer must
    not collide, and two different ones must not be confused for each other."""
    X = pt.matrix("X")
    output = Sequential(
        Linear("fc1", n_in=4, n_out=4, weight_initializer=constant(value=7.0)),
        Linear("fc2", n_in=4, n_out=4, weight_initializer=constant(value=-2.0)),
        Linear("fc3", n_in=4, n_out=4, weight_initializer=arange_fill(start=100.0)),
    )(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    weights = {p.name: p for p in collect_trainable_params(restored) if p.name.endswith("_W")}

    assert isinstance(weights["fc1_W"].initializer, constant)
    assert isinstance(weights["fc2_W"].initializer, constant)
    assert isinstance(weights["fc3_W"].initializer, arange_fill)

    assert weights["fc1_W"].initializer.value == 7.0
    assert weights["fc2_W"].initializer.value == -2.0
    assert weights["fc3_W"].initializer.start == 100.0

    np.testing.assert_array_equal(weights["fc1_W"].get_value(), 7.0)
    np.testing.assert_array_equal(weights["fc2_W"].get_value(), -2.0)
    assert weights["fc3_W"].get_value().min() == 100.0


def test_an_initializer_with_no_parameters_round_trips(tmp_path):
    """Serializability comes from the recorded parameters, and this has none -- the fans are computed inside
    the sampler from the shape it is handed, which the config never sees. So a scaled initializer written by
    hand survives a round trip on the strength of its import path alone."""
    X = pt.matrix("X")
    output = Linear("fc", n_in=16, n_out=4, weight_initializer=he_normal())(X)
    save_network(output, tmp_path / "config.json")

    _, restored = load_network(tmp_path / "config.json")
    weight = next(p for p in collect_trainable_params(restored) if p.name == "fc_W")

    assert isinstance(weight.initializer, he_normal)
    assert weight.initializer.__props__ == ()
    # Redrawn from the restored class, and scaled by the fan-in the layer's shape implies.
    [value] = initialize_params([weight], rng=0)
    assert value.std() == pytest.approx(np.sqrt(2.0 / 16), rel=0.25)


def generator_of(outputs):
    variables = collect_shared_variables(outputs)
    return next(
        v.get_value(borrow=True) for v in variables if isinstance(v.type, RandomGeneratorType)
    )


@pytest.mark.parametrize("bit_generator", ["PCG64", "MT19937", "Philox", "SFC64"])
def test_a_network_saves_and_restores_any_bit_generator(tmp_path, bit_generator):
    """A generator's state is config JSON here, and MT19937 keeps its key as an array while Philox keeps
    its counter, so the state has to survive JSON and rebuild the kind it came from."""
    X = pt.matrix("X")
    source = np.random.Generator(getattr(np.random, bit_generator)(0))
    output = Sequential(Linear("fc", n_in=4, n_out=4), Dropout(p=0.5, random_state=source))(X)

    path = tmp_path / "config.json"
    save_network(output, path)
    expected = generator_of(output).random(3)

    _, restored = load_network(path, restore_rng=True)
    assert type(generator_of(restored).bit_generator).__name__ == bit_generator
    np.testing.assert_array_equal(generator_of(restored).random(3), expected)


def test_a_config_written_before_arrays_were_tagged_still_loads(tmp_path):
    """Configs already on disk hold the raw state dict. Only PCG64 could ever have been written -- the
    others raised on the way out -- and its state is plain scalars, so the stored form is unchanged and
    an old config is still a readable one."""
    X = pt.matrix("X")
    output = Sequential(Linear("fc", n_in=4, n_out=4), Dropout("drop", p=0.5, random_state=0))(X)
    path = tmp_path / "config.json"
    save_network(output, path)

    stored = next(
        meta["rng_state"]
        for meta in json.loads(path.read_text())["input_meta"]
        if "rng_state" in meta
    )
    assert stored == generator_of(output).bit_generator.state

    expected = generator_of(output).random(3)
    _, restored = load_network(path, restore_rng=True)
    np.testing.assert_array_equal(generator_of(restored).random(3), expected)


@pytest.mark.parametrize("bit_generator", ["PCG64", "MT19937"])
def test_a_stochastic_network_round_trips_whole(tmp_path, bit_generator):
    """Architecture, weights, running statistics and generator together: a restored network has to
    reproduce the output of the one that was saved, not merely load without raising."""
    X = pt.matrix("X")
    output = Sequential(
        Linear("fc1", n_in=4, n_out=8),
        BatchNorm("bn", n_in=8),
        ReLU(),
        Dropout(
            "drop", p=0.5, random_state=np.random.Generator(getattr(np.random, bit_generator)(0))
        ),
        Linear("fc2", n_in=8, n_out=2),
    )(X)
    parameters = collect_trainable_params(output)
    for parameter, value in zip(
        parameters, initialize_params(parameters, rng=np.random.default_rng(0))
    ):
        parameter.set_value(value)

    save_pretrained(output, tmp_path)
    X_value = np.ones((3, 4), dtype=floatX)
    expected = function([X], output)(X_value)

    inputs, restored = from_pretrained(tmp_path, restore_rng=True)
    np.testing.assert_allclose(function(inputs, restored)(X_value), expected)


@pytest.mark.parametrize("bit_generator", ["Philox", "MT19937"])
def test_a_fresh_generator_keeps_the_kind_the_network_was_saved_with(tmp_path, bit_generator):
    """``restore_rng=False`` asks for a fresh stream, not a different architecture. Rebuilding the
    default kind instead would also strand a later `load_state` on the kind it finds."""
    X = pt.matrix("X")
    source = np.random.Generator(getattr(np.random, bit_generator)(0))
    output = Sequential(Linear("fc", n_in=4, n_out=4), Dropout("drop", p=0.5, random_state=source))(
        X
    )
    path = tmp_path / "config.json"
    save_network(output, path)

    _, restored = load_network(path, restore_rng=False)
    generator = generator_of(restored)
    assert type(generator.bit_generator).__name__ == bit_generator
    assert jsonable_rng_state(generator.bit_generator.state) != jsonable_rng_state(
        generator_of(output).bit_generator.state
    )


def test_a_generator_that_is_not_shared_stays_a_data_input(tmp_path):
    """A free generator input is part of the call signature, not state the network owns, so rebuilding
    it as shared would quietly change how the reloaded network is called."""
    Z = pt.matrix("Z")
    free = random_generator_type(name="free_rng")
    _, draw = pt.random.normal(rng=free, size=(), return_next_rng=True)
    output = Z.sum() + draw
    path = tmp_path / "config.json"
    save_network(output, path, inputs=[Z, free])

    inputs, _ = load_network(path, restore_rng=False)
    assert len(inputs) == 2
    assert not any(isinstance(variable, SharedVariable) for variable in inputs)


def test_a_fresh_generator_does_not_need_the_state_it_discards(tmp_path):
    """``restore_rng=False`` reads only the recorded kind, so a config whose generator state is
    unusable still rebuilds a network that never wanted that state."""
    X = pt.matrix("X")
    source = np.random.Generator(np.random.MT19937(0))
    output = Sequential(Linear("fc", n_in=4, n_out=4), Dropout("drop", p=0.5, random_state=source))(
        X
    )
    path = tmp_path / "config.json"
    save_network(output, path)

    config = json.loads(path.read_text())
    for meta in config["input_meta"]:
        if "rng_state" in meta:
            meta["rng_state"]["state"]["key"]["__array__"] = [0] * 10
    path.write_text(json.dumps(config))

    _, restored = load_network(path, restore_rng=False)
    assert type(generator_of(restored).bit_generator).__name__ == "MT19937"


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


@pytest.mark.parametrize(
    "config",
    [
        {"_class_name": "AutoencoderKL"},
        {"model_type": "clip_text_model", "architectures": ["CLIPTextModel"]},
    ],
    ids=["diffusers", "transformers"],
)
def test_a_foreign_config_is_detected_as_huggingface(config):
    """Diffusers configs carry only _class_name, so a detector keyed on transformers' spellings
    rejects the components this loads."""
    assert _detect_format(config) == "huggingface"


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


TINY_CLIP = {
    "architectures": ["CLIPTextModel"],
    "hidden_size": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 16,
    "vocab_size": 50,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "projection_dim": 4,
}


def test_clip_builder_binds_the_checkpoint_key_of_every_parameter():
    """The bound keys are HuggingFace's own module paths, so this is what decides whether a real
    checkpoint loads. Two layers is enough to pin the numbering and the nesting."""
    _, _, keys = build_from_config(TINY_CLIP)

    per_layer = [
        "layer_norm1.weight",
        "layer_norm1.bias",
        "layer_norm2.weight",
        "layer_norm2.bias",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
    ]
    assert keys.keys() == {
        "text_model.embeddings.token_embedding.weight",
        "text_model.embeddings.position_embedding.weight",
        "text_model.final_layer_norm.weight",
        "text_model.final_layer_norm.bias",
        *(f"text_model.encoder.layers.{i}.{name}" for i in range(2) for name in per_layer),
    }


def test_clip_builder_returns_the_final_state_then_every_layer():
    """SDXL conditions on the second-to-last layer, so the per-layer states are outputs rather than
    internals a caller would have to rebuild the model to reach."""
    inputs, outputs, _ = build_from_config(TINY_CLIP)

    assert [variable.name for variable in inputs] == ["input_ids"]
    assert len(outputs) == TINY_CLIP["num_hidden_layers"] + 1
    assert outputs[0].name == "last_hidden_state"
    assert all(output.type.shape == (None, None, 8) for output in outputs)


@pytest.mark.parametrize(
    "hidden_act, expected",
    [("quick_gelu", "QuickGELU"), ("gelu", "GELU")],
    ids=["clip_l", "clip_big_g"],
)
def test_clip_builder_uses_the_configured_activation(hidden_act, expected):
    """SDXL's two encoders differ here -- CLIP-L is quick_gelu and bigG is gelu -- so hardcoding
    either one gets half the conditioning quietly wrong."""
    _, outputs, _ = build_from_config({**TINY_CLIP, "hidden_act": hidden_act})

    names = {variable.name for variable in ancestors(outputs)}
    assert expected in names


def test_clip_builder_rejects_an_intermediate_size_that_is_not_a_multiple():
    with pytest.raises(ValueError, match="not a whole multiple of hidden_size"):
        build_from_config({**TINY_CLIP, "intermediate_size": 30})


def test_clip_builder_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="hidden_act is 'silu'"):
        build_from_config({**TINY_CLIP, "hidden_act": "silu"})


def test_clip_projection_builder_adds_only_the_projection_head():
    """The projection sits at the checkpoint's top level, not under text_model, and is the sole
    difference from the base architecture."""
    _, _, base = build_from_config(TINY_CLIP)
    _, _, projected = build_from_config(
        {**TINY_CLIP, "architectures": ["CLIPTextModelWithProjection"]}
    )

    assert projected.keys() - base.keys() == {"text_projection.weight"}
    assert base.keys() - projected.keys() == set()


def test_clip_projection_builder_returns_the_pooled_embedding_first():
    inputs, outputs, _ = build_from_config(
        {**TINY_CLIP, "architectures": ["CLIPTextModelWithProjection"]}
    )

    assert outputs[0].name == "text_embeds"
    assert outputs[0].type.shape == (None, TINY_CLIP["projection_dim"])
    assert outputs[1].name == "last_hidden_state"
    assert len(outputs) == TINY_CLIP["num_hidden_layers"] + 2


@pytest.mark.parametrize(
    "eos_token_id, ids, expected_position",
    [(2, [5, 40, 9, 40], 1), (9, [5, 40, 9, 40], 2)],
    ids=["legacy_takes_the_largest_id", "matches_the_configured_id"],
)
def test_clip_pools_at_the_end_of_stream_token(eos_token_id, ids, expected_position):
    """Configs written before transformers#24773 carry eos_token_id 2 whatever the tokenizer uses, so
    they locate the token by taking the largest id instead. SDXL's are of that vintage."""
    config = {
        **TINY_CLIP,
        "architectures": ["CLIPTextModelWithProjection"],
        "eos_token_id": eos_token_id,
    }
    inputs, outputs, keys = build_from_config(config)
    rng = np.random.default_rng(0)
    for parameter in collect_trainable_params(outputs[0]):
        parameter.set_value(rng.normal(size=parameter.get_value().shape))

    pooled_output, final = pytensor.function(inputs, [outputs[0], outputs[1]])(
        np.array([ids], dtype="int32")
    )
    projection = keys.parameter_for("text_projection.weight").get_value()

    np.testing.assert_allclose(
        pooled_output[0], final[0, expected_position] @ projection, rtol=1e-5, atol=1e-5
    )


def _write_huggingface_component(directory, config, tensors, filename):
    """A HuggingFace component directory: a config and one safetensors file."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    save_file({key: np.asarray(value) for key, value in tensors.items()}, directory / filename)
    return directory


# The key names and checkpoint shapes HuggingFace writes for a two-layer CLIP of TINY_CLIP's size,
# spelled out rather than read back from the builder. A fixture derived from the builder cannot fail
# on a wrong key, a wrong scope or a missing transpose, which is most of what the loader can get
# wrong.
TINY_CLIP_CHECKPOINT = {
    "text_model.embeddings.token_embedding.weight": (50, 8),
    "text_model.embeddings.position_embedding.weight": (16, 8),
    "text_model.final_layer_norm.weight": (8,),
    "text_model.final_layer_norm.bias": (8,),
    **{
        f"text_model.encoder.layers.{layer}.{name}": shape
        for layer in range(2)
        for name, shape in {
            "layer_norm1.weight": (8,),
            "layer_norm1.bias": (8,),
            "layer_norm2.weight": (8,),
            "layer_norm2.bias": (8,),
            "self_attn.q_proj.weight": (8, 8),
            "self_attn.q_proj.bias": (8,),
            "self_attn.k_proj.weight": (8, 8),
            "self_attn.k_proj.bias": (8,),
            "self_attn.v_proj.weight": (8, 8),
            "self_attn.v_proj.bias": (8,),
            "self_attn.out_proj.weight": (8, 8),
            "self_attn.out_proj.bias": (8,),
            "mlp.fc1.weight": (32, 8),
            "mlp.fc1.bias": (32,),
            "mlp.fc2.weight": (8, 32),
            "mlp.fc2.bias": (8,),
        }.items()
    },
}


def _tiny_clip_tensors():
    rng = np.random.default_rng(0)
    return {
        key: rng.normal(size=shape).astype("float16") for key, shape in TINY_CLIP_CHECKPOINT.items()
    }


def test_the_clip_builder_binds_exactly_what_a_checkpoint_holds():
    """The fixture below is only worth trusting if it is the checkpoint HuggingFace would write, so
    the builder's bindings are checked against it rather than the other way round."""
    _, _, keys = build_from_config(TINY_CLIP)

    assert keys.keys() == set(TINY_CLIP_CHECKPOINT)
    for key, checkpoint_shape in TINY_CLIP_CHECKPOINT.items():
        # nn.Linear stores a dense weight channel-first, so every 2-D tensor but the two embedding
        # tables is the parameter's transpose.
        stored_as_built = key.endswith(("token_embedding.weight", "position_embedding.weight"))
        expected = (
            checkpoint_shape[::-1]
            if len(checkpoint_shape) == 2 and not stored_as_built
            else checkpoint_shape
        )
        assert keys.parameter_for(key).get_value().shape == expected, key


def test_from_pretrained_builds_and_loads_a_huggingface_directory(tmp_path):
    config = {**TINY_CLIP}
    component = _write_huggingface_component(
        tmp_path / "text_encoder", config, _tiny_clip_tensors(), "model.safetensors"
    )

    inputs, outputs = from_pretrained(component)

    assert [variable.name for variable in inputs] == ["input_ids"]
    assert outputs[0].name == "last_hidden_state"
    result = pytensor.function(inputs, outputs[0])(np.zeros((2, 5), dtype="int32"))
    assert result.shape == (2, 5, config["hidden_size"])
    assert np.isfinite(result).all()


def test_from_pretrained_needs_a_variant_when_several_weight_files_exist(tmp_path):
    config = {**TINY_CLIP}
    tensors = _tiny_clip_tensors()
    component = _write_huggingface_component(
        tmp_path / "text_encoder", config, tensors, "model.safetensors"
    )
    save_file(
        {key: np.asarray(value) for key, value in tensors.items()},
        component / "model.fp16.safetensors",
    )

    with pytest.raises(ValueError, match="several weight files"):
        from_pretrained(component)

    inputs, outputs = from_pretrained(component, variant="fp16")
    assert outputs[0].name == "last_hidden_state"


def test_from_pretrained_reports_a_directory_with_no_safetensors(tmp_path):
    component = tmp_path / "text_encoder"
    component.mkdir()
    (component / "config.json").write_text(json.dumps(TINY_CLIP))

    with pytest.raises(FileNotFoundError, match=r"No \.safetensors weights"):
        from_pretrained(component)


TINY_GPT2 = {
    "architectures": ["GPT2LMHeadModel"],
    "n_embd": 8,
    "n_head": 2,
    "n_layer": 2,
    "n_positions": 16,
    "vocab_size": 20,
    "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-5,
    "n_inner": None,
}


def test_gpt2_builder_binds_one_key_for_the_fused_attention():
    """GPT-2 stores q, k and v in a single c_attn tensor, so the graph must own one weight there
    rather than three. Its Conv1D already stores (in, out), so nothing transposes."""
    _, _, keys = build_from_config(TINY_GPT2)

    per_layer = [
        "ln_1.weight",
        "ln_1.bias",
        "ln_2.weight",
        "ln_2.bias",
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    ]
    assert keys.keys() == {
        "wte.weight",
        "wpe.weight",
        "ln_f.weight",
        "ln_f.bias",
        *(f"h.{i}.{name}" for i in range(2) for name in per_layer),
    }
    assert keys.parameter_for("h.0.attn.c_attn.weight").get_value().shape == (8, 24)


def test_gpt2_builder_ties_the_head_to_the_token_embedding():
    """The checkpoint carries no lm_head weight, so the head has to reuse wte rather than bind a
    parameter the file cannot fill."""
    inputs, outputs, keys = build_from_config(TINY_GPT2)
    assert not any("lm_head" in key for key in keys.keys())

    rng = np.random.default_rng(0)
    for parameter in collect_trainable_params(outputs[0]):
        parameter.set_value(rng.normal(size=parameter.get_value().shape))

    logits, final = pytensor.function(inputs, [outputs[0], outputs[1]])(
        np.array([[1, 2, 3]], dtype="int32")
    )
    token_embedding = keys.parameter_for("wte.weight").get_value()

    np.testing.assert_allclose(logits, final @ token_embedding.T, rtol=1e-5, atol=1e-5)


def test_gpt2_builder_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="activation_function is 'swiglu'"):
        build_from_config({**TINY_GPT2, "activation_function": "swiglu"})


@pytest.mark.parametrize("config", [TINY_CLIP, TINY_GPT2], ids=["clip", "gpt2"])
def test_a_token_id_input_is_an_integer_matrix(config):
    """Token ids index the embedding table, and pt.matrix defaults to floatX, so the dtype has to be
    asked for rather than inherited."""
    inputs, _, _ = build_from_config(config)

    assert inputs[0].type.dtype == "int64"


@pytest.mark.parametrize("config", [TINY_CLIP, TINY_GPT2], ids=["clip", "gpt2"])
def test_a_sequence_longer_than_the_position_table_raises(config):
    """Past the table the position gather reads out of bounds and returns whatever memory it finds,
    which is a wrong model that runs."""
    inputs, outputs, _ = build_from_config(config)
    predict = pytensor.function(inputs, outputs[0])

    with pytest.raises(AssertionError, match="longer than the 16 positions"):
        predict(np.zeros((1, 17), dtype="int64"))

    assert predict(np.zeros((1, 16), dtype="int64")).shape[1] == 16


@pytest.mark.parametrize(
    "flag, value",
    [
        ("scale_attn_weights", False),
        ("scale_attn_by_inverse_layer_idx", True),
        ("reorder_and_upcast_attn", True),
    ],
)
def test_gpt2_builder_rejects_a_config_that_changes_the_attention_arithmetic(flag, value):
    """These flags load cleanly and return wrong numbers, which is the one failure the key map cannot
    catch for itself."""
    with pytest.raises(ValueError, match=flag):
        build_from_config({**TINY_GPT2, flag: value})


def test_a_sharded_checkpoint_is_reported_rather_than_partly_loaded(tmp_path):
    """One shard of a sharded checkpoint would fill some parameters and leave the rest at their
    initialization."""
    component = _write_huggingface_component(
        tmp_path / "text_encoder",
        TINY_CLIP,
        _tiny_clip_tensors(),
        "model-00001-of-00002.safetensors",
    )
    (component / "model.safetensors.index.json").write_text("{}")

    with pytest.raises(NotImplementedError, match="sharded checkpoint"):
        from_pretrained(component)


def test_from_pretrained_rejects_restore_rng_for_a_huggingface_directory(tmp_path):
    component = _write_huggingface_component(
        tmp_path / "text_encoder", TINY_CLIP, _tiny_clip_tensors(), "model.safetensors"
    )

    with pytest.raises(ValueError, match="restore_rng"):
        from_pretrained(component, restore_rng=True)


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


def test_bind_linear_binds_no_bias_for_a_bias_free_layer():
    """CLIP's text projection has no bias, and binding one would ask the checkpoint for a tensor it
    does not hold."""
    keys = KeyMap()
    bind_linear(keys, Linear("text_projection", n_in=4, n_out=2, bias=False), "text_projection")

    assert keys.keys() == {"text_projection.weight"}


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
