from typing import get_args

import numpy as np
import pytensor
import pytest

from pytensor_ml.layers import Linear
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import collect_trainable_params
from pytensor_ml.state import (
    _INITIALIZERS,
    EmptyInitializer,
    InitializationScheme,
    Initializer,
    NormalInitializer,
    OneInitializer,
    OrthogonalInitializer,
    UnitUniformInitializer,
    XavierNormalInitializer,
    XavierUniformInitializer,
    ZeroInitializer,
    fans,
    initial_values_from,
    initialize_params,
    initializer,
)
from tests.conftest import constant


def test_every_registered_name_maps_to_a_class_that_needs_no_arguments():
    """The registry is how an initializer crosses into a serialized config, as a name. A class needing a
    constructor argument could be written out and not read back, so the Literal and the registry have to
    agree and every entry has to rebuild from its name alone."""
    assert set(get_args(InitializationScheme)) == set(_INITIALIZERS)

    for name, initializer_class in _INITIALIZERS.items():
        assert isinstance(initializer_class(), Initializer), name


class TestInitializeParams:
    @pytest.mark.parametrize("initializer_class", sorted(_INITIALIZERS.values(), key=repr))
    def test_values_match_parameter_shapes_and_dtypes(self, initializer_class, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)
        # Only the weight matrices, since a fan-scaled initializer cannot draw a 1-D bias. The biases come
        # from their own declarations, and the assertion covers every parameter either way.
        drawn_by = {p: initializer_class() for p in params if p.get_value().ndim > 1}

        values = initialize_params(params, rng=np.random.default_rng(0), initializers=drawn_by)

        assert len(values) == len(params)
        for param, val in zip(params, values):
            assert val.shape == param.get_value().shape
            assert str(val.dtype) == str(param.get_value().dtype)

    def test_xavier_normal_scales_by_fan_in_plus_fan_out(self, simple_network):
        # Drawn from fc1_W's own declaration, which is what a Linear layer records for its weight.
        X, y = simple_network
        weight = next(p for p in collect_trainable_params(y) if p.name == "fc1_W")
        fan_sum = sum(weight.get_value().shape)

        [value] = initialize_params([weight], rng=np.random.default_rng(42))

        assert value.std() == pytest.approx(np.sqrt(2.0 / fan_sum), rel=0.05)
        # Xavier uniform targets this same variance; only a normal draw crosses its hard bound.
        assert np.abs(value).max() > np.sqrt(6.0 / fan_sum)

    def test_reproducible_with_seed(self, simple_network):
        X, y = simple_network
        params = collect_trainable_params(y)

        values1 = initialize_params(params, rng=np.random.default_rng(123))
        values2 = initialize_params(params, rng=np.random.default_rng(123))

        for v1, v2 in zip(values1, values2):
            np.testing.assert_array_equal(v1, v2)

    def test_parameters_do_not_all_receive_the_same_draws(self):
        # The seed must be an int: passing a Generator masks the regression this guards.
        drawn = UnitUniformInitializer()
        first = trainable(np.zeros((4, 4), dtype="float64"), "first", initializer=drawn)
        second = trainable(np.zeros((4, 4), dtype="float64"), "second", initializer=drawn)

        values = initialize_params([first, second], rng=0)

        assert not np.array_equal(values[0], values[1])

    def test_accepts_an_initializer_built_from_a_function(self):
        params = [
            trainable(np.zeros((4, 4), dtype="float64"), "first"),
            trainable(np.zeros((4, 2), dtype="float64"), "second"),
        ]

        values = initialize_params(params, initializers=dict.fromkeys(params, constant(value=7.0)))

        assert len(values) == len(params)
        for val in values:
            np.testing.assert_array_equal(val, 7.0)


class TestDeclaredInitializers:
    def test_a_declaration_is_what_a_redraw_draws_from(self):
        # Starting from zeros, so the assertion only holds if the declared initializer actually ran.
        scale = trainable(np.zeros(4, dtype="float64"), "scale", initializer=OneInitializer())

        [value] = initialize_params([scale], rng=0)

        np.testing.assert_array_equal(value, 1)

    def test_a_parameter_declaring_nothing_refuses_to_be_redrawn(self):
        """There is no law to draw it from, and the alternative is worse than an error: a hand-built weight
        left at the value it was given is a stack of zeros that trains only its last bias."""
        weight = trainable(np.zeros((4, 4), dtype="float64"), "weight")

        with pytest.raises(ValueError, match="'weight' declares no initializer"):
            initialize_params([weight], rng=0)

    def test_naming_a_parameter_that_declares_nothing_is_enough(self):
        weight = trainable(np.zeros((4, 4), dtype="float64"), "weight")

        [value] = initialize_params([weight], rng=0, initializers={weight: constant(value=7.0)})

        np.testing.assert_array_equal(value, 7.0)

    def test_a_shared_variable_without_the_marker_class_has_to_be_named(self):
        """Plain shared state carries no ``initializer`` attribute at all, so the declaration lookup has to
        tolerate its absence rather than raise an AttributeError on the way past."""
        state = pytensor.shared(np.zeros(4, dtype="float64"), name="state")

        with pytest.raises(ValueError, match="'state' declares no initializer"):
            initialize_params([state], rng=0)

        [value] = initialize_params([state], rng=0, initializers={state: constant(value=7.0)})
        np.testing.assert_array_equal(value, 7.0)

    def test_calling_an_initializer_overrides_a_declaration(self):
        scale = trainable(np.ones(3, dtype="float64"), "scale", initializer=OneInitializer())

        ZeroInitializer()(scale)

        np.testing.assert_array_equal(scale.get_value(), 0)


def test_calling_an_initializer_assigns_the_parameter_in_place():
    param = trainable(np.ones(3, dtype="float64"), "w")

    returned = ZeroInitializer()(param)

    np.testing.assert_array_equal(param.get_value(), 0)
    assert returned is param


def test_a_convolution_kernel_folds_its_receptive_field_into_both_fans():
    """Summing the shape is the fan computation only for a matrix. For a ``(kH, kW, in, out)`` kernel every
    input channel reaches an output at each of the kH*kW offsets, so leaving the receptive field out of the
    fans overstates the spread: 0.258 where the correct scale for this shape is 0.096.

    The features are the two trailing dimensions, which is what makes one rule cover a weight matrix and a
    kernel alike, and it is the layout flax and keras store a kernel in."""
    kernel_shape = (3, 3, 8, 16)  # asymmetric, so the orientation of the two fans is pinned as well
    fan_in, fan_out = fans(kernel_shape)
    assert (fan_in, fan_out) == (8 * 9, 16 * 9)

    value = XavierNormalInitializer().sample(kernel_shape, "float64", np.random.default_rng(0))

    assert value.std() == pytest.approx(np.sqrt(2.0 / (fan_in + fan_out)), rel=0.05)


@pytest.mark.parametrize(
    "shape", [(768, 768), (50257, 768), (4, 7)], ids=["square", "embedding", "small"]
)
def test_a_weight_matrix_draws_exactly_as_it_did_before(shape):
    """No parameter that already existed may move: a matrix has no dimensions past the second, so the sum of
    its fans is the sum of its shape, which is what the draw was scaled by before."""
    fan_in, fan_out = fans(shape)
    assert fan_in + fan_out == sum(shape)

    with_fans = XavierNormalInitializer().sample(shape, "float64", np.random.default_rng(7))
    with_shape_sum = np.random.default_rng(7).normal(0, np.sqrt(2.0 / sum(shape)), size=shape)

    np.testing.assert_array_equal(with_fans, with_shape_sum)


def test_the_fan_in_is_the_dimension_the_layers_treat_as_input():
    """Weights here are ``(n_in, n_out)``, the transpose of torch's layout, so the leading dimension is the
    fan-in. Xavier only reads the sum and cannot tell the difference; anything scaling by fan-in alone can."""
    layer = Linear("fc", n_in=4, n_out=7)

    fan_in, fan_out = fans(layer.W.get_value().shape)

    assert (fan_in, fan_out) == (4, 7)


@pytest.mark.parametrize(
    "initializer",
    [XavierNormalInitializer(), XavierUniformInitializer()],
    ids=["normal", "uniform"],
)
def test_a_fan_scaled_initializer_rejects_a_parameter_with_no_fans(initializer):
    """A bias or a norm scale has no fan-in and fan-out, and scaling by the length of a vector is a number
    with no meaning behind it. Pointing one of these at such a parameter -- through a layer keyword or an
    `initializers` entry -- should say so rather than draw something arbitrary."""
    with pytest.raises(ValueError, match="at least two dimensions"):
        initializer.sample((768,), "float64", np.random.default_rng(0))


def test_a_normal_initializer_draws_at_the_standard_deviation_it_was_given():
    """The fan-scaled initializers derive their spread from the shape; this one is told it, so the same
    standard deviation has to come out whatever the shape's fans work out to. GPT-2 applies 0.02 to a
    50257x768 embedding and a 768x768 weight alike, where Xavier gives 0.006 and 0.036. Both orientations
    are checked at a size large enough to estimate a standard deviation from; four samples cannot."""
    initializer = NormalInitializer(0.0, 0.02)

    for shape in [(1000, 64), (64, 1000)]:
        value = initializer.sample(shape, "float64", np.random.default_rng(0))
        assert value.std() == pytest.approx(0.02, rel=0.05)
        assert value.mean() == pytest.approx(0.0, abs=0.001)


def test_a_normal_initializer_has_no_fans_to_satisfy():
    """Unlike Xavier it accepts a 1-D parameter, which is the point: a bias drawn from a normal is torch's
    convention and needs no fan computation. Asserting the spread rather than only the shape, so this says
    the draw was right and not merely that nothing raised."""
    value = NormalInitializer(0.0, 1.0).sample((4096,), "float64", np.random.default_rng(0))

    assert value.shape == (4096,)
    assert value.std() == pytest.approx(1.0, rel=0.05)


def test_a_normal_initializer_built_from_its_registry_name_has_a_usable_default_spread():
    """Both arguments default, which is what lets it into the registry at all. The default has to be a
    spread something could train from, since a config naming 'normal' rebuilds exactly this."""
    value = _INITIALIZERS["normal"]().sample((100, 100), "float64", np.random.default_rng(0))

    assert value.std() == pytest.approx(0.01, rel=0.05)


@pytest.mark.parametrize("shape", [(5, 5), (7, 3), (3, 7)], ids=["square", "tall", "wide"])
def test_an_orthogonal_draw_preserves_the_norm_it_can(shape):
    """Whichever of the two products is the identity, the map leaves a vector's norm alone, which is the
    whole reason a recurrence draws from here. Only the smaller side can be orthonormal on a rectangle."""
    value = OrthogonalInitializer().sample(shape, "float64", np.random.default_rng(0))
    n_rows, n_cols = shape
    product = value @ value.T if n_rows < n_cols else value.T @ value

    assert value.shape == shape
    np.testing.assert_allclose(product, np.eye(min(shape)), atol=1e-10)


def test_an_orthogonal_gain_sets_the_norm_the_map_scales_by():
    """The singular values are the factors the map stretches by, and they are all `gain`. Asserting them
    rather than the entries says the scaling left the matrix orthogonal instead of merely larger."""
    value = OrthogonalInitializer(gain=2.0).sample((4, 4), "float64", np.random.default_rng(1))

    np.testing.assert_allclose(np.linalg.svd(value, compute_uv=False), 2.0, atol=1e-10)


def test_an_orthogonal_draw_is_not_biased_toward_the_identity():
    """A QR fixes Q only up to the signs of R's diagonal, and the ones LAPACK returns lean positive: taking
    Q as it comes gives a mean diagonal around -0.2 rather than 0, a draw quietly pulled off the group's
    center. Nothing about a single draw's orthogonality would reveal it."""
    rng = np.random.default_rng(0)
    diagonals = [
        np.diag(OrthogonalInitializer().sample((4, 4), "float64", rng)).mean() for _ in range(2000)
    ]

    assert np.mean(diagonals) == pytest.approx(0.0, abs=0.02)


@pytest.mark.parametrize("shape", [(5,), (4, 4, 4)], ids=["bias", "conv_kernel"])
def test_an_orthogonal_draw_refuses_a_parameter_that_is_not_a_matrix(shape):
    """Every other rank needs a flattening convention to be orthogonal at all, and the frameworks disagree
    about which one. Raising beats picking one silently on the caller's behalf. The 1-D case is the one a
    caller actually hits, by handing the layer's own default to a bias."""
    with pytest.raises(ValueError, match="needs a matrix"):
        OrthogonalInitializer().sample(shape, "float64", np.random.default_rng(0))


def test_initial_value_draws_at_floatx_in_the_requested_shape():
    """What every layer relies on when it creates a parameter: the dtype matches the graph's, so nothing has
    to remember `config.floatX` at each of the five call sites."""
    value = XavierNormalInitializer().initial_value((6, 4))

    assert value.shape == (6, 4)
    assert str(value.dtype) == pytensor.config.floatX
    assert len(np.unique(value)) > 1  # drawn, not filled


class TestPerParameterInitializers:
    """Naming a parameter is how a caller overrules its declaration, which is otherwise impossible: the
    declaration is the only thing a redraw consults, and the alternative is assigning `param.initializer`
    between construction and initialization."""

    def test_a_named_initializer_beats_a_declaration(self):
        # A norm scale declares ones precisely so nothing else can move it; naming the parameter is the one
        # thing that should, since the caller picked this parameter rather than every parameter.
        scale = trainable(np.zeros(4, dtype="float64"), "scale", initializer=OneInitializer())

        [value] = initialize_params([scale], initializers={scale: constant(value=7.0)}, rng=0)

        np.testing.assert_allclose(value, 7.0)

    def test_an_unnamed_parameter_keeps_its_declaration(self):
        scale = trainable(np.zeros(4, dtype="float64"), "scale", initializer=OneInitializer())
        weight = trainable(np.zeros((4, 4), dtype="float64"), "w", initializer=ZeroInitializer())

        scale_value, weight_value = initialize_params(
            [scale, weight], initializers={weight: constant(value=7.0)}, rng=0
        )

        np.testing.assert_allclose(
            scale_value, 1.0
        )  # its declaration, untouched by the entry below
        np.testing.assert_allclose(weight_value, 7.0)

    def test_an_entry_is_keyed_by_the_parameter_not_its_name(self):
        """Two parameters can share a name -- nothing prevents it, and optimizer state collides on it rather
        than parameters -- so an entry keyed by name would reach both. Identity reaches one."""
        declared = ZeroInitializer()
        first = trainable(np.zeros((4, 4), dtype="float64"), "w", initializer=declared)
        second = trainable(np.zeros((4, 4), dtype="float64"), "w", initializer=declared)

        first_value, second_value = initialize_params(
            [first, second], initializers={first: constant(value=7.0)}, rng=0
        )

        np.testing.assert_allclose(first_value, 7.0)
        np.testing.assert_allclose(second_value, 0.0)

    def test_naming_a_parameter_that_is_not_being_initialized_does_nothing(self):
        """The mapping is consulted per parameter rather than iterated, so an entry for something outside
        `params` is inert instead of an error -- one dict can serve several initialize calls."""
        weight = trainable(np.zeros((4, 4), dtype="float64"), "w", initializer=ZeroInitializer())
        elsewhere = trainable(np.zeros((4, 4), dtype="float64"), "elsewhere")

        [value] = initialize_params([weight], initializers={elsewhere: constant(value=7.0)}, rng=0)

        np.testing.assert_allclose(value, 0.0)


class TestInitializerDecorator:
    def test_a_scaled_sampler_gets_its_fans_from_the_shape(self):
        """Nothing is supplied but the draw's two arguments, so a fan-scaled sampler computes its own. This
        also pins that `fans` reads the leading dimension as the fan-in, matching how layers build weights."""

        @initializer
        def fan_in_fill(rng, shape):
            fan_in, _ = fans(shape)
            return np.full(shape, fan_in)

        value = fan_in_fill().sample((4, 6), "float64", np.random.default_rng(0))

        np.testing.assert_array_equal(value, 4)

    @pytest.mark.parametrize(
        "source",
        ["def sampler(shape, rng): ...", "def sampler(shape): ...", "def sampler(): ..."],
        ids=["wrong_order", "rng_missing", "neither"],
    )
    def test_requires_the_draw_arguments_first(self, source):
        """Position is the contract, so a sampler that ignores `rng` still declares it. Caught at decoration,
        naming what was declared, rather than as a confusing argument mismatch on the first draw."""
        namespace: dict = {}
        exec(source, namespace)

        with pytest.raises(ValueError, match="must take rng and shape as its first two parameters"):
            initializer(namespace["sampler"])

    def test_the_draw_arguments_arrive_in_that_order(self):
        """Cheap to get backwards, and a swap would hand `rng` a tuple. A real draw off the generator, sized
        by the shape, is only possible if both arrived as themselves."""

        @initializer
        def drawn(rng, shape):
            return rng.normal(size=shape)

        value = drawn().sample((3,), "float64", np.random.default_rng(0))

        assert value.shape == (3,)
        assert len(np.unique(value)) == 3

    def test_the_dtype_is_applied_for_the_sampler(self):
        """The ergonomic point of taking dtype out of the signature: numpy defaults to float64, and a float64
        array assigned to a float32 parameter raises about container precision, naming no initializer."""

        @initializer
        def float64_fill(rng, shape):
            return np.full(shape, 7.0)

        assert float64_fill().sample((2,), "float32", np.random.default_rng(0)).dtype == np.float32

    def test_a_parameter_with_a_default_may_be_omitted(self):
        @initializer
        def scaled(rng, shape, factor=2.0):
            return np.full(shape, factor)

        rng = np.random.default_rng(0)

        np.testing.assert_array_equal(scaled().sample((2,), "float64", rng), 2.0)
        np.testing.assert_array_equal(scaled(factor=5.0).sample((2,), "float64", rng), 5.0)

    def test_every_name_after_the_draw_arguments_is_a_parameter(self):
        """The promise of reading the signature: parameters are whatever you call them. No name is reserved
        and none has to be marked out -- position decides, and everything past the first two is yours."""

        @initializer
        def mine(rng, shape, foo):
            return np.full(shape, foo)

        assert mine.__props__ == ("foo",)
        np.testing.assert_array_equal(
            mine(foo=42).sample((2,), "float64", np.random.default_rng(0)), 42
        )

    def test_one_initializer_draws_correctly_for_every_shape_it_is_used_on(self):
        """Parameters are frozen at construction, so anything that varies with the parameter being drawn has
        to be derived per draw. One instance reused across two layers is the ordinary case, and a shape-
        dependent value cached on the instance would serve the second layer the first one's number."""

        @initializer
        def fan_in_fill(rng, shape):
            fan_in, _ = fans(shape)
            return np.full(shape, fan_in)

        drawn = fan_in_fill()

        np.testing.assert_array_equal(drawn.sample((4, 6), "float64", np.random.default_rng(0)), 4)
        np.testing.assert_array_equal(drawn.sample((8, 2), "float64", np.random.default_rng(0)), 8)

    def test_a_draw_that_forgot_the_shape_says_so(self):
        """The mistake omitting `shape` invites. Deliberately not broadcast: filling a parameter with one
        drawn number gives every unit the same weight, which is the failure a fan-scaled draw exists to
        avoid, and it would pass silently."""

        @initializer
        def forgot_the_size(rng, shape):
            return rng.normal(0.0, 1.0)

        with pytest.raises(
            ValueError, match=r"returned shape \(\) for a parameter of shape \(2, 2\)"
        ):
            forgot_the_size().sample((2, 2), "float64", np.random.default_rng(0))

    @pytest.mark.parametrize(
        "source, expected",
        [
            ("def sampler(rng, shape, *rest): ...", r"takes \*rest"),
            ("def sampler(rng, shape, **rest): ...", r"takes \*\*rest"),
        ],
        ids=["var_positional", "var_keyword"],
    )
    def test_rejects_a_signature_it_cannot_record(self, source, expected):
        """A var-arg has no name to store, so it cannot be recorded. Caught at decoration rather than at the
        first draw."""
        namespace: dict = {}
        exec(source, namespace)

        with pytest.raises(ValueError, match=expected):
            initializer(namespace["sampler"])

    @pytest.mark.parametrize(
        "given, expected",
        [({}, "missing parameters"), ({"std": 1.0}, "unexpected parameters")],
        ids=["missing", "unexpected"],
    )
    def test_rejects_the_wrong_parameters_at_construction(self, given, expected):
        @initializer
        def scaled(rng, shape, factor):
            return np.full(shape, factor)

        with pytest.raises(TypeError, match=expected):
            scaled(**given)


def test_initial_values_from_replaces_the_draw_and_leaves_the_declared_law_alone():
    """A checkpoint fills every weight, so drawing one first is wasted -- but the parameter still has to
    know how to redraw itself, or reinitializing a loaded model would produce garbage."""
    with initial_values_from(ZeroInitializer()):
        layer = Linear("fc", n_in=4, n_out=4)

    np.testing.assert_array_equal(layer.W.get_value(), 0.0)
    assert isinstance(layer.W.initializer, XavierNormalInitializer)
    assert (initialize_params([layer.W], rng=0)[0] != 0).any()


def test_initial_values_from_unwinds_when_the_body_raises():
    with pytest.raises(ValueError, match="boom"):
        with initial_values_from(ZeroInitializer()):
            raise ValueError("boom")

    assert (Linear("fc", n_in=4, n_out=4).W.get_value() != 0).any()


def test_an_empty_draw_allocates_the_shape_without_touching_the_generator():
    """The point is to do no work, so a draw that quietly consumed entropy would mean it still ran."""
    rng = np.random.default_rng(0)
    value = EmptyInitializer().sample((3, 4), "float32", rng)

    assert value.shape == (3, 4)
    assert value.dtype == np.float32
    assert rng.bit_generator.state == np.random.default_rng(0).bit_generator.state
