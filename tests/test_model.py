import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import config

import pytensor_ml.model

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import GRU, BatchNorm, LayerNorm, Linear, Sequential
from pytensor_ml.loss import SquaredError
from pytensor_ml.model import Model
from pytensor_ml.optim import sgd
from pytensor_ml.state import ZeroInitializer
from tests.conftest import constant


class TestModelPredict:
    def test_matches_a_hand_computed_forward_pass(self):
        X = pt.tensor("X", shape=(None, 6))
        fc1 = Linear("fc1", n_in=6, n_out=3)
        fc2 = Linear("fc2", n_in=3, n_out=1)
        model = Model(Sequential(fc1, fc2)(X)).initialize(seed=42)

        X_test = np.random.default_rng(0).normal(size=(10, 6)).astype(config.floatX)
        hidden = X_test @ fc1.W.get_value() + fc1.b.get_value()
        expected = hidden @ fc2.W.get_value() + fc2.b.get_value()

        result = model.predict(X_test)

        assert result.shape == (10, 1)
        assert result.dtype == config.floatX
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_normalizes_with_running_stats_not_batch_stats(self):
        X = pt.tensor("X", shape=(None, 4))
        fc1 = Linear("fc1", n_in=4, n_out=4)
        bn = BatchNorm("bn1", n_in=4)
        model = Model(Sequential(fc1, bn)(X)).initialize(seed=42)

        bn.running_mean.set_value(np.array([1.0, 2.0, 3.0, 4.0], dtype=config.floatX))
        bn.running_var.set_value(np.ones(4, dtype=config.floatX))

        rng = np.random.default_rng(0)
        # Two batch sizes: batch statistics would differ between them, running statistics cannot.
        for n_rows in (5, 20):
            X_test = rng.normal(size=(n_rows, 4)).astype(config.floatX)
            fc_out = X_test @ fc1.W.get_value() + fc1.b.get_value()
            standardized = (fc_out - bn.running_mean.get_value()) / np.sqrt(
                bn.running_var.get_value() + bn.epsilon
            )
            expected = standardized * bn.scale.get_value() + bn.loc.get_value()

            np.testing.assert_allclose(model.predict(X_test), expected, rtol=1e-5)

    def test_compiles_once_and_reuses_the_function(self, monkeypatch):
        X = pt.tensor("X", shape=(None, 4))
        model = Model(Linear("fc1", n_in=4, n_out=2)(X)).initialize(seed=42)

        uncounted_compile_predict = pytensor_ml.model.compile_predict
        compile_count = 0

        def counting_compile_predict(*args, **kwargs):
            nonlocal compile_count
            compile_count += 1
            return uncounted_compile_predict(*args, **kwargs)

        monkeypatch.setattr(pytensor_ml.model, "compile_predict", counting_compile_predict)

        X_test = np.random.default_rng(0).normal(size=(5, 4)).astype(config.floatX)
        first = model.predict(X_test)
        second = model.predict(X_test)

        assert compile_count == 1
        np.testing.assert_array_equal(first, second)

    def test_takes_one_array_per_data_input(self):
        """A sequence mask is a second data input, and nothing about the model says which is which, so
        the arrays have to reach the variables the graph reads rather than merely fit their shapes."""
        X = pt.tensor("X", shape=(None, None, 4))
        mask = pt.matrix("mask")
        model = Model(GRU("gru", n_in=4, n_hidden=3)(X, mask=mask)).initialize(seed=0)

        X_test = np.random.default_rng(0).normal(size=(2, 5, 4)).astype(config.floatX)
        # Padded, so a mask that never arrived would give a different answer than one that did.
        mask_test = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=config.floatX)

        result = model.predict(X_test, mask_test)

        assert result.shape == (2, 5, 3)
        np.testing.assert_allclose(result, model.y.eval({X: X_test, mask: mask_test}), rtol=1e-6)

    def test_says_which_inputs_it_wanted_when_given_the_wrong_count(self):
        """The count is checked before compiling, where a missing input would otherwise surface as a
        pytensor error naming an internal op rather than the input."""
        X = pt.tensor("X", shape=(None, None, 4))
        mask = pt.matrix("mask")
        model = Model(GRU("gru", n_in=4, n_hidden=3)(X, mask=mask)).initialize(seed=0)

        with pytest.raises(ValueError, match=r"reads 2 \(X, mask\), and 1 were given"):
            model.predict(np.zeros((2, 5, 4), dtype=config.floatX))


class TestModelInitialize:
    @pytest.mark.parametrize("norm_layer", [BatchNorm, LayerNorm], ids=["batch_norm", "layer_norm"])
    def test_leaves_a_norm_layer_at_the_identity_transform(self, norm_layer):
        X = pt.tensor("X", shape=(None, 8))
        norm = norm_layer("norm", n_in=4)
        y = Sequential(
            Linear("fc1", n_in=8, n_out=4), norm, ReLU(), Linear("fc2", n_in=4, n_out=2)
        )(X)

        Model(y).initialize(seed=0)

        np.testing.assert_array_equal(norm.scale.get_value(), 1)
        np.testing.assert_array_equal(norm.loc.get_value(), 0)

    def test_draws_weight_matrices_and_leaves_biases_at_zero(self):
        X = pt.tensor("X", shape=(None, 8))
        fc1 = Linear("fc1", n_in=8, n_out=4)
        y = Sequential(fc1, ReLU(), Linear("fc2", n_in=4, n_out=2))(X)

        Model(y).initialize(seed=0)

        assert np.abs(fc1.W.get_value()).min() > 0
        np.testing.assert_array_equal(fc1.b.get_value(), 0)

    def test_the_seed_is_what_a_reproducible_run_rests_on(self):
        """The whole remaining job of `initialize`: same seed, same network, same values. The differing-seed
        half is what fails if the seed is dropped on the way to the draw, which same-seed alone would miss --
        construction draws from fresh entropy, so two networks agree only because the seed made them."""

        def values_for(seed):
            X = pt.tensor("X", shape=(None, 8))
            y = Sequential(Linear("fc1", n_in=8, n_out=4), ReLU(), Linear("fc2", n_in=4, n_out=2))(
                X
            )
            model = Model(y).initialize(seed=seed)
            return {p.name: p.get_value() for p in model.weights}

        first, again, other = values_for(0), values_for(0), values_for(1)

        for name, value in first.items():
            np.testing.assert_array_equal(value, again[name], err_msg=name)
        assert not np.array_equal(first["fc1_W"], other["fc1_W"])

    def test_a_declared_initializer_does_not_freeze_the_parameter(self):
        """Declaring an initializer protects a starting value, not the parameter. Excluding declared
        parameters from training instead would leave batch norm's scale pinned at one and still satisfy
        every assertion above."""
        rng = np.random.default_rng(0)
        X = pt.tensor("X", shape=(None, 8))
        norm = BatchNorm("norm", n_in=4)
        y = Sequential(
            Linear("fc1", n_in=8, n_out=4), norm, ReLU(), Linear("fc2", n_in=4, n_out=2)
        )(X)
        model = Model(y).initialize(seed=0)

        target = pt.matrix("target")
        step = model.compile_train(
            sgd(learning_rate=0.1), loss=((y - target) ** 2).mean(), inputs=[X, target]
        )
        step(
            rng.normal(size=(8, 8)).astype(config.floatX),
            rng.normal(size=(8, 2)).astype(config.floatX),
        )

        assert not np.array_equal(norm.scale.get_value(), np.ones(4))
        assert not np.array_equal(norm.loc.get_value(), np.zeros(4))


def test_compile_train_accepts_a_prebuilt_loss():
    # An autoencoder reconstructs its own input, so there is no target separate from X and the supervised
    # path cannot express it. The step takes one argument, not two.
    X = pt.tensor("X", shape=(None, 4))
    reconstruction = Sequential(Linear("enc", n_in=4, n_out=2), Linear("dec", n_in=2, n_out=4))(X)
    model = Model(reconstruction).initialize(seed=0)

    step = model.compile_train(sgd(learning_rate=1e-2), loss=SquaredError()(X, reconstruction))

    X_batch = np.random.default_rng(0).normal(size=(64, 4)).astype(config.floatX)
    history = [float(step(X_batch)) for _ in range(50)]

    assert history[-1] < history[0]


def test_compile_train_reduces_loss():
    X = pt.tensor("X", shape=(None, 4))
    y = Sequential(Linear("fc1", n_in=4, n_out=8), Linear("fc2", n_in=8, n_out=1))(X)
    model = Model(y).initialize(seed=0)

    step = model.compile_train(sgd(learning_rate=1e-2), SquaredError())

    rng = np.random.default_rng(0)
    X_batch = rng.normal(size=(64, 4)).astype(config.floatX)
    target = rng.normal(size=(64, 1)).astype(config.floatX)

    history = [float(step(X_batch, target)) for _ in range(50)]

    assert history[-1] < history[0]


def test_a_layer_keyword_survives_a_redraw():
    """A zeroed output head is a real technique, and the constructor keyword is how it is asked for. The
    redraw has to honor it rather than treat every weight matrix as a fresh Xavier draw."""
    X = pt.tensor("X", shape=(None, 8))
    first = Linear("fc1", n_in=8, n_out=4)
    head = Linear("head", n_in=4, n_out=2, weight_initializer=ZeroInitializer())
    y = Sequential(first, ReLU(), head)(X)

    Model(y).initialize(seed=0)

    np.testing.assert_array_equal(head.W.get_value(), 0)  # redrawn from its declaration
    assert np.abs(first.W.get_value()).min() > 0  # and its sibling from the layer default


def test_initialize_takes_per_parameter_initializers():
    """The model-level entry point a user actually calls. Reaching a parameter through the layer that owns
    it is what makes a bias inside a composed layer addressable, since no constructor keyword exposes one."""
    X = pt.tensor("X", shape=(None, 8))
    fc1 = Linear("fc1", n_in=8, n_out=4)
    norm = BatchNorm("norm", n_in=4)
    y = Sequential(fc1, norm, ReLU(), Linear("fc2", n_in=4, n_out=2))(X)
    drawn = constant(value=7.0)

    Model(y).initialize(seed=0, initializers={fc1.b: drawn, norm.scale: drawn})

    np.testing.assert_allclose(fc1.b.get_value(), 7.0)  # outranks its zero declaration
    np.testing.assert_allclose(norm.scale.get_value(), 7.0)  # and the norm's unit declaration
    np.testing.assert_allclose(norm.loc.get_value(), 0.0)  # unnamed, so still its declaration
    assert len(np.unique(fc1.W.get_value())) > 1  # its own declaration, since nothing named it


def test_a_constant_reaches_one_parameter_through_initializers():
    """The same zeroed head as above, asked for at initialize time rather than at construction -- the route
    for a parameter whose layer you did not build, such as one inside a loaded network."""
    X = pt.tensor("X", shape=(None, 8))
    first = Linear("fc1", n_in=8, n_out=4)
    head = Linear("head", n_in=4, n_out=2)
    y = Sequential(first, ReLU(), head)(X)

    Model(y).initialize(seed=0, initializers={head.W: ZeroInitializer()})

    np.testing.assert_array_equal(head.W.get_value(), 0)
    assert np.abs(first.W.get_value()).min() > 0  # its sibling still drew from its declaration


def test_compile_train_takes_one_batch_per_data_input():
    """The supervised path builds the target itself, so it also has to know every input the model reads."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.matrix("mask")
    model = Model(GRU("gru", n_in=4, n_hidden=3)(X, mask=mask)).initialize(seed=0)

    step = model.compile_train(sgd(1e-3), SquaredError())

    rng = np.random.default_rng(0)
    X_batch = rng.normal(size=(2, 5, 4)).astype(config.floatX)
    mask_batch = np.ones((2, 5), dtype=config.floatX)
    target_batch = rng.normal(size=(2, 5, 3)).astype(config.floatX)

    losses = [float(step(X_batch, mask_batch, target_batch)) for _ in range(20)]

    assert losses[-1] < losses[0]
