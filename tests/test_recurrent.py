import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.activations import Activation, ReLU, Tanh
from pytensor_ml.layers import (
    GRU,
    LSTM,
    RNN,
    Bidirectional,
    ElmanCell,
    GRUCell,
    Input,
    Linear,
    Recurrent,
    RecurrentCell,
)
from pytensor_ml.loss import SquaredError
from pytensor_ml.model import Model
from pytensor_ml.optim import adam
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import collect_trainable_params
from pytensor_ml.state import OneInitializer, ZeroInitializer

floatX = pytensor.config.floatX

# The reference loop below sums in a different order than the graph, so the gap tracks the precision.
ATOL = 1e-6 if floatX == "float64" else 1e-5


@pytest.fixture
def rng():
    return np.random.default_rng(sum(map(ord, "pytensor_ml recurrent")))


def unrolled(X_np, W_ih, b, W_hh, phi, h0=None):
    """The recurrence written as a python loop, one step at a time, as the reference to check against."""
    h = np.zeros((*X_np.shape[:-2], W_hh.shape[0]), dtype=floatX) if h0 is None else h0
    states = []
    for t in range(X_np.shape[-2]):
        h = phi(X_np[..., t, :] @ W_ih + b + h @ W_hh)
        states.append(h)
    return np.stack(states, axis=-2)


def draw_parameters(layer, rng):
    """Set every parameter to a fresh draw and hand the values back for the reference to use."""
    W_ih = rng.normal(size=(layer.cell.n_in, layer.cell.n_hidden)).astype(floatX)
    b = rng.normal(size=(layer.cell.n_hidden,)).astype(floatX)
    W_hh = rng.normal(size=(layer.cell.n_hidden, layer.cell.n_hidden)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.b.set_value(b)
    layer.cell.W_hh.set_value(W_hh)
    return W_ih, b, W_hh


@pytest.mark.parametrize(
    "activation, reference",
    [(Tanh(), np.tanh), (ReLU(), lambda x: np.maximum(x, 0.0))],
    ids=["tanh", "relu"],
)
def test_matches_a_step_by_step_reference(activation, reference, rng):
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3, activation=activation)
    out = layer(X)
    assert out.type.shape == (None, None, 3)

    W_ih, b, W_hh = draw_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled(X_np, W_ih, b, W_hh, reference), atol=ATOL
    )


def test_starts_from_a_given_state(rng):
    """The state a caller hands in has to reach the first step, not just sit in the graph -- a zeros
    default that quietly ignored it would agree with the reference on every other test here."""
    X = pt.tensor("X", shape=(None, None, 4))
    h0 = pt.tensor("h0", shape=(None, 3))
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X, h0)

    W_ih, b, W_hh = draw_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    h0_np = rng.normal(size=(5, 3)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np, h0: h0_np}),
        unrolled(X_np, W_ih, b, W_hh, np.tanh, h0=h0_np),
        atol=ATOL,
    )


def test_the_recurrent_weight_first_acts_on_the_second_step(rng):
    """The recurrence is real and runs forward in time. Starting from a zero state, the first output does
    not touch the recurrent weight at all and the second does, which is what separates a scan from an
    input projection applied position by position."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X)
    draw_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    first, second = (
        pytensor.grad(out[..., step, :].sum(), layer.cell.W_hh).eval({X: X_np}) for step in (0, 1)
    )

    np.testing.assert_allclose(first, np.zeros((3, 3)), atol=ATOL)
    assert np.abs(second).max() > 1e-3


def test_every_parameter_is_reachable_through_the_scan(rng):
    """The recurrent weight enters the graph as a non-sequence of the scan rather than as a plain input, so
    a collector that stopped at the scan node would train the projection and leave the recurrence frozen."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X)

    assert set(collect_trainable_params(out)) == {
        layer.cell.W_ih,
        layer.cell.b,
        layer.cell.W_hh,
    }


def test_an_activation_brings_its_own_parameters_into_the_recurrence():
    """The step closes over whatever the activation holds, and scan lifts it in. A strict scan would reject
    a parameterized activation instead, telling the caller to add it to an input list they do not have."""

    class PReLU(Activation):
        def __init__(self):
            self.slope = trainable(
                np.asarray(0.25, dtype=floatX), "prelu_slope", initializer=OneInitializer()
            )

        def __call__(self, x):
            return pt.switch(x > 0, x, self.slope * x)

    activation = PReLU()
    layer = RNN("rnn", n_in=4, n_hidden=3, activation=activation)
    out = layer(pt.tensor("X", shape=(None, None, 4)))

    assert activation.slope in collect_trainable_params(out)


def test_trains_end_to_end(rng):
    """Gradients survive the round trip through the scan and the training machinery moves the parameters."""
    X = Input("X", shape=(None, 6, 4))
    y = Linear("head", n_in=5, n_out=1)(RNN("rnn", n_in=4, n_hidden=5)(X)[..., -1, :])
    model = Model(y).initialize(seed=1)
    step = model.compile_train(adam(learning_rate=0.05), SquaredError())

    X_np = rng.normal(size=(32, 6, 4)).astype(floatX)
    y_np = X_np.sum(axis=(1, 2))[:, None].astype(floatX)

    losses = [float(step(X_np, y_np)) for _ in range(50)]
    assert losses[-1] < losses[0] / 5


def test_the_recurrent_weight_is_drawn_orthogonal_by_default():
    """Applied once per step, so its singular values compound: at one they leave the state alone however
    long the sequence, and spread around one they explode the gradient along some directions while
    vanishing it along others. The input weight keeps the usual fan-scaled draw, checked structurally as
    well as by spread -- on a square matrix the two draws have the same entry standard deviation, so
    spread alone would not notice it picking up the recurrent default."""
    layer = RNN("rnn", n_in=16, n_hidden=64)

    W_hh = layer.cell.W_hh.get_value()
    np.testing.assert_allclose(W_hh.T @ W_hh, np.eye(64), atol=ATOL)

    W_ih = layer.cell.W_ih.get_value()
    assert np.abs(W_ih @ W_ih.T - np.eye(16)).max() > 0.1
    assert W_ih.std() == pytest.approx(np.sqrt(2.0 / 80), rel=0.1)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_the_bias_is_optional(bias, rng):
    """Dropping the bias has to drop the parameter as well as the term. Leaving an unused one behind would
    hand the optimizer moment state to carry for a weight that never moves, and nothing else here builds
    the layer without it."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3, bias=bias)
    out = layer(X)

    W_ih = rng.normal(size=(4, 3)).astype(floatX)
    W_hh = rng.normal(size=(3, 3)).astype(floatX)
    b = rng.normal(size=(3,)).astype(floatX) if bias else np.zeros(3, dtype=floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.W_hh.set_value(W_hh)
    if bias:
        layer.cell.b.set_value(b)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    assert set(collect_trainable_params(out)) == (
        {layer.cell.W_ih, layer.cell.W_hh, layer.cell.b}
        if bias
        else {layer.cell.W_ih, layer.cell.W_hh}
    )
    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL
    )


def test_the_recurrent_weight_takes_its_own_initializer():
    """The recurrent draw has a keyword of its own, and using it must not disturb the input projection,
    which shares the layer's other two."""
    layer = RNN("rnn", n_in=4, n_hidden=3, recurrent_initializer=ZeroInitializer())

    np.testing.assert_array_equal(layer.cell.W_hh.get_value(), np.zeros((3, 3)))
    assert np.abs(layer.cell.W_ih.get_value()).max() > 0.0


@pytest.mark.parametrize(
    "batch_shape", [(), (5,), (2, 5)], ids=["unbatched", "one_axis", "two_axes"]
)
def test_recurs_over_any_number_of_batch_axes(batch_shape, rng):
    """Time is the second-to-last axis, as it is for every other layer here. Taking the batch axis to be
    the leading one instead would give the right answer for a single batch axis and quietly transpose a
    stacked one -- and refuse a bare sequence, which needs no batch axis at all."""
    X = pt.tensor("X", shape=(*(None for _ in batch_shape), None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X)

    W_ih, b, W_hh = draw_parameters(layer, rng)
    X_np = rng.normal(size=(*batch_shape, 7, 4)).astype(floatX)

    result = out.eval({X: X_np})
    assert result.shape == (*batch_shape, 7, 3)
    np.testing.assert_allclose(result, unrolled(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL)


def test_the_state_takes_the_dtype_the_step_produces():
    """A float32 network fed a float64 sequence. The step promotes, so the state has to promote with it;
    a state pinned to floatX leaves scan comparing float32 against the float64 its inner function returns
    and refusing the graph. Nothing else here catches it, because every other test runs at one dtype."""
    with pytensor.config.change_flags(floatX="float32"):
        layer = RNN("rnn", n_in=4, n_hidden=3)
        X = pt.tensor("X", shape=(None, None, 4), dtype="float64")
        out = layer(X)

        assert out.dtype == "float64"
        assert out.eval({X: np.zeros((2, 5, 4), dtype="float64")}).dtype == np.dtype("float64")


class TwoStateCell(RecurrentCell):
    """A cell carrying more than one tensor, as an LSTM does. It sums the input into the first state and
    counts steps in the second, so both have to survive the round trip to give the right answer."""

    def __init__(self, n_hidden):
        self.n_hidden = n_hidden

    def step(self, x_t, running_sum, count):
        return running_sum + x_t, count + 1.0

    def initial_state(self, X):
        zeros = pt.zeros((*X.shape[:-2], self.n_hidden), dtype=X.dtype)
        return zeros, pt.zeros_like(zeros)


def test_a_cell_may_carry_more_than_one_state(rng):
    """Scan hands back a list once a cell carries several states rather than the bare variable it returns
    for one, and only the first is the output. Taking the wrong one would return the step count."""
    X = pt.tensor("X", shape=(None, None, 3))
    out = Recurrent(TwoStateCell(3), name="two_state")(X)

    X_np = rng.normal(size=(5, 7, 3)).astype(floatX)

    np.testing.assert_allclose(out.eval({X: X_np}), np.cumsum(X_np, axis=-2), atol=ATOL)


def test_a_cell_carrying_several_states_can_be_started_from_all_of_them(rng):
    """Each state has to reach the step it belongs to. Threading them in the wrong order, or dropping all
    but the first, would still run and still return something of the right shape."""
    X = pt.tensor("X", shape=(None, None, 3))
    start = pt.tensor("start", shape=(None, 3))
    out = Recurrent(TwoStateCell(3), name="two_state")(X, [start, pt.zeros_like(start)])

    X_np = rng.normal(size=(5, 7, 3)).astype(floatX)
    start_np = rng.normal(size=(5, 3)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np, start: start_np}),
        np.cumsum(X_np, axis=-2) + start_np[:, None, :],
        atol=ATOL,
    )


def test_a_rejected_state_names_which_one_of_several_is_wrong():
    """The message carries a position because a cell may carry many states, and a caller staring at two
    identically shaped arguments needs to know which one is wrong. A hardcoded index would read as 0 here."""
    X = pt.tensor("X", shape=(None, None, 3))
    good, bad = pt.matrix("good"), pt.vector("bad")

    with pytest.raises(ValueError, match="state at position 1; got a 1-dimensional one"):
        Recurrent(TwoStateCell(3), name="two_state")(X, [good, bad])


def test_rejects_a_starting_state_the_cell_does_not_carry():
    """A cell's state count is part of its contract, and scan would otherwise report the mismatch from
    inside the inner function, where the message names nothing the caller wrote."""
    X = pt.tensor("X", shape=(None, None, 3))

    with pytest.raises(ValueError, match="carries 2 state tensor\\(s\\), but got 1"):
        Recurrent(TwoStateCell(3), name="two_state")(X, pt.matrix("only_one"))


def test_the_rnn_is_a_recurrent_over_an_elman_cell():
    """The flat constructor is a convenience over the same two pieces, so a caller who wants the cell on
    its own gets exactly what the layer would have built."""
    layer = RNN("rnn", n_in=4, n_hidden=3)

    assert isinstance(layer, Recurrent)
    assert isinstance(layer.cell, ElmanCell)
    assert [p.name for p in (layer.cell.W_ih, layer.cell.b, layer.cell.W_hh)] == [
        "rnn_W_ih",
        "rnn_b",
        "rnn_W_hh",
    ]


def test_a_hand_built_cell_scans_the_same_as_the_flat_constructor(rng):
    """What the split is for: `Recurrent` takes any cell, and wrapping the same one the flat constructor
    builds has to give the same graph back."""
    X = pt.tensor("X", shape=(None, None, 4))
    cell = ElmanCell("cell", n_in=4, n_hidden=3)
    wrapped = Recurrent(cell, name="wrapped")(X)

    W_ih = rng.normal(size=(4, 3)).astype(floatX)
    b = rng.normal(size=(3,)).astype(floatX)
    W_hh = rng.normal(size=(3, 3)).astype(floatX)
    cell.W_ih.set_value(W_ih)
    cell.b.set_value(b)
    cell.W_hh.set_value(W_hh)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    np.testing.assert_allclose(
        wrapped.eval({X: X_np}), unrolled(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL
    )


def test_rejects_an_input_with_no_time_axis():
    layer = RNN("rnn", n_in=4, n_hidden=3)

    with pytest.raises(ValueError, match="no time axis to recur over"):
        layer(pt.tensor("X", shape=(4,)))


def test_rejects_an_initial_state_that_does_not_match_the_batch_axes():
    """The state carries one value per batch element, so its rank is fixed by the input's. Scan would
    otherwise broadcast a mismatched state into the recurrence and return a silently wrong shape."""
    layer = RNN("rnn", n_in=4, n_hidden=3)

    with pytest.raises(
        ValueError, match="needs a 2-dimensional state at position 0; got a 1-dimensional one"
    ):
        layer(pt.tensor("X", shape=(None, None, 4)), pt.tensor("h0", shape=(3,)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def unrolled_gru(X_np, W_ih, b, W_hh, c, phi, gate=sigmoid):
    """The gated recurrence written as a python loop, as the reference to check the scan against."""
    n_hidden = W_hh.shape[0]
    h = np.zeros((*X_np.shape[:-2], n_hidden), dtype=floatX)
    states = []
    for t in range(X_np.shape[-2]):
        from_input = X_np[..., t, :] @ W_ih + b
        from_state = h @ W_hh
        reset = gate(from_input[..., :n_hidden] + from_state[..., :n_hidden])
        update = gate(
            from_input[..., n_hidden : 2 * n_hidden] + from_state[..., n_hidden : 2 * n_hidden]
        )
        candidate = phi(
            from_input[..., 2 * n_hidden :] + reset * (from_state[..., 2 * n_hidden :] + c)
        )
        h = (1 - update) * candidate + update * h
        states.append(h)
    return np.stack(states, axis=-2)


def draw_gru_parameters(layer, rng):
    """Set every parameter to a fresh draw and hand the values back for the reference to use."""
    n_in, n_hidden = layer.cell.n_in, layer.cell.n_hidden
    W_ih = rng.normal(size=(n_in, 3 * n_hidden)).astype(floatX)
    W_hh = rng.normal(size=(n_hidden, 3 * n_hidden)).astype(floatX)
    b = rng.normal(size=(3 * n_hidden,)).astype(floatX)
    c = rng.normal(size=(n_hidden,)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.W_hh.set_value(W_hh)
    layer.cell.b.set_value(b)
    layer.cell.c.set_value(c)
    return W_ih, b, W_hh, c


@pytest.mark.parametrize(
    "activation, reference",
    [(Tanh(), np.tanh), (ReLU(), lambda x: np.maximum(x, 0.0))],
    ids=["tanh", "relu"],
)
def test_the_gru_matches_a_step_by_step_reference(activation, reference, rng):
    X = pt.tensor("X", shape=(None, None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3, activation=activation)
    out = layer(X)
    assert out.type.shape == (None, None, 3)

    W_ih, b, W_hh, c = draw_gru_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_gru(X_np, W_ih, b, W_hh, c, reference), atol=ATOL
    )


def test_the_gru_gate_slices_do_not_cross(rng):
    """Three gates read three slices of one projection, and swapping two of them still produces a
    plausible sequence. Driving each gate to its own extreme in turn pins which slice is which: the
    reference loop alone would agree with any consistent misordering of the parameter layout."""
    X = pt.tensor("X", shape=(None, None, 4))
    h0 = pt.tensor("h0", shape=(None, 3))
    layer = GRU("gru", n_in=4, n_hidden=3)
    out = layer(X, h0)

    W_in = rng.normal(size=(4, 3)).astype(floatX)
    layer.cell.W_ih.set_value(np.concatenate([np.zeros((4, 6), dtype=floatX), W_in], axis=1))
    layer.cell.W_hh.set_value(np.zeros((3, 9), dtype=floatX))
    layer.cell.c.set_value(np.zeros(3, dtype=floatX))
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    h0_np = rng.normal(size=(5, 3)).astype(floatX)

    # An update gate held open carries the starting state to the end untouched, whatever the input does.
    layer.cell.b.set_value(np.array([0, 0, 0, 20, 20, 20, 0, 0, 0], dtype=floatX))
    held = out.eval({X: X_np, h0: h0_np})
    np.testing.assert_allclose(held, np.broadcast_to(h0_np[:, None, :], (5, 7, 3)), atol=1e-6)

    # A reset gate held shut cuts the state out of the candidate, and with the update gate shut too the
    # step keeps nothing at all: the layer becomes a memoryless projection.
    layer.cell.b.set_value(np.array([-20, -20, -20, -20, -20, -20, 0, 0, 0], dtype=floatX))
    forgotten = out.eval({X: X_np, h0: h0_np})
    np.testing.assert_allclose(forgotten, np.tanh(X_np @ W_in), atol=1e-6)


def test_the_gru_candidate_bias_sits_inside_the_reset_gate(rng):
    """``c`` is a separate parameter from the candidate's slice of ``b`` only because the reset gate
    scales it; folded into ``b`` it would survive a shut gate. Everything else is zeroed, so the whole
    output is the bias the gate does or does not let through."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3)
    out = layer(X)

    layer.cell.W_ih.set_value(np.zeros((4, 9), dtype=floatX))
    layer.cell.W_hh.set_value(np.zeros((3, 9), dtype=floatX))
    c = rng.normal(size=(3,)).astype(floatX)
    layer.cell.c.set_value(c)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    layer.cell.b.set_value(np.array([20, 20, 20, -20, -20, -20, 0, 0, 0], dtype=floatX))
    np.testing.assert_allclose(
        out.eval({X: X_np}), np.broadcast_to(np.tanh(c), (5, 7, 3)), atol=1e-6
    )

    layer.cell.b.set_value(np.array([-20, -20, -20, -20, -20, -20, 0, 0, 0], dtype=floatX))
    np.testing.assert_allclose(out.eval({X: X_np}), np.zeros((5, 7, 3)), atol=1e-6)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_the_gru_biases_are_optional(bias, rng):
    """Dropping the bias drops both parameters as well as both terms; an unused one left behind would
    hand the optimizer moment state to carry for a weight that never moves."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3, bias=bias)
    out = layer(X)

    W_ih = rng.normal(size=(4, 9)).astype(floatX)
    W_hh = rng.normal(size=(3, 9)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.W_hh.set_value(W_hh)
    b = np.zeros(9, dtype=floatX)
    c = np.zeros(3, dtype=floatX)
    if bias:
        b = rng.normal(size=(9,)).astype(floatX)
        c = rng.normal(size=(3,)).astype(floatX)
        layer.cell.b.set_value(b)
        layer.cell.c.set_value(c)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    assert set(collect_trainable_params(out)) == (
        {layer.cell.W_ih, layer.cell.W_hh, layer.cell.b, layer.cell.c}
        if bias
        else {layer.cell.W_ih, layer.cell.W_hh}
    )
    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_gru(X_np, W_ih, b, W_hh, c, np.tanh), atol=ATOL
    )


def test_the_gru_recurrent_weight_is_drawn_orthogonal_by_default():
    """One draw covers all three gates, as in keras, so the check is on the whole wide matrix rather
    than on each gate's block."""
    layer = GRU("gru", n_in=16, n_hidden=32)

    W_hh = layer.cell.W_hh.get_value()
    assert W_hh.shape == (32, 96)
    np.testing.assert_allclose(W_hh @ W_hh.T, np.eye(32), atol=ATOL)

    W_ih = layer.cell.W_ih.get_value()
    assert np.abs(W_ih @ W_ih.T - np.eye(16)).max() > 0.1
    # Both fans count the stacked axis, so the spread would be wrong if the draw saw one gate's shape.
    assert W_ih.std() == pytest.approx(np.sqrt(2.0 / (16 + 96)), rel=0.1)


def test_the_gru_forwards_its_initializers_to_the_cell():
    """Four keyword-only arguments reach the cell through the flat constructor, and one dropped on the
    floor leaves a parameter silently at its default draw. Both biases share the one keyword."""
    layer = GRU(
        "gru",
        n_in=4,
        n_hidden=3,
        recurrent_initializer=ZeroInitializer(),
        bias_initializer=OneInitializer(),
    )

    np.testing.assert_array_equal(layer.cell.W_hh.get_value(), np.zeros((3, 9)))
    np.testing.assert_array_equal(layer.cell.b.get_value(), np.ones(9))
    np.testing.assert_array_equal(layer.cell.c.get_value(), np.ones(3))
    assert np.abs(layer.cell.W_ih.get_value()).max() > 0.0


def test_the_gru_trains_end_to_end(rng):
    """Gradients survive the round trip through the gates and the training machinery moves them."""
    X = Input("X", shape=(None, 6, 4))
    y = Linear("head", n_in=5, n_out=1)(GRU("gru", n_in=4, n_hidden=5)(X)[..., -1, :])
    model = Model(y).initialize(seed=1)
    step = model.compile_train(adam(learning_rate=0.05), SquaredError())

    X_np = rng.normal(size=(32, 6, 4)).astype(floatX)
    y_np = X_np.sum(axis=(1, 2))[:, None].astype(floatX)

    losses = [float(step(X_np, y_np)) for _ in range(50)]
    assert losses[-1] < losses[0] / 5


def test_the_gru_is_a_recurrent_over_a_gru_cell(rng):
    """The flat constructor is a convenience over the cell, and has to build the same graph as writing
    the two out by hand."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3)
    assert isinstance(layer, Recurrent)
    assert isinstance(layer.cell, GRUCell)

    by_hand = Recurrent(GRUCell("gru", n_in=4, n_hidden=3), name="gru")
    W_ih, b, W_hh, c = draw_gru_parameters(layer, rng)
    by_hand.cell.W_ih.set_value(W_ih)
    by_hand.cell.W_hh.set_value(W_hh)
    by_hand.cell.b.set_value(b)
    by_hand.cell.c.set_value(c)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    np.testing.assert_allclose(layer(X).eval({X: X_np}), by_hand(X).eval({X: X_np}), atol=ATOL)


def test_the_gru_gates_take_their_own_activation(rng):
    """The gates and the candidate have separate keywords, and setting one must leave the other alone.
    A hard sigmoid clipped at the same endpoints is the substitution a reader would actually make, and
    it disagrees with the logistic everywhere except the two points where they cross."""

    class HardSigmoid(Activation):
        def __call__(self, x):
            return pt.clip(x * 0.2 + 0.5, 0.0, 1.0)

    def hard_sigmoid(x):
        return np.clip(x * 0.2 + 0.5, 0.0, 1.0)

    X = pt.tensor("X", shape=(None, None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3, gate_activation=HardSigmoid())
    out = layer(X)

    W_ih, b, W_hh, c = draw_gru_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    hard = out.eval({X: X_np})

    np.testing.assert_allclose(
        hard, unrolled_gru(X_np, W_ih, b, W_hh, c, np.tanh, gate=hard_sigmoid), atol=ATOL
    )
    # The candidate still runs through tanh, so the default cell is a different function, not a rescaling.
    logistic = unrolled_gru(X_np, W_ih, b, W_hh, c, np.tanh)
    assert np.abs(hard - logistic).max() > 0.01


@pytest.mark.parametrize(
    "batch_shape", [(), (5,), (2, 5)], ids=["unbatched", "one_axis", "two_axes"]
)
def test_the_gru_recurs_over_any_number_of_batch_axes(batch_shape, rng):
    """Every gate is a slice of the last axis, and the candidate's bias broadcasts against whatever
    batch axes precede it. A bare sequence has none at all, and a stacked batch has two."""
    X = pt.tensor("X", shape=(*(None for _ in batch_shape), None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3)
    out = layer(X)

    W_ih, b, W_hh, c = draw_gru_parameters(layer, rng)
    X_np = rng.normal(size=(*batch_shape, 7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_gru(X_np, W_ih, b, W_hh, c, np.tanh), atol=ATOL
    )


def test_the_gru_state_takes_the_dtype_the_step_produces():
    """A float32 cell fed a float64 sequence. The cell has to hand the state builder every parameter its
    step touches; naming too few, or pinning the state to floatX, leaves scan comparing float32 against
    the float64 the step returns and refusing the graph. Every other test here runs at one dtype."""
    with pytensor.config.change_flags(floatX="float32"):
        layer = GRU("gru", n_in=4, n_hidden=3)
        X = pt.tensor("X", shape=(None, None, 4), dtype="float64")
        out = layer(X)

        assert out.dtype == "float64"
        assert out.eval({X: np.zeros((2, 5, 4), dtype="float64")}).dtype == np.dtype("float64")


def unrolled_lstm(X_np, W_ih, b, W_hh, phi, gate=sigmoid):
    """The gated recurrence written as a python loop, as the reference to check the scan against."""
    n_hidden = W_hh.shape[0]
    h = np.zeros((*X_np.shape[:-2], n_hidden), dtype=floatX)
    c = np.zeros_like(h)
    states = []
    for t in range(X_np.shape[-2]):
        projected = X_np[..., t, :] @ W_ih + h @ W_hh + b
        pre_in, pre_forget, pre_candidate, pre_out = (
            projected[..., i * n_hidden : (i + 1) * n_hidden] for i in range(4)
        )
        c = gate(pre_forget) * c + gate(pre_in) * phi(pre_candidate)
        h = gate(pre_out) * phi(c)
        states.append(h)
    return np.stack(states, axis=-2)


def draw_lstm_parameters(layer, rng):
    """Set every parameter to a fresh draw and hand the values back for the reference to use."""
    n_in, n_hidden = layer.cell.n_in, layer.cell.n_hidden
    W_ih = rng.normal(size=(n_in, 4 * n_hidden)).astype(floatX)
    W_hh = rng.normal(size=(n_hidden, 4 * n_hidden)).astype(floatX)
    b = rng.normal(size=(4 * n_hidden,)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.W_hh.set_value(W_hh)
    layer.cell.b.set_value(b)
    return W_ih, b, W_hh


@pytest.mark.parametrize(
    "activation, reference",
    [(Tanh(), np.tanh), (ReLU(), lambda x: np.maximum(x, 0.0))],
    ids=["tanh", "relu"],
)
def test_the_lstm_matches_a_step_by_step_reference(activation, reference, rng):
    """The relu case is what pins ``activation`` being applied twice: once to the candidate and again
    to the memory on the way out. At tanh alone, hardcoding either one would still agree."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = LSTM("lstm", n_in=4, n_hidden=3, activation=activation)
    out = layer(X)
    assert out.type.shape == (None, None, 3)

    W_ih, b, W_hh = draw_lstm_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_lstm(X_np, W_ih, b, W_hh, reference), atol=ATOL
    )


def test_the_lstm_gate_slices_do_not_cross(rng):
    """Four gates read four slices of one projection, and swapping two still produces a plausible
    sequence. Driving each to its own extreme pins which slice is which; the reference loop alone would
    agree with any consistent misordering of the parameter layout."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = LSTM("lstm", n_in=4, n_hidden=3)
    out = layer(X)

    layer.cell.W_ih.set_value(np.zeros((4, 12), dtype=floatX))
    layer.cell.W_hh.set_value(np.zeros((3, 12), dtype=floatX))
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    shut, opened = -20.0, 20.0

    def biases(gate_in, gate_forget, candidate, gate_out):
        return np.repeat([gate_in, gate_forget, candidate, gate_out], 3).astype(floatX)

    # The input gate open onto a saturated candidate writes tanh(20) into the memory, and the output
    # gate open exposes tanh of that. The forget gate is irrelevant while the memory starts at zero.
    layer.cell.b.set_value(biases(opened, shut, opened, opened))
    np.testing.assert_allclose(
        out.eval({X: X_np}), np.full((5, 7, 3), np.tanh(np.tanh(20.0)), dtype=floatX), atol=1e-6
    )

    # The output gate shut hides that same memory, so nothing reaches h however full the memory is.
    layer.cell.b.set_value(biases(opened, shut, opened, shut))
    np.testing.assert_allclose(out.eval({X: X_np}), np.zeros((5, 7, 3)), atol=1e-6)

    # The input gate shut writes nothing, so an open output gate exposes an empty memory.
    layer.cell.b.set_value(biases(shut, shut, opened, opened))
    np.testing.assert_allclose(out.eval({X: X_np}), np.zeros((5, 7, 3)), atol=1e-6)


def test_the_lstm_forget_gate_decays_the_memory_over_the_sequence(rng):
    """The memory is the state the layer exists for, and it is the one the output never shows directly.
    Writing it once and then shutting the input gate leaves the forget gate alone with it: held open the
    memory survives every later step, held shut it is gone by the next one."""
    X = pt.tensor("X", shape=(None, None, 4))
    h0 = pt.tensor("h0", shape=(None, 3))
    c0 = pt.tensor("c0", shape=(None, 3))
    layer = LSTM("lstm", n_in=4, n_hidden=3)
    out = layer(X, [h0, c0])

    layer.cell.W_ih.set_value(np.zeros((4, 12), dtype=floatX))
    layer.cell.W_hh.set_value(np.zeros((3, 12), dtype=floatX))
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    h0_np = np.zeros((5, 3), dtype=floatX)
    c0_np = rng.normal(size=(5, 3)).astype(floatX)
    # Input gate shut, output gate open: h is tanh of whatever the memory still holds.
    remembering = np.repeat([-20.0, 20.0, 0.0, 20.0], 3).astype(floatX)

    layer.cell.b.set_value(remembering)
    held = out.eval({X: X_np, h0: h0_np, c0: c0_np})
    np.testing.assert_allclose(
        held, np.broadcast_to(np.tanh(c0_np)[:, None, :], (5, 7, 3)), atol=1e-5
    )

    forgetting = remembering.copy()
    forgetting[3:6] = -20.0
    layer.cell.b.set_value(forgetting)
    np.testing.assert_allclose(
        out.eval({X: X_np, h0: h0_np, c0: c0_np}), np.zeros((5, 7, 3)), atol=1e-6
    )


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
def test_the_lstm_bias_is_optional(bias, rng):
    """Dropping the bias drops the parameter as well as the term; an unused one left behind would hand
    the optimizer moment state to carry for a weight that never moves."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = LSTM("lstm", n_in=4, n_hidden=3, bias=bias)
    out = layer(X)

    W_ih = rng.normal(size=(4, 12)).astype(floatX)
    W_hh = rng.normal(size=(3, 12)).astype(floatX)
    layer.cell.W_ih.set_value(W_ih)
    layer.cell.W_hh.set_value(W_hh)
    b = np.zeros(12, dtype=floatX)
    if bias:
        b = rng.normal(size=(12,)).astype(floatX)
        layer.cell.b.set_value(b)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    assert set(collect_trainable_params(out)) == (
        {layer.cell.W_ih, layer.cell.W_hh, layer.cell.b}
        if bias
        else {layer.cell.W_ih, layer.cell.W_hh}
    )
    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_lstm(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL
    )


def test_the_lstm_gates_take_their_own_activation(rng):
    """The gates and the candidate have separate keywords, and setting one must leave the other alone."""

    class HardSigmoid(Activation):
        def __call__(self, x):
            return pt.clip(x * 0.2 + 0.5, 0.0, 1.0)

    def hard_sigmoid(x):
        return np.clip(x * 0.2 + 0.5, 0.0, 1.0)

    X = pt.tensor("X", shape=(None, None, 4))
    layer = LSTM("lstm", n_in=4, n_hidden=3, gate_activation=HardSigmoid())
    out = layer(X)

    W_ih, b, W_hh = draw_lstm_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    hard = out.eval({X: X_np})

    np.testing.assert_allclose(
        hard, unrolled_lstm(X_np, W_ih, b, W_hh, np.tanh, gate=hard_sigmoid), atol=ATOL
    )
    logistic = unrolled_lstm(X_np, W_ih, b, W_hh, np.tanh)
    assert np.abs(hard - logistic).max() > 0.01


def test_the_lstm_forwards_its_initializers_to_the_cell():
    """Four keyword-only arguments reach the cell through the flat constructor, and one dropped on the
    floor leaves a parameter silently at its default draw."""
    layer = LSTM(
        "lstm",
        n_in=4,
        n_hidden=3,
        recurrent_initializer=ZeroInitializer(),
        bias_initializer=OneInitializer(),
    )

    np.testing.assert_array_equal(layer.cell.W_hh.get_value(), np.zeros((3, 12)))
    np.testing.assert_array_equal(layer.cell.b.get_value(), np.ones(12))
    assert np.abs(layer.cell.W_ih.get_value()).max() > 0.0


def test_the_lstm_recurrent_weight_is_drawn_orthogonal_by_default():
    """One draw covers all four gates, so the check is on the whole wide matrix rather than per gate."""
    layer = LSTM("lstm", n_in=16, n_hidden=32)

    W_hh = layer.cell.W_hh.get_value()
    assert W_hh.shape == (32, 128)
    np.testing.assert_allclose(W_hh @ W_hh.T, np.eye(32), atol=ATOL)

    W_ih = layer.cell.W_ih.get_value()
    assert np.abs(W_ih @ W_ih.T - np.eye(16)).max() > 0.1
    assert W_ih.std() == pytest.approx(np.sqrt(2.0 / (16 + 128)), rel=0.1)


def test_the_lstm_trains_end_to_end(rng):
    """Gradients survive the round trip through both carried states and the training machinery moves
    them. Nothing else here differentiates a cell whose two states feed each other."""
    X = Input("X", shape=(None, 6, 4))
    y = Linear("head", n_in=5, n_out=1)(LSTM("lstm", n_in=4, n_hidden=5)(X)[..., -1, :])
    model = Model(y).initialize(seed=1)
    step = model.compile_train(adam(learning_rate=0.05), SquaredError())

    X_np = rng.normal(size=(32, 6, 4)).astype(floatX)
    y_np = X_np.sum(axis=(1, 2))[:, None].astype(floatX)

    losses = [float(step(X_np, y_np)) for _ in range(50)]
    assert losses[-1] < losses[0] / 5


@pytest.mark.parametrize(
    "batch_shape", [(), (5,), (2, 5)], ids=["unbatched", "one_axis", "two_axes"]
)
def test_the_lstm_recurs_over_any_number_of_batch_axes(batch_shape, rng):
    """Both carried states take their batch axes from the input, and the memory has to keep them across
    the step that combines it with the gates."""
    X = pt.tensor("X", shape=(*(None for _ in batch_shape), None, 4))
    layer = LSTM("lstm", n_in=4, n_hidden=3)
    out = layer(X)

    W_ih, b, W_hh = draw_lstm_parameters(layer, rng)
    X_np = rng.normal(size=(*batch_shape, 7, 4)).astype(floatX)

    np.testing.assert_allclose(
        out.eval({X: X_np}), unrolled_lstm(X_np, W_ih, b, W_hh, np.tanh), atol=ATOL
    )


def test_the_lstm_state_takes_the_dtype_the_step_produces():
    """A float32 cell fed a float64 sequence. Both carried states go through the one builder, so a state
    pinned to floatX leaves scan comparing float32 against the float64 the step returns."""
    with pytensor.config.change_flags(floatX="float32"):
        layer = LSTM("lstm", n_in=4, n_hidden=3)
        X = pt.tensor("X", shape=(None, None, 4), dtype="float64")
        out = layer(X)

        assert out.dtype == "float64"
        assert out.eval({X: np.zeros((2, 5, 4), dtype="float64")}).dtype == np.dtype("float64")


def test_the_lstm_memory_carries_gradient_across_a_long_sequence(rng):
    """What the memory is for. With the forget gate open it reaches the last step untouched by any
    weight, so the gradient back to the starting memory survives fifty steps; with the gate shut the
    same path is cut and the gradient is gone. An Elman state, multiplied by a weight every step,
    has no setting that does the first."""
    X = pt.tensor("X", shape=(None, None, 4))
    h0 = pt.tensor("h0", shape=(None, 3))
    c0 = pt.tensor("c0", shape=(None, 3))
    layer = LSTM("lstm", n_in=4, n_hidden=3)
    out = layer(X, [h0, c0])
    sensitivity = pt.grad(out[..., -1, :].sum(), c0)

    layer.cell.W_ih.set_value(np.zeros((4, 12), dtype=floatX))
    layer.cell.W_hh.set_value(np.zeros((3, 12), dtype=floatX))
    X_np = rng.normal(size=(5, 50, 4)).astype(floatX)
    h0_np = np.zeros((5, 3), dtype=floatX)
    c0_np = rng.normal(size=(5, 3)).astype(floatX)

    # Input gate shut so nothing is written, output gate open so the memory reaches h.
    layer.cell.b.set_value(np.repeat([-20.0, 20.0, 0.0, 20.0], 3).astype(floatX))
    remembered = sensitivity.eval({X: X_np, h0: h0_np, c0: c0_np})
    # d tanh(c_0) / d c_0, undiminished by the fifty steps in between.
    np.testing.assert_allclose(remembered, 1.0 - np.tanh(c0_np) ** 2, atol=1e-5)

    layer.cell.b.set_value(np.repeat([-20.0, -20.0, 0.0, 20.0], 3).astype(floatX))
    forgotten = sensitivity.eval({X: X_np, h0: h0_np, c0: c0_np})
    assert np.abs(forgotten).max() < 1e-6


def test_a_reversed_layer_reads_the_sequence_from_the_end(rng):
    """Running backward has to be exactly running forward over the flipped sequence, step for step.
    Anything that merely reordered the output would agree with a forward pass on a palindrome and on
    nothing else, so the input here is drawn."""
    X = pt.tensor("X", shape=(None, None, 4))
    forward = RNN("rnn", n_in=4, n_hidden=3)
    backward = RNN("rnn", n_in=4, n_hidden=3, reverse=True)

    W_ih, b, W_hh = draw_parameters(forward, rng)
    backward.cell.W_ih.set_value(W_ih)
    backward.cell.b.set_value(b)
    backward.cell.W_hh.set_value(W_hh)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    read_backward = backward(X).eval({X: X_np})

    # The reference reads the flipped sequence forward, then puts the answers back where they came from.
    on_flipped = unrolled(X_np[..., ::-1, :], W_ih, b, W_hh, np.tanh)
    np.testing.assert_allclose(read_backward, on_flipped[..., ::-1, :], atol=ATOL)
    assert np.abs(read_backward - forward(X).eval({X: X_np})).max() > 0.1


def test_a_reversed_layer_stays_aligned_with_the_input(rng):
    """The output keeps the input's time order, so step t is the step that read X[t] in both
    directions. Returning the backward pass last step first -- what keras and flax do by default -- is
    the shape that silently misaligns a bidirectional concatenation."""
    X = pt.tensor("X", shape=(None, None, 4))
    backward = RNN("rnn", n_in=4, n_hidden=3, reverse=True)
    draw_parameters(backward, rng)

    # A backward pass sees only the last step when it starts, so its first output is a function of that
    # step alone. Aligned, that output sits at the end of the sequence.
    X_np = rng.normal(size=(1, 6, 4)).astype(floatX)
    read_backward = backward(X)

    whole = read_backward.eval({X: X_np})
    last_step_alone = read_backward.eval({X: X_np[:, -1:, :]})
    np.testing.assert_allclose(whole[:, -1, :], last_step_alone[:, 0, :], atol=ATOL)


@pytest.mark.parametrize("layer_type", [RNN, GRU, LSTM], ids=["rnn", "gru", "lstm"])
def test_every_recurrent_layer_takes_a_direction(layer_type, rng):
    """``reverse`` lives on the loop, not on the cell, so all three flat constructors have to forward
    it. One that dropped it would quietly run forward. Both passes come off the one layer, so they are
    the same recurrence read two ways rather than two draws that happen to agree."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = layer_type("layer", n_in=4, n_hidden=3, reverse=True)
    assert layer.reverse

    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    backward = layer(X).eval({X: X_np})

    layer.reverse = False
    on_flipped = layer(X).eval({X: X_np[..., ::-1, :]})
    np.testing.assert_allclose(backward, on_flipped[..., ::-1, :], atol=ATOL)
    assert np.abs(backward - layer(X).eval({X: X_np})).max() > 0.1


def test_bidirectional_gives_each_step_both_halves_of_the_sequence(rng):
    """The point of the wrapper: at every step the forward half has read the prefix and the backward
    half the suffix, of the same step. Both halves are checked against the python loop rather than
    against another layer, so a concatenation joining mismatched steps cannot agree with a reference
    carrying the same misalignment."""
    X = pt.tensor("X", shape=(None, None, 4))
    forward = GRU("fwd", n_in=4, n_hidden=3)
    backward = GRU("bwd", n_in=4, n_hidden=5)
    out = Bidirectional(forward, backward)(X)
    assert out.type.shape == (None, None, 8)

    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    both = out.eval({X: X_np})

    def parameters(layer):
        cell = layer.cell
        return cell.W_ih.get_value(), cell.b.get_value(), cell.W_hh.get_value(), cell.c.get_value()

    np.testing.assert_allclose(
        both[..., :3], unrolled_gru(X_np, *parameters(forward), np.tanh), atol=ATOL
    )
    np.testing.assert_allclose(
        both[..., 3:],
        unrolled_gru(X_np[..., ::-1, :], *parameters(backward), np.tanh)[..., ::-1, :],
        atol=ATOL,
    )


def test_bidirectional_owns_the_direction_of_both_layers(rng):
    """A caller who builds both halves the same way, or reverses the wrong one, still gets one pass in
    each direction -- and the layers they handed over keep the direction they were built with, so using
    one on its own afterwards is unaffected."""
    X = pt.tensor("X", shape=(None, None, 4))
    forward = GRU("fwd", n_in=4, n_hidden=3, reverse=True)
    backward = GRU("bwd", n_in=4, n_hidden=3)
    both = Bidirectional(forward, backward)(X)

    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    evaluated = both.eval({X: X_np})

    def parameters(layer):
        cell = layer.cell
        return cell.W_ih.get_value(), cell.b.get_value(), cell.W_hh.get_value(), cell.c.get_value()

    np.testing.assert_allclose(
        evaluated[..., :3], unrolled_gru(X_np, *parameters(forward), np.tanh), atol=ATOL
    )
    np.testing.assert_allclose(
        evaluated[..., 3:],
        unrolled_gru(X_np[..., ::-1, :], *parameters(backward), np.tanh)[..., ::-1, :],
        atol=ATOL,
    )
    assert forward.reverse and not backward.reverse


def test_bidirectional_rejects_one_layer_used_twice():
    """One layer in both slots runs, but with a single set of parameters shared between the directions,
    which is the one thing the two-layer signature exists to prevent."""
    layer = GRU("gru", n_in=4, n_hidden=3)
    with pytest.raises(ValueError, match="its own parameters"):
        Bidirectional(layer, layer)


def test_bidirectional_trains_end_to_end(rng):
    """Both halves have to reach the optimizer; a wrapper that dropped one would still train, on half
    the parameters."""
    X = Input("X", shape=(None, 6, 4))
    both = Bidirectional(GRU("fwd", n_in=4, n_hidden=5), GRU("bwd", n_in=4, n_hidden=5))(X)
    y = Linear("head", n_in=10, n_out=1)(both[..., -1, :])
    model = Model(y).initialize(seed=1)

    assert len(collect_trainable_params(y)) == 10
    step = model.compile_train(adam(learning_rate=0.05), SquaredError())
    X_np = rng.normal(size=(32, 6, 4)).astype(floatX)
    y_np = X_np.sum(axis=(1, 2))[:, None].astype(floatX)

    losses = [float(step(X_np, y_np)) for _ in range(50)]
    assert losses[-1] < losses[0] / 5


def test_a_call_may_override_the_layers_own_direction(rng):
    """``Bidirectional`` asks each half for a direction per call rather than reaching in and setting it,
    so the override has to work both ways round and leave the layer as it found it."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = GRU("gru", n_in=4, n_hidden=3)
    W_ih, b, W_hh, c = draw_gru_parameters(layer, rng)
    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)

    read_forward = unrolled_gru(X_np, W_ih, b, W_hh, c, np.tanh)
    read_backward = unrolled_gru(X_np[..., ::-1, :], W_ih, b, W_hh, c, np.tanh)[..., ::-1, :]

    np.testing.assert_allclose(layer(X, reverse=True).eval({X: X_np}), read_backward, atol=ATOL)
    assert not layer.reverse

    layer.reverse = True
    np.testing.assert_allclose(layer(X, reverse=False).eval({X: X_np}), read_forward, atol=ATOL)
    np.testing.assert_allclose(layer(X).eval({X: X_np}), read_backward, atol=ATOL)
    assert layer.reverse


def pad_to(sequences, padded_length):
    """Stack ragged sequences into a rectangle, with the mask that says where each one ends."""
    batch = len(sequences)
    padded = np.zeros((batch, padded_length, sequences[0].shape[-1]), dtype=floatX)
    mask = np.zeros((batch, padded_length), dtype=bool)
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = sequence
        mask[row, : len(sequence)] = True
    return padded, mask


def test_a_mask_makes_padding_leave_a_backward_pass_alone(rng):
    """The case that needs the mask most. Read backward, the padding is consumed before any real step,
    so it spoils every output position and no indexing afterwards recovers them."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = RNN("rnn", n_in=4, n_hidden=3, reverse=True)
    draw_parameters(layer, rng)

    real = rng.normal(size=(3, 4)).astype(floatX)
    padded, mask_np = pad_to([real], padded_length=6)
    alone = layer(X).eval({X: real[None]})

    np.testing.assert_allclose(
        layer(X, mask=mask).eval({X: padded, mask: mask_np})[:, :3], alone, atol=ATOL
    )
    assert not np.allclose(layer(X).eval({X: padded})[:, :3], alone, atol=ATOL)


def test_a_mask_lets_one_batch_hold_sequences_of_different_lengths(rng):
    """The case padding exists for: every row a different length, run as one rectangle. Each row has to
    match what it would have given on its own."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = GRU("gru", n_in=4, n_hidden=3, reverse=True)
    draw_gru_parameters(layer, rng)

    lengths = [2, 5, 3]
    sequences = [rng.normal(size=(length, 4)).astype(floatX) for length in lengths]
    padded, mask_np = pad_to(sequences, padded_length=max(lengths))
    together = layer(X, mask=mask).eval({X: padded, mask: mask_np})

    for row, (sequence, length) in enumerate(zip(sequences, lengths)):
        alone = layer(X).eval({X: sequence[None]})
        np.testing.assert_allclose(together[row, :length], alone[0], atol=ATOL)


def test_a_mask_holds_every_state_a_cell_carries(rng):
    """An LSTM's memory is masked alongside its output, or a padded step would go on writing to the
    memory that the output gate reads at the next real step."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = LSTM("lstm", n_in=4, n_hidden=3, reverse=True)
    draw_lstm_parameters(layer, rng)

    real = rng.normal(size=(4, 4)).astype(floatX)
    padded, mask_np = pad_to([real], padded_length=9)

    np.testing.assert_allclose(
        layer(X, mask=mask).eval({X: padded, mask: mask_np})[:, :4],
        layer(X).eval({X: real[None]}),
        atol=ATOL,
    )


def test_a_mask_freezes_the_final_state_of_a_padded_forward_pass(rng):
    """A forward pass is causal, so padding leaves the real positions already correct and only the
    final state drifts on. A masked step emits the state the step before it left, which holds that
    final state flat across the padding and makes ``out[..., -1, :]`` true without a per-row gather."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = RNN("rnn", n_in=4, n_hidden=3)
    draw_parameters(layer, rng)

    real = rng.normal(size=(3, 4)).astype(floatX)
    padded, mask_np = pad_to([real], padded_length=7)
    out = layer(X, mask=mask).eval({X: padded, mask: mask_np})

    alone = layer(X).eval({X: real[None]})
    last_real = alone[0, -1]
    np.testing.assert_allclose(out[0, -1], last_real, atol=ATOL)
    np.testing.assert_allclose(out[0, 3:], np.broadcast_to(last_real, (4, 3)), atol=ATOL)

    unmasked = layer(X).eval({X: padded})
    np.testing.assert_allclose(unmasked[:, :3], alone, atol=ATOL)
    assert not np.allclose(unmasked[0, -1], last_real, atol=ATOL)


def test_bidirectional_reads_the_mask_in_both_directions(rng):
    """The wrapper's backward half is the one that needs the mask, and it has to reach both halves from
    the one argument."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    forward = GRU("fwd", n_in=4, n_hidden=3)
    backward = GRU("bwd", n_in=4, n_hidden=5)
    layer = Bidirectional(forward, backward)
    # Drawn, not left at their defaults: a zero bias makes the zero state a fixed point, so the
    # padding would not move the state and the mask would have nothing to undo.
    draw_gru_parameters(forward, rng)
    draw_gru_parameters(backward, rng)

    real = rng.normal(size=(3, 4)).astype(floatX)
    padded, mask_np = pad_to([real], padded_length=8)
    together = layer(X, mask=mask).eval({X: padded, mask: mask_np})

    alone_forward = forward(X).eval({X: real[None]})
    alone_backward = backward(X, reverse=True).eval({X: real[None]})
    np.testing.assert_allclose(together[:, :3, :3], alone_forward, atol=ATOL)
    np.testing.assert_allclose(together[:, :3, 3:], alone_backward, atol=ATOL)


def test_rejects_a_mask_that_does_not_match_the_batch_axes():
    """A mask shaped like the input, feature axis and all, is the natural mistake; scan would take it
    as a sequence and fail somewhere inside the loop."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3)

    with pytest.raises(ValueError, match="needs a 2-dimensional mask; got a 3-dimensional one"):
        layer(X, mask=pt.tensor("mask", shape=(None, None, 4), dtype=bool))


def test_rejects_a_mask_whose_time_axis_is_shorter_than_the_input(rng):
    """Scan takes its step count from the shortest sequence it is handed, so a mask one step short runs
    the whole recurrence one step short -- leaving ``out[..., -1, :]`` an early state rather than the
    last one, which is the failure the mask is there to prevent."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = RNN("rnn", n_in=4, n_hidden=3)
    out = layer(X, mask=mask)
    X_np = rng.normal(size=(2, 6, 4)).astype(floatX)

    assert out.eval({X: X_np, mask: np.ones((2, 6), dtype=bool)}).shape[-2] == 6
    with pytest.raises(AssertionError, match="has shape 5, expected 6"):
        out.eval({X: X_np, mask: np.ones((2, 5), dtype=bool)})


def test_a_mask_keeps_padding_out_of_the_gradient(rng):
    """Training on a padded batch has to move the parameters exactly as training on the sequences alone
    would. The masked step still computes -- the switch only discards its result -- so a gradient that
    leaked through the discarded branch would make the padding length a hyperparameter. Read backward,
    because a forward pass reaches the real steps before any padding and would agree either way."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = RNN("rnn", n_in=4, n_hidden=3, reverse=True)
    draw_parameters(layer, rng)
    parameters = [layer.cell.W_ih, layer.cell.W_hh, layer.cell.b]

    real = rng.normal(size=(3, 4)).astype(floatX)
    padded, mask_np = pad_to([real], padded_length=7)

    over_padded = pt.grad(layer(X, mask=mask)[:, :3].sum(), parameters)
    over_real = pt.grad(layer(X).sum(), parameters)

    for padded_gradient, real_gradient in zip(over_padded, over_real):
        np.testing.assert_allclose(
            padded_gradient.eval({X: padded, mask: mask_np}),
            real_gradient.eval({X: real[None]}),
            atol=ATOL,
        )


def test_a_mask_may_skip_a_step_in_the_middle(rng):
    """A mask says which steps count, not how many, so it can drop one from the middle -- which a
    per-example length cannot express. The recurrence carries on as if that step were not there."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = RNN("rnn", n_in=4, n_hidden=3)
    draw_parameters(layer, rng)

    full = rng.normal(size=(5, 4)).astype(floatX)
    skipped = 2
    mask_np = np.ones((1, 5), dtype=bool)
    mask_np[0, skipped] = False

    with_gap = layer(X, mask=mask).eval({X: full[None], mask: mask_np})
    without = layer(X).eval({X: np.delete(full, skipped, axis=0)[None]})

    np.testing.assert_allclose(with_gap[:, :skipped], without[:, :skipped], atol=ATOL)
    np.testing.assert_allclose(with_gap[:, skipped + 1 :], without[:, skipped:], atol=ATOL)


class MatrixMemoryCell(RecurrentCell):
    """A cell whose state carries two feature axes, as a matrix-memory recurrence does. It sums the
    step's input into every entry, so a step that ran shows up everywhere in the state."""

    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols

    def step(self, x_t, *state):
        (memory,) = state
        return (memory + x_t.sum(axis=-1)[..., None, None],)

    def initial_state(self, X):
        return (pt.zeros((*X.shape[:-2], self.rows, self.cols), dtype=X.dtype),)


def test_a_state_may_carry_more_than_one_feature_axis(rng):
    """Nothing here constrains a carried state to a single feature axis, so time has to land directly
    after the input's batch axes rather than second-to-last -- those differ the moment a cell carries a
    matrix. The output is the running sum of each step's input, one entry per state cell."""
    X = pt.tensor("X", shape=(None, None, 4))
    out = Recurrent(MatrixMemoryCell(2, 3), name="matrix")(X)

    X_np = rng.normal(size=(5, 7, 4)).astype(floatX)
    evaluated = out.eval({X: X_np})

    assert evaluated.shape == (5, 7, 2, 3)
    running = np.cumsum(X_np.sum(axis=-1), axis=-1)
    np.testing.assert_allclose(
        evaluated, np.broadcast_to(running[:, :, None, None], (5, 7, 2, 3)), atol=ATOL
    )


def test_a_mask_holds_a_state_of_any_rank(rng):
    """The mask names batch elements and the state adds however many feature axes the cell wants, so
    the two are lined up by the state's rank rather than by assuming exactly one feature axis."""
    X = pt.tensor("X", shape=(None, None, 4))
    mask = pt.tensor("mask", shape=(None, None), dtype=bool)
    layer = Recurrent(MatrixMemoryCell(2, 3), name="matrix")

    real = rng.normal(size=(3, 4)).astype(floatX)
    padded, mask_np = pad_to([real], padded_length=6)

    np.testing.assert_allclose(
        layer(X, mask=mask).eval({X: padded, mask: mask_np})[:, :3],
        layer(X).eval({X: real[None]}),
        atol=ATOL,
    )
