import numpy as np
import pytensor
import pytensor.tensor as pt
import pytensor.tensor.random as ptr
import pytest

from pytensor.scan import scan, until
from pytensor.scan.op import Scan
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.variable import shared_rng

from pytensor_ml.activations import Activation, Tanh
from pytensor_ml.layers import RNN, BatchNorm, Dropout, Input, Linear, Recurrent, RecurrentCell
from pytensor_ml.loss import SquaredError
from pytensor_ml.model import Model
from pytensor_ml.optim import adam
from pytensor_ml.pytensorf import collect_trainable_params, function
from pytensor_ml.pytensorf.rewrite import carry_scan_statistics, hoist_scan_draws
from pytensor_ml.rewriting.scan import carry_statistics_through_scan, hoist_draws_out_of_scan

floatX = pytensor.config.floatX


class TanhDropout(Activation):
    """Dropout written the obvious way inside a step: no generator threading, no recorded outputs.

    Seeded, because the layer draws from fresh entropy when told nothing and every assertion here is
    about which values come out.
    """

    def __init__(self, p=0.5, random_state=0):
        self.tanh = Tanh()
        self.dropout = Dropout("recurrent_dropout", p=p, random_state=random_state)

    def __call__(self, x):
        return self.dropout(self.tanh(x))


def scan_nodes(graph):
    return [
        node
        for node in pytensor.graph.FunctionGraph(outputs=[graph], clone=False).apply_nodes
        if isinstance(node.op, Scan)
    ]


def draws_inside(graph):
    """The draws still living inside every scan reachable from ``graph``, however deeply nested.

    Recursive because a scan's inner graph may hold another scan, and stopping at the first level hides a
    draw the rewrite never reached.
    """
    found = []

    def walk(nodes):
        for node in nodes:
            if isinstance(node.op, Scan):
                inner = node.op.fgraph.toposort()
                found.extend(one for one in inner if isinstance(one.op, RandomVariable))
                walk(inner)

    walk(pytensor.graph.FunctionGraph(outputs=[graph], clone=False).apply_nodes)
    return found


def test_a_draw_written_inside_a_recurrence_is_lifted_out():
    """The draw reads a generator the loop cannot advance, so left alone it yields one value reused at
    every step. Lifting it out is what turns it into one independent value per step."""
    X = pt.tensor("X", shape=(None, None, 4))
    out = RNN("rnn", n_in=4, n_hidden=3, activation=TanhDropout())(X)

    assert len(draws_inside(out)) == 1

    [hoisted] = hoist_scan_draws([out])

    assert draws_inside(hoisted) == []
    assert not any(
        isinstance(inp.type, pytensor.tensor.random.type.RandomType)
        for node in scan_nodes(hoisted)
        for inp in node.inputs
    )


def test_each_step_draws_its_own_value():
    """The behavior the lift exists for. Before it the graph does not compile at all; the failure it
    replaces is one mask reused for the whole sequence and every call after it."""
    X = pt.tensor("X", shape=(None, None, 4))
    fn = function([X], RNN("rnn", n_in=4, n_hidden=32, activation=TanhDropout(p=0.5))(X))

    X_np = np.random.default_rng(0).normal(size=(1, 6, 4)).astype(floatX)
    first, second = fn(X_np), fn(X_np)
    zeroed_per_step = {tuple(np.flatnonzero(step == 0)) for step in first[0]}

    assert len(zeroed_per_step) == 6
    assert not np.allclose(first, second)


def test_the_lifted_graph_has_the_gradient_finite_differences_give():
    """The point of lifting rather than working around the error. A draw inside the differentiated region
    leaves no fixed sample to take a gradient against; once it is outside, the loop is a deterministic
    function of the draws and its gradient is the one finite differences find."""
    X = pt.tensor("X", shape=(None, None, 4))
    layer = RNN("rnn", n_in=4, n_hidden=3, activation=TanhDropout(p=0.5))
    [hoisted] = hoist_scan_draws([layer(X)])

    cost = (hoisted**2).sum()
    cost_fn = pytensor.function([X], cost)
    grad_fn = pytensor.function([X], pytensor.grad(cost, layer.cell.W_hh))
    [generator] = layer.cell.activation.dropout.generators
    W, W0 = layer.cell.W_hh, layer.cell.W_hh.get_value()
    X_np = np.random.default_rng(3).normal(size=(2, 6, 4)).astype(floatX)

    def pinned(fn):
        # Reset the generator so the graph is a deterministic function of the weight.
        generator.set_value(np.random.default_rng(0))
        return fn(X_np)

    analytic, epsilon = pinned(grad_fn), 1e-6
    numeric = np.zeros_like(W0)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            for sign in (1, -1):
                perturbed = W0.copy()
                perturbed[i, j] += sign * epsilon
                W.set_value(perturbed)
                numeric[i, j] += sign * pinned(cost_fn)
            numeric[i, j] /= 2 * epsilon
    W.set_value(W0)

    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_a_recurrence_that_draws_trains_end_to_end():
    X = Input("X", shape=(None, 6, 4))
    cell = RNN("rnn", n_in=4, n_hidden=8, activation=TanhDropout(p=0.2))
    y = Linear("head", n_in=8, n_out=1)(cell(X)[..., -1, :])
    model = Model(y).initialize(seed=1)
    step = model.compile_train(adam(learning_rate=0.05), SquaredError())

    rng = np.random.default_rng(0)
    X_np = rng.normal(size=(32, 6, 4)).astype(floatX)
    y_np = X_np.sum(axis=(1, 2))[:, None].astype(floatX)

    losses = [float(step(X_np, y_np)) for _ in range(60)]

    # Averaged over a window: dropout makes any single step's loss noisy enough that comparing the last
    # one against the first measures the draw as much as the training.
    assert np.mean(losses[-10:]) < np.mean(losses[:10]) / 3


def test_a_loop_that_draws_nothing_is_left_alone():
    """The rewrite has to decline cleanly, or every scan in every graph would be rebuilt for nothing."""
    X = pt.tensor("X", shape=(None, None, 4))
    plain = RNN("rnn", n_in=4, n_hidden=3)(X)
    fgraph = pytensor.graph.FunctionGraph(outputs=[plain], clone=False)
    [node] = scan_nodes(plain)

    assert hoist_draws_out_of_scan.transform(fgraph, node) is None


def test_a_generator_the_cell_threads_itself_is_left_alone():
    """A generator carried as recurrent state is threaded on purpose and the loop does advance it, so
    there is no frozen draw to rescue. Lifting it would silently replace the caller's scheme with another."""

    class ThreadedCell(RecurrentCell):
        def __init__(self, n_hidden):
            self.n_hidden = n_hidden
            self.rng = shared_rng(np.random.default_rng(0), name="threaded/rng")

        def step(self, x_t, h_prev, rng):
            next_rng, mask = ptr.bernoulli(p=0.5, size=h_prev.shape, rng=rng, return_next_rng=True)
            return h_prev + mask.astype(h_prev.dtype), next_rng

        def initial_state(self, X):
            zeros = pt.zeros((*X.shape[:-2], self.n_hidden), dtype=floatX)
            return zeros, self.rng

    out = Recurrent(ThreadedCell(3), name="threaded")(pt.tensor("X", shape=(None, None, 4)))

    [unchanged] = hoist_scan_draws([out])

    assert len(draws_inside(unchanged)) == 1


def test_two_draws_sharing_one_generator_are_left_alone():
    """They would need a generator each to come apart, and inventing one would change which values each
    draw gets. The compile-time check that nothing advances the generator still reports it."""
    rng = shared_rng(np.random.default_rng(0), name="shared/rng")
    X = pt.tensor("X", shape=(None, None, 4))
    h0 = pt.zeros((X.shape[0], 4), dtype=floatX)

    def step(x_t, h):
        _, first = ptr.bernoulli(p=0.5, size=h.shape, rng=rng, return_next_rng=True)
        _, second = ptr.bernoulli(p=0.5, size=h.shape, rng=rng, return_next_rng=True)
        first, second = first.astype(floatX), second.astype(floatX)
        return h + x_t * first + second

    out = scan(step, sequences=[pt.moveaxis(X, -2, 0)], outputs_info=[h0], return_updates=False)

    [unchanged] = hoist_scan_draws([out])

    assert len(draws_inside(unchanged)) == 2
    with pytest.raises(ValueError, match="No update found for at least one RNG used in Scan"):
        function([X], unchanged)


def test_a_while_loop_is_left_alone():
    """A while loop's condition is an output of its own, and the rewrite substitutes only the output
    fields it knows. Rewriting one would leave the condition referencing a generator that had just left
    the loop's inputs, so it declines instead."""
    rng = shared_rng(np.random.default_rng(0), name="while/rng")

    def step(total):
        _, draw = ptr.normal(size=(), rng=rng, return_next_rng=True)
        running = total + abs(draw)
        return running, until(running > 3.0)

    trace = scan(step, outputs_info=[pt.zeros(())], n_steps=20, return_updates=False)
    fgraph = pytensor.graph.FunctionGraph(outputs=[trace], clone=False)
    [node] = scan_nodes(trace)

    assert node.op.info.as_while
    assert hoist_draws_out_of_scan.transform(fgraph, node) is None


def test_a_draw_whose_parameters_move_with_the_state_is_left_alone():
    """Its values depend on where the loop has got to, so a sequence of them cannot be drawn before the
    loop runs. Hoisting anyway would freeze `loc` at the initial state and quietly change the answer --
    the shape may depend on the state, since every inner variable holds one shape at every step, but a
    parameter may not."""
    rng = shared_rng(np.random.default_rng(0), name="moving/rng")
    X = pt.vector("X", dtype=floatX)

    def step(x_t, h):
        _, drawn = ptr.normal(loc=2.0 * h, scale=1e-12, size=(), rng=rng, return_next_rng=True)
        return drawn

    doubling = scan(
        step, sequences=[X], outputs_info=[pt.ones((), dtype=floatX)], return_updates=False
    )
    fgraph = pytensor.graph.FunctionGraph(outputs=[doubling], clone=False)
    [node] = scan_nodes(doubling)

    assert hoist_draws_out_of_scan.transform(fgraph, node) is None
    # The state doubles each step, which only holds if `loc` is read afresh at every one.
    np.testing.assert_allclose(
        pytensor.function([X], doubling)(np.zeros(5, dtype=floatX)), [2.0, 4.0, 8.0, 16.0, 32.0]
    )


@pytest.mark.parametrize(
    "loc, per_step_shape",
    [(0.0, ()), (np.zeros(3), (3,))],
    ids=["scalar", "vector_parameter"],
)
def test_a_draw_with_no_explicit_size_takes_its_shape_from_its_parameters(loc, per_step_shape):
    """With no size the shape follows from the parameters, and those are loop-invariant or the draw would
    not be lifted at all -- so the op's own shape inference gives the per-step shape to prepend the step
    count to. Reading it off the draw's output instead builds a second draw from the same generator."""
    rng = shared_rng(np.random.default_rng(0), name="sizeless/rng")
    X = pt.vector("X", dtype=floatX)
    start = pt.zeros(per_step_shape, dtype=floatX)

    def step(x_t, h):
        _, drawn = ptr.normal(loc=loc, scale=1.0, rng=rng, return_next_rng=True)
        return h + drawn

    accumulated = scan(step, sequences=[X], outputs_info=[start], return_updates=False)
    fn = function([X], accumulated)

    running = fn(np.zeros(5, dtype=floatX))
    per_step = np.diff(np.concatenate([np.zeros((1, *per_step_shape)), running]), axis=0)

    assert running.shape == (5, *per_step_shape)
    assert len({tuple(np.atleast_1d(step_draw).round(12)) for step_draw in per_step}) == 5
    assert not np.allclose(running, fn(np.zeros(5, dtype=floatX)))


def test_several_draws_each_with_a_generator_of_its_own_are_all_lifted():
    """Several draws are lifted in one pass, each taking its own generator out of the loop's inputs, and
    what is left has to be a loop that still runs, still advances, and still differentiates."""
    X = pt.matrix("X", dtype=floatX)
    first_rng = shared_rng(np.random.default_rng(0), name="first/rng")
    second_rng = shared_rng(np.random.default_rng(1), name="second/rng")

    def step(x_t, h):
        _, first = ptr.bernoulli(p=0.5, size=(3,), rng=first_rng, return_next_rng=True)
        _, second = ptr.bernoulli(p=0.5, size=(3,), rng=second_rng, return_next_rng=True)
        return h + x_t * first.astype(h.dtype) + second.astype(h.dtype)

    out = scan(
        step,
        sequences=[X],
        outputs_info=[pt.zeros((3,), dtype=floatX)],
        return_updates=False,
    )
    assert len(draws_inside(out)) == 2

    [hoisted] = hoist_scan_draws([out])
    assert draws_inside(hoisted) == []

    fn = function([X], hoisted)
    X_np = np.ones((6, 3), dtype=floatX)
    assert fn(X_np).shape == (6, 3)
    assert not np.allclose(fn(X_np), fn(X_np))
    assert pytensor.grad(hoisted.sum(), X) is not None


def test_a_multi_tap_recurrence_is_lifted():
    """A mit-sot's states reach the inner graph as a nested field and its outer buffer holds several taps,
    so a shape expression built from one has to be mapped back through a different field than a sit-sot's."""
    X = pt.matrix("X", dtype=floatX)
    rng = shared_rng(np.random.default_rng(0), name="mit/rng")

    def step(x_t, h_previous, h_before_that):
        _, mask = ptr.bernoulli(p=0.5, size=h_previous.shape, rng=rng, return_next_rng=True)
        return (h_previous + h_before_that) * 0.5 + x_t * mask.astype(h_previous.dtype)

    out = scan(
        step,
        sequences=[X],
        outputs_info=[{"initial": pt.zeros((2, 3), dtype=floatX), "taps": [-1, -2]}],
        return_updates=False,
    )
    [node] = scan_nodes(out)
    assert node.op.info.mit_sot_in_slices, "the point of this test is the mit-sot path"

    [hoisted] = hoist_scan_draws([out])
    assert draws_inside(hoisted) == []

    fn = function([X], hoisted)
    X_np = np.ones((6, 3), dtype=floatX)
    assert fn(X_np).shape == (6, 3)
    assert not np.allclose(fn(X_np), fn(X_np))
    assert pytensor.grad(hoisted.sum(), X) is not None


def test_a_shape_taken_from_a_later_tap_is_still_mapped_outward():
    """A multi-tap state reaches the inner graph once per tap but the outer graph once in total, so a
    mapping that pairs them off one-to-one covers only the first tap. A shape read from any other one
    then survives into the lifted expression as a variable the outer graph has never heard of."""
    X = pt.matrix("X", dtype=floatX)
    rng = shared_rng(np.random.default_rng(0), name="later_tap/rng")

    def step(x_t, h_previous, h_before_that):
        _, mask = ptr.bernoulli(p=0.5, size=h_before_that.shape, rng=rng, return_next_rng=True)
        return (h_previous + h_before_that) * 0.5 + x_t * mask.astype(h_previous.dtype)

    out = scan(
        step,
        sequences=[X],
        outputs_info=[{"initial": pt.zeros((2, 3), dtype=floatX), "taps": [-1, -2]}],
        return_updates=False,
    )

    [hoisted] = hoist_scan_draws([out])
    assert draws_inside(hoisted) == []
    assert function([X], hoisted)(np.ones((5, 3), dtype=floatX)).shape == (5, 3)


def test_two_recurrences_in_one_graph_are_both_lifted():
    """Stacking is the ordinary way to build a deep RNN, and each layer brings a generator of its own."""
    X = pt.tensor("X", shape=(None, None, 4))
    first = RNN("rnn1", n_in=4, n_hidden=5, activation=TanhDropout(p=0.5, random_state=0))(X)
    stacked = RNN("rnn2", n_in=5, n_hidden=3, activation=TanhDropout(p=0.5, random_state=1))(first)

    assert len(draws_inside(stacked)) == 2

    fn = function([X], stacked)
    X_np = np.random.default_rng(0).normal(size=(2, 6, 4)).astype(floatX)

    assert fn(X_np).shape == (2, 6, 3)
    assert not np.allclose(fn(X_np), fn(X_np))

    # On the graph as written the gradient is undefined, which is what the lift is for.
    [hoisted] = hoist_scan_draws([stacked])
    assert pytensor.grad(hoisted.sum(), X) is not None
    with pytest.raises(Exception):
        pytensor.grad(stacked.sum(), X)


def test_a_draw_inside_a_nested_loop_is_not_reached():
    """A known limit, pinned so that making it work is a deliberate change rather than a surprise. The
    rewriter walks the outer graph's scan nodes and does not descend into a scan living inside another
    scan's inner graph, so the draw stays put and compiling still refuses it -- loudly, not wrongly."""
    X = pt.matrix("X", dtype=floatX)
    rng = shared_rng(np.random.default_rng(0), name="nested/rng")

    def inner_step(row, carried):
        _, mask = ptr.bernoulli(p=0.5, size=(3,), rng=rng, return_next_rng=True)
        return carried + row * mask.astype(carried.dtype)

    def outer_step(x_t, h):
        return scan(
            inner_step,
            sequences=[pt.tile(x_t, (2, 1))],
            outputs_info=[h],
            return_updates=False,
        )[-1]

    nested = scan(
        outer_step, sequences=[X], outputs_info=[pt.zeros((3,), dtype=floatX)], return_updates=False
    )

    [hoisted] = hoist_scan_draws([nested])
    assert len(draws_inside(hoisted)) == 1

    with pytest.raises(ValueError, match="No update found for at least one RNG used in Scan"):
        function([X], hoisted)


def test_two_graphs_over_one_loop_are_lifted_once():
    """`hoist_scan_draws` takes the outputs and the updates together so a loop they share stays shared.
    Lifting each separately would give them a scan apiece, and with it a second draw off the generator --
    which then has two readers and no single next state, so nothing would compile."""
    X = pt.tensor("X", shape=(None, None, 4))
    recurrence = RNN("rnn", n_in=4, n_hidden=3, activation=TanhDropout(p=0.5))(X)

    first, second = hoist_scan_draws([recurrence.sum(), recurrence[..., -1, :]])

    assert draws_inside(first) == []
    [shared_by_first] = scan_nodes(first)
    assert shared_by_first in scan_nodes(second)
    assert function([X], [first, second])(np.ones((2, 6, 4), dtype=floatX))[1].shape == (2, 3)


def chained_statistics(X, momentum, epsilon=None):
    """The running mean and variance a step-by-step loop leaves behind, computed with numpy."""
    running_mean, running_var = np.zeros(X.shape[-1]), np.ones(X.shape[-1])
    for step in range(X.shape[0]):
        running_mean = momentum * X[step].mean(axis=0) + (1 - momentum) * running_mean
        running_var = momentum * X[step].var(axis=0) + (1 - momentum) * running_var
    return running_mean, running_var


def test_a_statistic_written_inside_a_recurrence_is_carried_through_it():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=50.0, scale=3.0, size=(6, 8, 4)).astype(floatX)
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    batch_norm = BatchNorm("bn", n_in=4, momentum=0.1)
    normalized = scan(lambda x_t: batch_norm(x_t), sequences=[xseq], return_updates=False)
    loss = (normalized**2).sum()

    step = function([xseq], loss, updates=adam(1e-3)(loss, collect_trainable_params(loss)))
    step(X)

    # The statistics are of the loop's input, which the affine parameters do not touch, so one training
    # step leaves exactly what stepping through the sequence by hand would.
    expected_mean, expected_var = chained_statistics(X, batch_norm.momentum)
    np.testing.assert_allclose(batch_norm.running_mean.get_value(), expected_mean, rtol=1e-5)
    np.testing.assert_allclose(batch_norm.running_var.get_value(), expected_var, rtol=1e-5)


def test_carrying_a_statistic_leaves_the_loop_output_alone():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 8, 4)).astype(floatX)
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    batch_norm = BatchNorm("bn", n_in=4)
    normalized = scan(lambda x_t: batch_norm(x_t), sequences=[xseq], return_updates=False)

    [carried], _ = carry_scan_statistics([normalized])

    np.testing.assert_allclose(
        pytensor.function([xseq], carried)(X), pytensor.function([xseq], normalized)(X), rtol=1e-6
    )


def test_a_carried_statistic_leaves_the_gradient_finite_differences_give():
    """Turning a non-sequence into a recurrent state adds an input and an output to the loop, and the
    gradient has to survive that: a statistic is not differentiated through, but the loop it now lives in
    is."""
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    batch_norm = BatchNorm("bn", n_in=4)
    normalized = scan(lambda x_t: batch_norm(x_t), sequences=[xseq], return_updates=False)
    [carried], _ = carry_scan_statistics([normalized])

    cost = (carried**2).sum()
    cost_fn = pytensor.function([xseq], cost)
    grad_fn = pytensor.function([xseq], pytensor.grad(cost, batch_norm.scale))
    scale, scale_0 = batch_norm.scale, batch_norm.scale.get_value()
    X = np.random.default_rng(3).normal(size=(5, 8, 4)).astype(floatX)

    analytic, epsilon = grad_fn(X), 1e-6
    numeric = np.zeros_like(scale_0)
    for i in range(scale_0.shape[0]):
        for sign in (1, -1):
            perturbed = scale_0.copy()
            perturbed[i] += sign * epsilon
            scale.set_value(perturbed)
            numeric[i] += sign * cost_fn(X)
        numeric[i] /= 2 * epsilon
    scale.set_value(scale_0)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5)


def test_one_layer_applied_twice_inside_a_loop_counts_both_applications():
    """Reuse chains the applications, so the state the loop carries has to be the last one's, not the
    first's. Carrying the first would drop half of what the layer saw at every step."""
    rng = np.random.default_rng(0)
    X = rng.normal(loc=50.0, scale=3.0, size=(6, 8, 4)).astype(floatX)
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    batch_norm = BatchNorm("bn", n_in=4, momentum=0.1, epsilon=1e-5)
    normalized = scan(
        lambda x_t: batch_norm(batch_norm(x_t)), sequences=[xseq], return_updates=False
    )
    loss = (normalized**2).sum()

    function([xseq], loss, updates=adam(1e-3)(loss, collect_trainable_params(loss)))(X)

    momentum, expected_mean = batch_norm.momentum, np.zeros(4)
    for step in range(X.shape[0]):
        expected_mean = momentum * X[step].mean(axis=0) + (1 - momentum) * expected_mean
        twice = (X[step] - X[step].mean(axis=0)) / np.sqrt(X[step].var(axis=0) + batch_norm.epsilon)
        expected_mean = momentum * twice.mean(axis=0) + (1 - momentum) * expected_mean

    np.testing.assert_allclose(batch_norm.running_mean.get_value(), expected_mean, rtol=1e-5)


def test_two_layers_in_one_loop_are_both_carried():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=50.0, scale=3.0, size=(6, 8, 4)).astype(floatX)
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    first = BatchNorm("first", n_in=4, momentum=0.1)
    second = BatchNorm("second", n_in=4, momentum=0.1)
    # Shifted between the two, because the second layer reads what the first already normalized and a
    # statistic of that sits at zero whether it was carried or not.
    normalized = scan(
        lambda x_t: second(first(x_t) + 100.0), sequences=[xseq], return_updates=False
    )
    loss = (normalized**2).sum()

    function([xseq], loss, updates=adam(1e-3)(loss, collect_trainable_params(loss)))(X)

    expected_first, _ = chained_statistics(X, first.momentum)
    steps, momentum = X.shape[0], second.momentum
    np.testing.assert_allclose(first.running_mean.get_value(), expected_first, rtol=1e-5)
    np.testing.assert_allclose(
        second.running_mean.get_value(), 100.0 * (1 - (1 - momentum) ** steps), rtol=1e-4
    )


def test_a_while_loop_is_left_alone_by_the_carry():
    """A while loop runs a number of steps the accumulated statistic would depend on, and its condition
    is an output the rebuild does not know how to carry, so it declines instead."""
    batch_norm = BatchNorm("bn", n_in=4)

    def step(state):
        running = state + batch_norm(state)
        return running, until(running.sum() > 3.0)

    trace = scan(step, outputs_info=[pt.zeros((2, 4))], n_steps=20, return_updates=False)
    fgraph = pytensor.graph.FunctionGraph(outputs=[trace], clone=False)
    [node] = scan_nodes(trace)

    assert node.op.info.as_while
    assert carry_statistics_through_scan.transform(fgraph, node) is None


def test_a_loop_that_writes_no_statistic_is_left_alone():
    """The rewrite has to decline cleanly, or every scan in every graph would be rebuilt for nothing."""
    X = pt.tensor("X", shape=(None, None, 4))
    plain = RNN("rnn", n_in=4, n_hidden=3)(X)
    fgraph = pytensor.graph.FunctionGraph(outputs=[plain], clone=False)
    [node] = scan_nodes(plain)

    assert carry_statistics_through_scan.transform(fgraph, node) is None


def test_a_statistic_written_in_a_nested_loop_is_reported():
    """The carry reaches the loop whose own inner graph holds the layer. A loop inside a loop would
    otherwise read a value no step advances, which is the silence this replaces."""
    xseq = pt.tensor("xseq", shape=(None, None, None, 4))
    batch_norm = BatchNorm("bn", n_in=4)
    nested = scan(
        lambda block: scan(lambda x_t: batch_norm(x_t), sequences=[block], return_updates=False),
        sequences=[xseq],
        return_updates=False,
    )

    with pytest.raises(NotImplementedError, match="nested in another loop"):
        carry_scan_statistics([nested])
