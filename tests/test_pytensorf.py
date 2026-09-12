import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import config, shared
from pytensor.compile.builders import OpFromGraph
from pytensor.gradient import (
    DisconnectedInputError,
    disconnected_grad,
    grad,
    grad_clip,
    zero_grad,
)
from pytensor.graph.traversal import ancestors
from pytensor.scan import scan
from pytensor.tensor import random as ptr

from pytensor_ml.layers import BatchNorm, Dropout, DropoutLayer, Linear, Sequential
from pytensor_ml.optim import adam
from pytensor_ml.params import step_counter
from pytensor_ml.pytensorf import (
    collect_non_trainable_params,
    collect_trainable_params,
    compile_predict,
    function,
    rewrite_for_prediction,
    rewrite_pregrad,
)
from pytensor_ml.state import initialize_params


def contains_op(graph, op_type):
    return any(isinstance(var.owner.op, op_type) for var in ancestors([graph]) if var.owner)


def test_compile_predict_removes_dropout():
    # The inference rewrite drops Dropout, so repeated calls are deterministic; without it they would differ.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), Dropout(p=0.5))(X)
    parameters = collect_trainable_params(prediction)
    for parameter, value in zip(
        parameters, initialize_params(parameters, rng=np.random.default_rng(0))
    ):
        parameter.set_value(value)

    predict = compile_predict(prediction, inputs=[X])
    X_values = np.random.default_rng(0).normal(size=(8, 4)).astype(config.floatX)
    first, second = predict(X_values), predict(X_values)

    # Guards the test itself: at the zero initialization the output is identically zero, and so stable
    # across calls whether or not dropout was removed.
    assert np.any(first != 0)
    np.testing.assert_allclose(first, second)


def test_rewrite_for_prediction_leaves_the_original_graph_intact():
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), Dropout(p=0.5))(X)

    assert not contains_op(rewrite_for_prediction(prediction), DropoutLayer)
    assert contains_op(prediction, DropoutLayer)


@pytest.mark.parametrize(
    "marker, expected_gradient",
    [
        (zero_grad, [0.0, 0.0, 0.0]),
        (lambda W: grad_clip(W, -1.0, 1.0), [1.0, 1.0, 1.0]),
    ],
    ids=["zero_grad", "grad_clip"],
)
def test_rewrite_pregrad_keeps_gradient_markers(marker, expected_gradient):
    # Every marker is a ViewOp, which canonicalize is otherwise free to splice out. Unmarked, the gradient
    # of this loss is 2 * W, so both expectations only hold while the marker survives into grad.
    W = shared(np.array([1.0, 2.0, 3.0], dtype=config.floatX), name="W")
    loss = (marker(W) ** 2).sum()

    gradient = grad(rewrite_pregrad(loss), W)

    np.testing.assert_allclose(gradient.eval(), expected_gradient)


def test_rewrite_pregrad_keeps_a_stop_gradient_out_of_the_gradient():
    W = shared(np.array([1.0, 2.0, 3.0], dtype=config.floatX), name="W")
    # Differentiating this reads the detached factor as a constant, so the gradient is W itself; with the
    # marker gone the graph is W ** 2 and the gradient comes out twice as large.
    loss = (W * disconnected_grad(W)).sum()

    gradient = grad(rewrite_pregrad(loss), W)

    np.testing.assert_allclose(gradient.eval(), W.get_value())


def test_rewrite_pregrad_leaves_a_fully_detached_parameter_disconnected():
    W = shared(np.array([1.0, 2.0, 3.0], dtype=config.floatX), name="W")
    loss = (disconnected_grad(W) ** 2).sum()

    with pytest.raises(DisconnectedInputError):
        grad(rewrite_pregrad(loss), W)


def test_compiling_leaves_a_generator_another_function_is_drawing_from_alone():
    """Compiling used to replace every generator it touched, so building a second function mid-training
    jumped the first one's noise stream."""

    def four_draws(with_intervening_compile):
        rng = shared(np.random.default_rng(0), name="rng")
        _, noise = ptr.normal(rng=rng, return_next_rng=True)
        draw = function([], noise)
        drawn = [float(draw()) for _ in range(2)]
        if with_intervening_compile:
            _, other_noise = ptr.normal(rng=rng, return_next_rng=True)
            function([], other_noise)
        return drawn + [float(draw()) for _ in range(2)]

    assert four_draws(with_intervening_compile=True) == four_draws(with_intervening_compile=False)


@pytest.mark.parametrize("applications", [1, 2], ids=["applied_once", "applied_twice"])
def test_a_seeded_dropout_reproduces_across_identical_runs(applications):
    """The seed a caller puts on a layer has to survive compilation, or `random_state=` means nothing. It has
    to hold however many times the layer is applied, since each application draws off its own generator."""

    def dropout_masks():
        X = pt.tensor("X", shape=(None, 3))
        dropout = Dropout(p=0.5, random_state=0)
        layers = [Linear("fc", n_in=3, n_out=3)]
        for _ in range(applications):
            layers.append(dropout)
        prediction = Sequential(*layers)(X)
        for parameter in collect_trainable_params(prediction):
            parameter.set_value(np.ones_like(parameter.get_value()))
        forward = function([X], prediction)
        features = np.ones((4, 3), dtype=config.floatX)
        return [forward(features) for _ in range(3)]

    for first, second in zip(dropout_masks(), dropout_masks()):
        np.testing.assert_allclose(first, second)


def test_random_seed_makes_an_unseeded_graph_reproducible():
    def two_draws():
        rng = shared(np.random.default_rng(), name="rng")  # deliberately unseeded
        _, noise = ptr.normal(rng=rng, return_next_rng=True)
        draw = function([], noise, random_seed=42)
        return [float(draw()) for _ in range(2)]

    assert two_draws() == two_draws()


def test_a_reused_dropout_instance_keeps_drawing_new_masks():
    """Using one Dropout object at two points in a network is an ordinary thing to write, and it used to
    freeze the mask for the whole run."""
    X = pt.tensor("X", shape=(None, 3))
    dropout = Dropout(p=0.5, random_state=0)
    prediction = Sequential(
        Linear("fc1", n_in=3, n_out=4), dropout, Linear("fc2", n_in=4, n_out=2), dropout
    )(X)
    for parameter in collect_trainable_params(prediction):
        parameter.set_value(np.ones_like(parameter.get_value()))

    forward = function([X], prediction)
    features = np.ones((4, 3), dtype=config.floatX)
    outputs = [forward(features) for _ in range(4)]

    assert (
        len(dropout.generators) == 2
    )  # one per application, so neither draw is starved of updates
    assert all(
        np.any(outputs[i] != outputs[j]) for i in range(4) for j in range(i + 1, 4)
    )  # a fresh mask on every call, not just eventually


def test_a_returned_generator_is_the_one_the_caller_asked_for():
    """Returning a generator alongside a draw from it is not two draws: an output reads the generator without
    consuming it, so it must keep reading the caller's own."""
    rng = shared(np.random.default_rng(0), name="rng")
    _, noise = ptr.normal(rng=rng, return_next_rng=True)
    draw = function([], [noise, rng])

    expected = rng.get_value(borrow=True).bit_generator.state
    returned = draw()[1].bit_generator.state

    assert returned == expected


def test_a_generator_read_only_by_an_update_still_advances():
    """A rule that adds noise to its step reads a generator the outputs never touch, so collecting updates
    from the outputs alone left that generator frozen and every step took the identical perturbation."""
    rng = shared(np.random.default_rng(0), name="rng")
    parameter = shared(np.zeros(()), name="w")
    _, noise = ptr.normal(rng=rng, return_next_rng=True)

    step = function([], parameter**2, updates={parameter: parameter + noise})

    increments = []
    for _ in range(3):
        before = float(parameter.get_value())
        step()
        increments.append(float(parameter.get_value()) - before)

    assert not np.allclose(increments[0], increments[1])
    assert not np.allclose(increments[1], increments[2])


def test_two_distinct_draws_off_one_generator_are_rejected():
    """One generator cannot serve two different draws: it has no single next state, so nothing can advance
    it and every call would repeat. pytensor calls such a graph inconsistent and threads no update, which
    would otherwise show up as a training run whose noise never changes."""
    rng = shared(np.random.default_rng(0), name="rng")
    _, normal_draw = ptr.normal(rng=rng, return_next_rng=True)
    _, uniform_draw = ptr.uniform(rng=rng, return_next_rng=True)

    with pytest.raises(ValueError, match="which nothing advances"):
        function([], [normal_draw, uniform_draw])


def test_a_shared_generator_is_accepted_when_the_caller_advances_it():
    """The check is about the assembled updates, not about how the graph looks: a caller who threads the
    generator themselves has answered the question, and their update stands."""
    rng = shared(np.random.default_rng(0), name="rng")
    next_rng, normal_draw = ptr.normal(rng=rng, return_next_rng=True)
    _, uniform_draw = ptr.uniform(rng=next_rng, return_next_rng=True)

    draw = function([], [normal_draw, uniform_draw], updates={rng: next_rng})

    first, second = draw(), draw()
    assert float(first[0]) != float(second[0])  # the draw off the generator advances
    assert float(first[1]) != float(second[1])  # and so does the one off the threaded next state


def test_a_draw_inside_a_scan_advances_its_generator():
    """Nothing is passed to `function` as an update: a scan carrying its generator as a recurrent state
    exposes it as an outer input with a matching outer output, and the collector reads that mapping. This is
    the whole of the inner-graph handling, inherited from pymc, and the first test to run it."""
    rng = shared(np.random.default_rng(0), name="rng")

    def one_step(total, generator):
        next_generator, draw = ptr.normal(size=(), rng=generator, return_next_rng=True)
        return total + draw, next_generator

    # Two recurrent states out, not a graph and an updates dict -- the generator is carried as state.
    trace, _generators = scan(
        one_step, outputs_info=[pt.zeros(()), rng], n_steps=5, return_updates=False
    )
    step = function([], trace)

    draws_within_one_call = np.diff(step(), prepend=0.0)

    # Both halves matter: a loop reusing one generator state draws the same number five times while its
    # outer state still advances between calls, so neither assertion catches that alone.
    assert len(np.unique(draws_within_one_call)) == 5
    assert not np.array_equal(step(), step())


def test_a_scan_that_draws_without_threading_its_generator_has_the_draw_lifted_out():
    """The loop draws from a generator it captured rather than carried, so nothing inside can advance it
    and every step would replay one value. Compiling lifts the draw out into five independent ones, which
    is what `hoist_draws_out_of_scan` is for; the sum of five distinct draws is what proves it happened."""
    rng = shared(np.random.default_rng(0), name="rng")

    def one_step(total):
        _, draw = ptr.normal(size=(), rng=rng, return_next_rng=True)
        return total + draw

    trace = scan(one_step, outputs_info=[pt.zeros(())], n_steps=5, return_updates=False)
    fn = function([], trace)

    steps = fn()
    per_step = np.diff(np.concatenate([[0.0], steps]))

    assert len(set(per_step.round(12))) == 5
    assert not np.allclose(steps, fn())


def test_a_draw_inside_an_op_from_graph_advances_its_generator():
    """Every layer in this library is an OpFromGraph, so a layer that draws inside its own inner graph lands
    here. Handing the advanced generator back as an output is what makes it reachable from outside."""
    rng = shared(np.random.default_rng(0), name="rng")
    inner_rng = rng.type()
    next_rng, inner_draw = ptr.normal(size=(3,), rng=inner_rng, return_next_rng=True)
    draw, _ = OpFromGraph([inner_rng], [inner_draw, next_rng])(rng)

    step = function([], draw)

    assert not np.array_equal(step(), step())


def test_two_op_from_graphs_drawing_off_one_generator_are_rejected():
    """An op carrying an inner graph is not itself a draw op, so a generator consumed only inside two of
    them has no single next state and no client that looks like a consumer from outside. Both draws would
    return the same values on every call, correlated with each other."""
    rng = shared(np.random.default_rng(0), name="rng")

    def draw_of(size):
        inner_rng = rng.type()
        next_rng, inner_draw = ptr.normal(size=(size,), rng=inner_rng, return_next_rng=True)
        return OpFromGraph([inner_rng], [inner_draw, next_rng])(rng)[0]

    with pytest.raises(ValueError, match=r"draws from \['rng'\], which nothing advances"):
        function([], [draw_of(2), draw_of(3)])


def test_two_scans_carrying_one_generator_as_state_are_rejected():
    """A Scan threading its generator correctly exposes an outer input and a matching outer output, so
    each scan alone is advanceable. Two of them off one generator still leaves it with no single next
    state, and a Scan is no more a draw op from outside than an OpFromGraph is."""
    rng = shared(np.random.default_rng(0), name="rng")

    def one_step(total, generator):
        next_generator, draw = ptr.normal(size=(), rng=generator, return_next_rng=True)
        return total + draw, next_generator

    def scan_draw(n_steps):
        trace, _ = scan(
            one_step, outputs_info=[pt.zeros(()), rng], n_steps=n_steps, return_updates=False
        )
        return trace[-1]

    with pytest.raises(ValueError, match=r"draws from \['rng'\], which nothing advances"):
        function([], [scan_draw(3), scan_draw(4)])


def test_a_draw_nested_two_inner_graphs_deep_is_found():
    """Layers nest -- an attention block is an OpFromGraph of Linear layers, each its own -- so reaching
    only the first inner graph would miss a generator drawn from further in."""
    rng = shared(np.random.default_rng(0), name="rng")

    def nested_draw_of(size):
        inner_rng = rng.type()
        next_rng, inner_draw = ptr.normal(size=(size,), rng=inner_rng, return_next_rng=True)
        inner_op = OpFromGraph([inner_rng], [inner_draw, next_rng])
        outer_rng = rng.type()
        return OpFromGraph([outer_rng], list(inner_op(outer_rng)))(rng)[0]

    with pytest.raises(ValueError, match=r"draws from \['rng'\], which nothing advances"):
        function([], [nested_draw_of(2), nested_draw_of(3)])


def test_an_update_leaves_the_multiple_client_warning_standing():
    """The check that replaces this warning only speaks for a generator the caller wrote no update for.
    An update silences that check without separating the draws, so the warning has to survive: both still
    read the state each call starts from and return the same values as each other."""
    rng = shared(np.random.default_rng(0), name="rng")

    def draw_and_next(size):
        inner_rng = rng.type()
        next_rng, inner_draw = ptr.normal(size=(size,), rng=inner_rng, return_next_rng=True)
        return OpFromGraph([inner_rng], [inner_draw, next_rng])(rng)

    first, next_rng = draw_and_next(2)
    second, _ = draw_and_next(3)

    with pytest.warns(UserWarning, match="multiple distinct clients"):
        function([], [first, second], updates={rng: next_rng})


def test_identical_draws_off_one_generator_are_allowed():
    """A dropout mask has to be the same in the backward pass as in the forward one, which is two draws
    reading one state on purpose. They are identical operations, so a single update covers both and
    nothing is reported."""
    rng = shared(np.random.default_rng(0), name="rng")

    def draw_of(size):
        inner_rng = rng.type()
        next_rng, inner_draw = ptr.normal(size=(size,), rng=inner_rng, return_next_rng=True)
        return OpFromGraph([inner_rng], [inner_draw, next_rng])(rng)[0]

    step = function([], [draw_of(3), draw_of(3)])

    one_call = step()
    np.testing.assert_array_equal(one_call[0], one_call[1])
    assert not np.array_equal(one_call[0], step()[0])


def test_an_op_from_graph_that_draws_without_threading_is_rejected():
    rng = shared(np.random.default_rng(0), name="rng")
    inner_rng = rng.type()
    _, inner_draw = ptr.normal(size=(3,), rng=inner_rng, return_next_rng=True)
    draw = OpFromGraph([inner_rng], [inner_draw])(rng)

    with pytest.raises(
        ValueError, match="No update found for at least one RNG used in OpFromGraph"
    ):
        function([], draw)


def test_a_generator_an_inner_graph_never_draws_from_is_left_alone():
    """Receiving a generator is not drawing from one, and only the inner graph knows which happened. Treating
    every non-output use as a draw rejected this graph with a message saying it draws from a generator that
    nothing in it touches."""
    rng = shared(np.random.default_rng(0), name="rng")
    inner_rng = rng.type()
    inner_x = pt.vector("x")
    doubled = OpFromGraph([inner_rng, inner_x], [inner_x * 2.0])(rng, pt.ones(3))

    np.testing.assert_allclose(function([], doubled)(), np.full(3, 2.0))


def test_a_draw_shared_between_an_output_and_an_update_is_rejected():
    """An update expression is a reader like any other, so a generator drawn from by both an output and an
    update is read twice and cannot be advanced once."""
    rng = shared(np.random.default_rng(0), name="rng")
    accumulator = shared(np.zeros(()), name="accumulator")
    _, output_draw = ptr.normal(rng=rng, return_next_rng=True)
    _, update_draw = ptr.uniform(rng=rng, return_next_rng=True)

    with pytest.raises(ValueError, match="which nothing advances"):
        function([], output_draw, updates={accumulator: accumulator + update_draw})


def test_a_clock_read_only_by_an_update_advances():
    """A schedule reads the clock and feeds a learning rate into an update expression, never into an
    output, so collecting from the outputs alone would leave the clock at zero and pin the schedule with
    it -- a warmup then holds the rate at zero and the loss never moves."""
    clock = step_counter(name="schedule/step_count")
    parameter = shared(np.zeros(()), name="w")

    step = function([], parameter, updates={parameter: parameter + clock.astype(config.floatX)})

    for expected_count in range(1, 4):
        step()
        assert int(clock.get_value()) == expected_count
    assert float(parameter.get_value()) == 0.0 + 1.0 + 2.0


def test_a_clock_read_by_an_output_advances():
    clock = step_counter()

    read = function([], clock * 2)

    assert [int(read()) for _ in range(3)] == [0, 2, 4]


def test_every_reader_shares_one_advance_of_the_clock():
    """Readers share a clock by referring to the same variable, so a clock two schedules read still counts
    one step per call rather than one per reader."""
    clock = step_counter()
    parameter = shared(np.zeros(()), name="w")
    reported_rate = clock * 2
    applied_rate = clock.astype(config.floatX)

    step = function([], reported_rate, updates={parameter: parameter + applied_rate})

    step()
    step()
    assert int(clock.get_value()) == 2


def test_the_callers_own_clock_update_wins():
    """The threaded advance is a default, not a rule: a caller who writes the clock themselves has said
    what time does here, which is how a diagnostic pins one with `updates={clock: clock}`."""
    clock = step_counter()

    read = function([], clock, updates={clock: clock})

    assert [int(read()) for _ in range(3)] == [0, 0, 0]
    assert int(clock.get_value()) == 0


def test_clocks_holding_different_counts_are_rejected():
    """Every clock counts training steps, so two of them disagreeing means a checkpoint restored one and
    not the other; advancing both from where they are would keep the two schedules out of step forever."""
    restored, fresh = step_counter(name="restored"), step_counter(name="fresh")
    restored.set_value(np.asarray(7, dtype="int64"))

    with pytest.raises(ValueError, match="hold different step counts"):
        function([], restored + fresh)


def test_a_graph_reading_both_a_clock_and_a_generator_advances_each():
    """Clocks and generators are threaded from the same traversal and merged into one updates dict, so a
    step that both draws noise and reads the time has to advance both, not whichever lands last."""
    clock = step_counter(name="clock")
    rng = shared(np.random.default_rng(0), name="rng")
    _, noise = ptr.normal(rng=rng, return_next_rng=True)

    step = function([], [clock, noise])

    counts, draws = zip(*(step() for _ in range(3)))
    assert [int(count) for count in counts] == [0, 1, 2]
    assert len(set(float(draw) for draw in draws)) == 3


def test_pinned_clocks_are_exempt_from_the_agreeing_count_check():
    """Disagreeing clocks mean a half-restored checkpoint only when the step is advancing them. A caller
    who pins every clock has said none of them counts this step, so there is nothing left to disagree."""
    restored, fresh = step_counter(name="restored"), step_counter(name="fresh")
    restored.set_value(np.asarray(7, dtype="int64"))

    read = function([], restored + fresh, updates={restored: restored, fresh: fresh})

    assert [int(read()) for _ in range(2)] == [7, 7]
    assert (int(restored.get_value()), int(fresh.get_value())) == (7, 0)


def batch_norm_regression():
    """A network whose batch norm sits on the gradient path, with data far off unit scale so frozen
    statistics show up as a large prediction error rather than a small one."""
    x, y = pt.matrix("x"), pt.vector("y")
    network = Sequential(
        Linear("fc1", n_in=3, n_out=4), BatchNorm("bn", n_in=4), Linear("fc2", n_in=4, n_out=1)
    )
    prediction = network(x)[:, 0]
    initialize_params(collect_trainable_params(prediction), rng=np.random.default_rng(0))

    rng = np.random.default_rng(0)
    features = rng.normal(size=(64, 3)) * 10.0 + 5.0
    return x, y, prediction, features, features @ np.arange(3.0)


def test_a_statistic_the_model_writes_advances_under_hand_assembled_updates():
    """The write-back is the op's own output, not a rule's, so a caller who assembles their own updates
    has nothing to write it with. Left uncollected, the network trains against batch statistics and then
    predicts against the ones it started with, which is a silent six-order-of-magnitude error."""
    x, y, prediction, features, targets = batch_norm_regression()
    parameters = collect_trainable_params(prediction)
    loss = pt.mean((prediction - y) ** 2)

    step = function([x, y], loss, updates=adam(1e-1)(loss, parameters))
    for _ in range(200):
        step(features, targets)

    predict = compile_predict(prediction, inputs=[x])
    assert float(np.mean((predict(features) - targets) ** 2)) < 1e-3


def test_a_statistic_reached_only_by_an_output_is_observed_not_advanced():
    """Reading a statistic to monitor it must not train it, so the write-back is traced from what feeds
    the updates rather than from everything the function returns."""
    x = pt.matrix("x")
    monitor = BatchNorm("bn", n_in=4)(Linear("fc", n_in=4, n_out=4)(x))
    running_mean = next(
        p for p in collect_non_trainable_params(monitor) if "running_mean" in p.name
    )
    before = running_mean.get_value().copy()

    read = function([x], monitor.mean())
    read(np.random.default_rng(0).normal(size=(16, 4)).astype(config.floatX))

    np.testing.assert_allclose(running_mean.get_value(), before)


def test_the_callers_own_statistic_update_wins():
    """The threaded write-back is a default, not a rule: a caller who writes the statistic themselves has
    said what it holds, which is how a warm-up pass pins one with ``updates={stat: stat}``. Two things
    guarantee that today -- the collector skips it and the merge puts the caller's updates last -- so this
    pins the behavior rather than either mechanism."""
    x = pt.matrix("x")
    normalized = BatchNorm("bn", n_in=4)(Linear("fc", n_in=4, n_out=4)(x))
    running_mean = next(
        p for p in collect_non_trainable_params(normalized) if "running_mean" in p.name
    )
    # Something has to be written from an expression that reaches the batch norm, or the collector finds
    # nothing to skip and the test passes whatever the code does.
    trained = shared(np.zeros(()), name="w")

    step = function(
        [x],
        trained,
        updates={trained: trained + normalized.sum(), running_mean: running_mean},
    )
    step(np.random.default_rng(0).normal(size=(16, 4)).astype(config.floatX))

    np.testing.assert_allclose(running_mean.get_value(), np.zeros(4, dtype=config.floatX))


def test_a_prediction_graph_writes_no_statistic():
    """The inference rewrite drops the stateful op, so specializing a graph is what makes scoring free of
    side effects -- on a loss as much as on a prediction."""
    x, y, prediction, features, targets = batch_norm_regression()
    loss = pt.mean((prediction - y) ** 2)
    running_mean = next(
        p for p in collect_non_trainable_params(prediction) if "running_mean" in p.name
    )
    before = running_mean.get_value().copy()

    score = function([x, y], rewrite_for_prediction(loss))
    score(features, targets)

    np.testing.assert_allclose(running_mean.get_value(), before)


def test_one_step_advances_every_kind_of_state():
    """`function` merges three separate collections into one updates dict: generator draws, clock advances
    and the model's own write-backs. A merge that let one shadow another would leave a step that trains
    but silently freezes whichever lost."""
    x = pt.matrix("x")
    clock = step_counter(name="clock")
    network = Sequential(
        Linear("fc", n_in=4, n_out=4), BatchNorm("bn", n_in=4), Dropout(p=0.5, random_state=0)
    )
    activations = network(x)
    running_mean = next(
        p for p in collect_non_trainable_params(activations) if "running_mean" in p.name
    )
    before = running_mean.get_value().copy()

    # The clock and the generator are read by the outputs; the statistic is written only because an
    # update expression reaches the batch norm, which is the asymmetry between the three collections.
    trained = shared(np.zeros(()), name="w")
    step = function([x], [activations.sum(), clock], updates={trained: trained + activations.sum()})
    batch = (np.random.default_rng(0).normal(size=(16, 4)) * 5.0).astype(config.floatX)
    draws, counts = zip(*(step(batch) for _ in range(3)))

    assert [int(count) for count in counts] == [0, 1, 2]
    assert len(set(float(draw) for draw in draws)) == 3
    assert not np.allclose(running_mean.get_value(), before)
