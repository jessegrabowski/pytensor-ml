import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor import config

from pytensor_ml.optim import (
    adadelta,
    adagrad,
    adam,
    adamax,
    adamw,
    chain,
    clip_by_global_norm,
    compile_train,
    constant_schedule,
    cosine_schedule,
    linear_schedule,
    nadam,
    rmsprop,
    rprop,
    rprop_updates,
    scalar_state,
    scale,
    scale_by_schedule,
    sgd,
    to_floatx,
)
from pytensor_ml.params import StepCounter, step_counter, trainable
from pytensor_ml.pytensorf import collect_step_counters, function

RTOL = 1e-6


def quadratic_problem(start=2.0):
    """A parameter whose gradient is itself, so a unit-rate step is exactly ``-p``."""
    p = trainable(np.array([start]), name="w")
    return p, 0.5 * (p**2).sum()


def test_float_rate_scales_the_step():
    p, loss = quadratic_problem()
    step = compile_train(loss, sgd(learning_rate=0.1))

    step()
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=RTOL)


def test_shared_rate_is_steerable_from_python():
    p, loss = quadratic_problem()
    learning_rate = scalar_state("learning_rate", fill_value=0.1)
    step = compile_train(loss, sgd(learning_rate=learning_rate))

    step()  # lr = 0.1 -> p = 2 - 0.1 * 2
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=RTOL)

    learning_rate.set_value(np.asarray(0.5, dtype=config.floatX))
    step()  # lr = 0.5 -> p = 1.8 - 0.5 * 1.8
    np.testing.assert_allclose(p.get_value(), [0.9], rtol=RTOL)

    # A shared rate is passed through, so no schedule state is allocated to drive it.
    assert not any(
        "schedule" in (key.name or "") for key in sgd(learning_rate=learning_rate)(loss, [p])
    )


def test_schedule_rate_alone_sets_the_step_size():
    p, loss = quadratic_problem()
    step = compile_train(loss, sgd(learning_rate=cosine_schedule(0.1, 2)))

    step()  # step 0 -> lr = 0.1
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=RTOL)
    step()  # step 1 -> lr = 0.05
    np.testing.assert_allclose(p.get_value(), [1.71], rtol=RTOL)
    step()  # step 2 -> lr = 0
    np.testing.assert_allclose(p.get_value(), [1.71], rtol=RTOL)


def test_a_scheduled_rate_matches_terminal_scaling_for_a_multiplicative_rate():
    """A rate the rule reads and the same rate applied to the finished step must agree wherever the rate is a
    plain multiplier, which holds for every rule in the tree."""
    schedule = cosine_schedule(0.05, 20)

    read_by_the_rule, rule_loss = quadratic_problem()
    rule_step = compile_train(rule_loss, adam(learning_rate=schedule))

    scaled, scaled_loss = quadratic_problem()
    clock = step_counter()
    scaled_step = compile_train(scaled_loss, chain(adam(learning_rate=1.0), scale(schedule(clock))))

    for _ in range(10):
        rule_step()
        scaled_step()

    np.testing.assert_allclose(read_by_the_rule.get_value(), scaled.get_value(), rtol=1e-5)


def test_a_scheduled_rule_allocates_no_rate_variable():
    """The rate is an expression over the clock, so there is no rate state to keep or checkpoint. A caller
    who wants to read it builds `curve(clock)` and asks for it as an extra output."""
    p, loss = quadratic_problem()
    updates = adam(learning_rate=cosine_schedule(0.1, 4))(loss, [p])

    assert not any("learning_rate" in (key.name or "") for key in updates)


@pytest.mark.parametrize(
    "learning_rate",
    [cosine_schedule(0.1, 10), scalar_state("rprop/rejected_rate", 0.1)],
    ids=["schedule", "shared_variable"],
)
def test_rprop_rejects_a_symbolic_rate(learning_rate):
    """Rprop's rate seeds per-parameter state at allocation time, so neither symbolic form can reach it."""
    with pytest.raises(TypeError, match="initializes its per-parameter step sizes"):
        rprop(learning_rate=learning_rate)

    p, loss = quadratic_problem()
    with pytest.raises(TypeError, match="initializes its per-parameter step sizes"):
        rprop_updates(loss, [p], learning_rate=learning_rate)


def test_a_scheduled_rule_preserves_parameter_identity():
    """The updates must keep updating the parameter object the model holds; a clone would train a copy and
    silently leave the model's weights untouched."""
    p, loss = quadratic_problem()
    rule = adam(learning_rate=cosine_schedule(0.1, 4))

    assert p in rule(loss, [p])
    # Adam's first step is -lr * sign(gradient) once bias correction is applied, so p = 2 - 0.1.
    compile_train(loss, rule)()
    np.testing.assert_allclose(p.get_value(), [1.9], rtol=1e-4)


def test_schedule_reaches_a_rate_applied_by_a_chained_scale():
    """sgd with momentum applies its rate through `scale` rather than through sgd_updates, so the rate has to
    reach it there. Unit-rate sgd gives step -p and momentum leaves the first step unchanged."""
    p, loss = quadratic_problem()
    step = compile_train(loss, sgd(learning_rate=cosine_schedule(0.1, 4), momentum=0.9))

    step()  # rate 0.1 on a step of -2, so p = 2 - 0.2
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=RTOL)


def test_a_clock_read_by_a_schedule_drives_the_rate():
    p, loss = quadratic_problem()
    clock = step_counter()
    step = compile_train(loss, sgd(learning_rate=cosine_schedule(0.1, 10)(clock)))

    step()  # rate at step 0 is 0.1 -> p = 2 - 0.1 * 2
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=RTOL)
    assert int(clock.get_value()) == 1


def test_a_caller_held_clock_matches_the_rule_own_clock_over_many_steps():
    """Handing a rule an unapplied schedule and handing it the same schedule read off a clock of your own must
    place the curve at the same step, or the two ways of scheduling drift apart."""
    schedule = cosine_schedule(0.05, 20)

    rule_clocked, rule_loss = quadratic_problem()
    rule_step = compile_train(rule_loss, adam(learning_rate=schedule))

    caller_clocked, caller_loss = quadratic_problem()
    clock = step_counter()
    caller_step = compile_train(caller_loss, adam(learning_rate=schedule(clock)))

    for _ in range(40):
        rule_step()
        caller_step()

    np.testing.assert_allclose(caller_clocked.get_value(), rule_clocked.get_value(), rtol=RTOL)


def test_a_clocked_rate_is_available_as_an_extra_output():
    """The rate is a node in the graph, so the rate that was applied comes back with the loss."""
    _, loss = quadratic_problem()
    clock = step_counter()
    schedule = cosine_schedule(0.1, 4)
    learning_rate = schedule(clock)
    step = compile_train(loss, sgd(learning_rate=learning_rate), extra_outputs=[learning_rate])

    applied = [float(step()[1]) for _ in range(4)]
    expected = [float(schedule(pt.as_tensor(t, dtype="int64")).eval()) for t in range(4)]

    np.testing.assert_allclose(applied, expected, rtol=RTOL)


def test_a_clock_advances_once_when_two_schedules_read_it():
    """Two rates off one clock must not tick it twice, or both schedules run at double speed."""
    _, loss = quadratic_problem()
    clock = step_counter()
    fast = cosine_schedule(0.1, 10)(clock)
    slow = cosine_schedule(0.01, 10)(clock)
    step = compile_train(loss, sgd(learning_rate=fast + slow), extra_outputs=[fast, slow])

    for _ in range(3):
        step()

    assert int(clock.get_value()) == 3


def test_two_training_functions_share_a_held_clock():
    """Compiling twice from one configured rule needs no state memoization when the caller owns time."""
    _, loss = quadratic_problem()
    clock = step_counter()
    rule = sgd(learning_rate=cosine_schedule(0.1, 10)(clock))

    first = compile_train(loss, rule)
    second = compile_train(loss, rule)
    for _ in range(3):
        first()
    for _ in range(3):
        second()

    assert int(clock.get_value()) == 6


def test_compile_train_accepts_a_rule_that_advances_the_clock_the_same_way():
    """Rules advance the clock they count their own steps on, so their write and the collected one agree.
    Accepting it is what lets a rule's updates stand on their own outside compile_train."""
    _, loss = quadratic_problem()
    clock = step_counter()

    def rule(loss_or_gradients, parameters):
        updates = sgd(learning_rate=cosine_schedule(0.1, 10)(clock))(loss_or_gradients, parameters)
        return {**updates, clock: clock + 1}

    step = compile_train(loss, rule)
    step()
    step()

    assert int(clock.get_value()) == 2  # advanced once per step, not twice


def test_compile_train_rejects_a_rule_that_advances_the_clock_differently():
    """Two disagreeing writers would leave the clock holding one of them while both looked configured."""
    _, loss = quadratic_problem()
    clock = step_counter()

    def rule(loss_or_gradients, parameters):
        updates = sgd(learning_rate=cosine_schedule(0.1, 10)(clock))(loss_or_gradients, parameters)
        return {**updates, clock: clock + 5}

    with pytest.raises(ValueError, match="not the one-step advance"):
        compile_train(loss, rule)


def test_a_clock_read_only_by_an_extra_output_still_advances():
    """Collection covers the diagnostics too, or a reported step count would sit at zero."""
    _, loss = quadratic_problem()
    clock = step_counter()
    step = compile_train(loss, sgd(learning_rate=0.1), extra_outputs=[clock.astype("float64")])

    reported = [float(step()[1]) for _ in range(3)]

    assert reported == [0.0, 1.0, 2.0]
    assert int(clock.get_value()) == 3


def test_a_clock_reachable_only_from_the_reported_loss_still_advances():
    """A rule is free to ignore the loss it is handed and build updates from gradients it already has, which
    leaves the loss graph out of the updates entirely. The clock is then reachable only from the loss the
    function reports, and it still has to tick."""
    p, loss = quadratic_problem()
    clock = step_counter()
    reported_loss = loss + clock.astype(config.floatX)
    precomputed = [pt.grad(loss, p)]

    step = compile_train(
        reported_loss,
        lambda _, parameters: sgd(learning_rate=0.1)(precomputed, parameters),
        parameters=[p],
    )
    step()
    step()

    assert int(clock.get_value()) == 2


def test_a_restored_clock_resumes_the_schedule_where_it_left_off():
    """Holding the clock is what makes a checkpoint resume mid-curve rather than restart the schedule."""
    _, loss = quadratic_problem()
    clock = step_counter()
    schedule = cosine_schedule(0.1, 20)
    learning_rate = schedule(clock)
    step = compile_train(loss, sgd(learning_rate=learning_rate), extra_outputs=[learning_rate])

    clock.set_value(np.asarray(15, dtype="int64"))
    _, applied = step()

    expected = float(schedule(pt.as_tensor(15, dtype="int64")).eval())
    np.testing.assert_allclose(applied, expected, rtol=RTOL)
    assert int(clock.get_value()) == 16


def test_a_clock_that_shadows_a_parameter_name_is_caught():
    """Clocks are collected before the name guard runs, so they cannot silently alias parameter state."""
    _, loss = quadratic_problem()
    shadowing_clock = step_counter(name="w")

    with pytest.raises(ValueError, match="share the name 'w'"):
        compile_train(loss, sgd(learning_rate=cosine_schedule(0.1, 10)(shadowing_clock)))


def test_a_rule_counts_its_own_steps_on_a_clock():
    """The counter adam keeps for bias correction is a clock, so it is collected as one and a schedule can
    read the same notion of time the rule uses."""
    p, loss = quadratic_problem()
    updates = adam(learning_rate=0.1)(loss, [p])

    counters = collect_step_counters(list(updates.values()))
    assert [counter.name for counter in counters] == ["adam/step_count"]
    assert all(isinstance(counter, StepCounter) for counter in counters)


def test_a_user_clock_out_of_step_with_a_rule_clock_is_caught():
    """Two clocks can still coexist when the caller holds one of their own, and a checkpoint that restored
    only one of them leaves the two measuring different times."""
    p, loss = quadratic_problem()
    their_clock = step_counter("their_clock")
    rule = adam(learning_rate=0.1)
    adam_clock = next(key for key in rule(loss, [p]) if key.name == "adam/step_count")
    adam_clock.set_value(np.asarray(37, dtype="int64"))

    with pytest.raises(ValueError, match="hold different step counts"):
        compile_train(loss, rule, extra_outputs=[their_clock.astype("float64")])


@pytest.mark.parametrize(
    "alias, clock_name",
    [
        (adam, "adam/step_count"),
        (adamw, "adamw/step_count"),
        (nadam, "nadam/step_count"),
        (adamax, "adamax/step_count"),
        (sgd, "sgd/step_count"),
        (rmsprop, "rmsprop/step_count"),
        (adagrad, "adagrad/step_count"),
        (adadelta, "adadelta/step_count"),
    ],
    ids=["adam", "adamw", "nadam", "adamax", "sgd", "rmsprop", "adagrad", "adadelta"],
)
def test_a_scheduled_rule_keeps_exactly_one_clock(alias, clock_name):
    """One notion of time per rule, under the name the schedule and the rule's own step count agree on. A rule
    that counted its steps under a different name would hold a second clock measuring the same time, and a
    checkpoint would carry two counts to keep in step."""
    _, loss = quadratic_problem()
    step = compile_train(loss, alias(learning_rate=cosine_schedule(0.1, 20)))

    for _ in range(3):
        step()

    clocks = [v for v in step.get_shared() if isinstance(v, StepCounter)]
    assert [clock.name for clock in clocks] == [clock_name]
    assert int(clocks[0].get_value()) == 3


def test_a_caller_held_clock_keeps_advancing_for_other_readers():
    """A clock is state that outlives the compile: the caller reads it, and so can another compiled function.
    Compiling a training step must not take it out of the set that step advances."""
    _, loss = quadratic_problem()
    clock = step_counter("agent_clock")
    step = compile_train(loss, adam(learning_rate=cosine_schedule(0.1, 100)(clock)))
    exploration_schedule = cosine_schedule(1.0, 100)
    epsilon = function([], exploration_schedule(clock))

    for _ in range(20):
        step()

    assert int(clock.get_value()) == 20
    expected = float(exploration_schedule(pt.as_tensor(20, dtype="int64")).eval())
    np.testing.assert_allclose(epsilon(), expected, rtol=RTOL)


@pytest.mark.parametrize("rate_dtype", ["float32", "float64"], ids=["float32", "float64"])
def test_a_shared_rate_is_read_at_the_graphs_floatx(rate_dtype):
    """A shared rate keeps whatever dtype it was allocated with, which need not match the floatX the graph
    is built under -- restoring a checkpoint into a differently configured session is the ordinary way to
    get there. Left alone the mismatch reaches the parameter update, where pytensor refuses it and names
    the parameter rather than the rate."""
    p, loss = quadratic_problem()
    learning_rate = pytensor.shared(np.asarray(0.1, dtype=rate_dtype), name="learning_rate")

    step = compile_train(loss, sgd(learning_rate=learning_rate))
    step()

    np.testing.assert_allclose(p.get_value(), [1.8], rtol=RTOL)


def test_to_floatx_leaves_a_well_typed_rate_untouched():
    """What keeps every graph that already worked bit-exact: nothing is wrapped that does not need it. A
    cast node on a matching rate, or an array in place of a float literal, would change the graph a rule
    builds without changing what it computes -- invisible except as drift."""
    matching = pytensor.shared(np.asarray(0.1, dtype=config.floatX), name="learning_rate")

    assert to_floatx(matching) is matching
    assert to_floatx(0.1) == 0.1
    assert isinstance(to_floatx(0.1), float)  # a number, not a 0-d array as pymc's floatX returns


def test_to_floatx_casts_a_rate_stored_at_another_dtype():
    other_dtype = "float32" if config.floatX == "float64" else "float64"
    stored = pytensor.shared(np.asarray(0.1, dtype=other_dtype), name="learning_rate")

    assert to_floatx(stored).dtype == config.floatX


def test_a_scheduled_scale_advances_its_own_clock():
    """The caller writes no clock bookkeeping. `compile_train` finds the clock in the graph and advances it,
    so the rate moves along the schedule from one step to the next on its own."""
    p, loss = quadratic_problem()
    schedule = linear_schedule(0.5, total_steps=4, final_learning_rate=0.0)
    step = compile_train(loss, chain(sgd(learning_rate=1.0), scale_by_schedule(schedule)))

    step()  # rate 0.5 on a unit-rate step of -2, so p = 2 - 1
    np.testing.assert_allclose(p.get_value(), [1.0], rtol=RTOL)
    step()  # the clock advanced, so the rate is now 0.375 on a step of -1
    np.testing.assert_allclose(p.get_value(), [0.625], rtol=RTOL)


def test_a_clip_before_a_scheduled_scale_bounds_the_step_in_gradient_units():
    """Scheduling after the rule buys exactly this over scheduling inside it. With the clip between the two
    the bound applies to a unit-rate step, so it stays in gradient units; a rule that has already applied the
    rate hands the clip a step the schedule has shrunk, and the same bound never binds."""
    rate = constant_schedule(0.1)

    scaled_after, scaled_loss = quadratic_problem()
    compile_train(
        scaled_loss,
        chain(sgd(learning_rate=1.0), clip_by_global_norm(0.5), scale_by_schedule(rate)),
    )()

    read_by_the_rule, rule_loss = quadratic_problem()
    compile_train(rule_loss, chain(sgd(learning_rate=rate), clip_by_global_norm(0.5)))()

    # Clipped first: the step of -2 is bounded to -0.5, then scaled to -0.05.
    np.testing.assert_allclose(scaled_after.get_value(), [1.95], rtol=RTOL)
    # Scaled first: the step is already -0.2, so the bound of 0.5 never binds.
    np.testing.assert_allclose(read_by_the_rule.get_value(), [1.8], rtol=RTOL)
