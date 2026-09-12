import numpy as np
import pytensor.tensor as pt

from pytensor_ml.optim import (
    add_weight_decay,
    chain,
    linear_schedule,
    scale,
    scale_by_schedule,
    sgd,
    sgd_updates,
    trace,
)
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import collect_step_counters, function


def test_scale_applies_factor():
    p = trainable(np.zeros(2), name="w")
    updates = {p: p + pt.constant(np.array([2.0, -4.0]))}
    out = scale(0.25)(updates, [p])
    np.testing.assert_allclose(function([], out[p])(), [0.5, -1.0])


def test_scale_by_schedule_applies_the_rate_its_clock_reads():
    p = trainable(np.zeros(2), name="w")
    updates = {p: p + pt.constant(np.array([2.0, -4.0]))}
    out = scale_by_schedule(linear_schedule(1.0, total_steps=4, final_learning_rate=0.0))(
        updates, [p]
    )

    (clock,) = collect_step_counters(out[p])
    step = function([], out[p], updates={clock: clock.advance()})

    # The clock starts at zero, where the schedule is still at its initial rate of 1.0.
    np.testing.assert_allclose(step(), [2.0, -4.0])
    # One quarter of the horizon later the rate is 0.75.
    np.testing.assert_allclose(step(), [1.5, -3.0])


def test_scale_by_schedule_allocates_a_clock_per_namespace():
    """Two scheduled scales in one graph measure their own time. Sharing a clock by default would make the
    second one's schedule start wherever the first had already advanced it to."""
    p = trainable(np.zeros(1), name="w")
    schedule = linear_schedule(1.0, total_steps=4)
    out = chain(
        scale_by_schedule(schedule, namespace="warmup"),
        scale_by_schedule(schedule, namespace="decay"),
    )({p: p + pt.constant(np.array([1.0]))}, [p])

    assert sorted(clock.name for clock in collect_step_counters(out[p])) == [
        "decay/step_count",
        "warmup/step_count",
    ]


def test_scale_by_schedule_reuses_its_clock_across_invocations():
    """Two functions compiled from one configured transform must read one clock; a second would restart the
    schedule at zero while the first kept counting."""
    p = trainable(np.zeros(1), name="w")
    transform = scale_by_schedule(linear_schedule(1.0, total_steps=4))

    (first_clock,) = collect_step_counters(transform({p: p + pt.constant(np.array([1.0]))}, [p])[p])
    (second_clock,) = collect_step_counters(
        transform({p: p + pt.constant(np.array([1.0]))}, [p])[p]
    )

    assert first_clock is second_clock


def test_trace_accumulates_velocity_with_decay():
    p = trainable(np.zeros(1), name="w")
    out = trace(0.9)({p: p + pt.constant(np.array([1.0]))}, [p])
    velocity = next(key for key in out if key.name == "w/trace/velocity")
    step = function([], [out[p], out[velocity]], updates=out)

    # Step 1 leaves the decay unexercised: velocity = 0.9 * 0 + 1 = 1, p = 0 + 1 = 1.
    new_p, new_velocity = step()
    np.testing.assert_allclose(new_velocity, [1.0])
    np.testing.assert_allclose(new_p, [1.0])
    # Step 2 exercises the decay: velocity = 0.9 * 1 + 1 = 1.9, p = 1 + 1.9 = 2.9.
    new_p, new_velocity = step()
    np.testing.assert_allclose(new_velocity, [1.9])
    np.testing.assert_allclose(new_p, [2.9])


def test_nesterov_differs_from_classical():
    p = trainable(np.zeros(1), name="w")
    step = pt.constant(np.array([1.0]))
    classical = trace(0.9, nesterov=False)({p: p + step}, [p])
    nesterov = trace(0.9, nesterov=True)({p: p + step}, [p])
    # classical -> velocity = 1; nesterov -> step + decay*velocity = 1 + 0.9 = 1.9
    np.testing.assert_allclose(function([], classical[p])(), [1.0])
    np.testing.assert_allclose(function([], nesterov[p])(), [1.9])


def test_chain_threads_updates_in_order():
    p = trainable(np.ones(1), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p = 1, so unit-rate sgd step is -1
    out = chain(trace(0.9), scale(0.1))(sgd_updates(loss, [p], learning_rate=1.0), [p])

    assert p in out
    assert any(k.name == "w/trace/velocity" for k in out)
    # velocity = -1, scaled step = 0.1 * -1, new p = 1 - 0.1 = 0.9
    np.testing.assert_allclose(function([], out[p])(), [0.9])


def test_a_rule_headed_chain_composes_again_as_a_rule():
    """A chain led by a rule is a rule, so it can head another chain. Composing in two places is how a
    caller shares a configured optimizer and adds one more transform at the point of use."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p, so a unit-rate sgd step is -p
    out = chain(chain(sgd(learning_rate=1.0), scale(0.5)), scale(0.5))(loss, [p])

    # Two nested halvings of a step of -2 leave p = 2 - 0.5.
    np.testing.assert_allclose(function([], out[p])(), [1.5])


def test_chain_reuses_transform_state_across_invocations():
    """A chain's transforms allocate state too, so two functions compiled from one chain must share it.
    Without reuse the second function silently restarts with its own velocity buffer."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()
    rule = chain(trace(0.9), scale(0.1))

    first = {key for key in rule(sgd_updates(loss, [p], learning_rate=1.0), [p]) if key is not p}
    second = {key for key in rule(sgd_updates(loss, [p], learning_rate=1.0), [p]) if key is not p}

    assert first and first == second


def test_separately_configured_chains_keep_independent_state():
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()

    def build_updates():
        rule = chain(trace(0.9), scale(0.1))
        return {key for key in rule(sgd_updates(loss, [p], learning_rate=1.0), [p]) if key is not p}

    assert not build_updates() & build_updates()


def test_add_weight_decay_subtracts_decay_term():
    p = trainable(np.array([4.0]), name="w")
    # An empty base step (updates[p] == p) isolates the decay term: new step = -0.1 * 4.
    out = chain(add_weight_decay(0.1), scale(1.0))({p: p}, [p])
    np.testing.assert_allclose(function([], out[p])(), [3.6])


def test_add_weight_decay_skips_masked_params():
    p = trainable(np.array([4.0]), name="bias")
    out = add_weight_decay(0.1, mask=lambda param: "bias" not in param.name)({p: p}, [p])
    np.testing.assert_allclose(function([], out[p])(), [4.0])
