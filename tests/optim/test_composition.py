from itertools import pairwise

import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import config

from pytensor_ml.optim import (
    adam,
    add_weight_decay,
    apply_if_finite,
    chain,
    clip_by_global_norm,
    compile_train,
    cosine_schedule,
    large_step,
    reduce_on_plateau,
    scalar_state,
    scale,
    scale_by_schedule,
    sgd,
    skip_if,
)
from pytensor_ml.params import step_counter, trainable
from pytensor_ml.pytensorf import function

GOOD = np.ones(2, dtype=config.floatX)
BAD = np.array([np.nan, 1.0], dtype=config.floatX)


def quadratic_problem():
    """A problem whose gradient is proportional to the batch, so one batch sets the step's scale."""
    x = pt.vector("x")
    p = trainable(np.ones(2, dtype=config.floatX), name="w")
    return p, ((p * x) ** 2).sum()


def state_named(step, name):
    """Return the shared variable the compiled step writes under ``name``."""
    return next(variable for variable in step.get_shared() if variable.name == name)


def test_clipping_bounds_a_rules_step_end_to_end():
    """The clipping transform is otherwise only exercised on a hand-built updates dict; here it has to
    survive a real rule, a real gradient, and compile_train's assembly."""
    p, loss = quadratic_problem()
    step = compile_train(loss, chain(sgd(learning_rate=100.0), clip_by_global_norm(0.5)))

    before = p.get_value().copy()
    step(GOOD)

    np.testing.assert_allclose(np.linalg.norm(p.get_value() - before), 0.5, rtol=1e-5)


def test_clipping_cancels_the_whole_step_around_one_nonfinite_coordinate():
    """Clipping rescales by ``max_norm / norm``, which is zero once the norm is inf. The poisoned
    coordinate becomes ``inf * 0``, and every healthy coordinate is multiplied to nothing, so a single inf
    both destroys its own parameter and silently cancels the step for all the others."""
    p = trainable(np.zeros(2, dtype=config.floatX), name="w")
    poisoned_step = pt.constant(np.array([np.inf, 3.0], dtype=config.floatX))
    clipped = clip_by_global_norm(1.0)({p: p + poisoned_step}, [p])

    poisoned, healthy = function([], clipped[p])()
    assert np.isnan(poisoned)
    assert healthy == 0.0


def test_a_guard_catches_what_clipping_launders():
    """Following from the above: clipping bounds a step's size but cannot survive one that has already
    gone non-finite, so the guard and the clip compose rather than compete."""
    unguarded_p, unguarded_loss = quadratic_problem()
    unguarded = compile_train(
        unguarded_loss, chain(sgd(learning_rate=0.1), clip_by_global_norm(1.0))
    )
    unguarded(BAD)
    assert not np.all(np.isfinite(unguarded_p.get_value()))

    guarded_p, guarded_loss = quadratic_problem()
    guarded = compile_train(
        guarded_loss, apply_if_finite(chain(sgd(learning_rate=0.1), clip_by_global_norm(1.0)))
    )
    guarded(BAD)
    assert np.all(np.isfinite(guarded_p.get_value()))


def test_a_plateau_policy_composes_with_a_schedule():
    """The policy owns a multiplier rather than the rate itself, so a schedule drives the rate underneath
    it. Both have to reach the step: the schedule falls along its curve while the policy cuts its scale."""
    p, loss = quadratic_problem()
    clock = step_counter()
    plateau_scale = scalar_state("plateau/scale", fill_value=1.0)
    rate = plateau_scale * cosine_schedule(0.1, 10)(clock)
    rule = reduce_on_plateau(adam(learning_rate=rate), plateau_scale, factor=0.5, patience=2)
    step = compile_train(loss, rule, extra_outputs=[rate])

    # A loss already parked at its floor can never improve, so every step after the first counts as bad.
    rates = []
    for _ in range(6):
        rates.append(float(step(np.zeros(2, dtype=config.floatX))[1]))

    # The rate the rule actually reads is the product of the two: the curve at the clock's count, scaled
    # by whatever the policy has cut its multiplier down to. Halfway along, cosine is at half its start.
    assert rates[0] == pytest.approx(0.1)
    assert rates[-1] == pytest.approx(0.25 * 0.05)
    assert all(later < earlier for earlier, later in pairwise(rates))
    assert float(plateau_scale.get_value()) == 0.25
    assert int(clock.get_value()) == 6


def test_a_guard_keeps_a_scheduled_rate_advancing_through_a_skip():
    """A schedule reads the clock the rule counts on, and the guard exempts clocks from the freeze, so a
    skipped step still consumes a step and the next batch gets the next rate on the curve."""
    p, loss = quadratic_problem()
    step = compile_train(loss, apply_if_finite(adam(cosine_schedule(0.1, 10))))
    clock = state_named(step, "adam/step_count")

    step(GOOD)
    step(BAD)
    step(GOOD)

    assert int(clock.get_value()) == 3
    assert np.all(np.isfinite(p.get_value()))


def test_weight_decay_still_reaches_a_step_the_guard_lets_through():
    """Decay is a transform inside the chain, so a guard wrapping the whole chain must leave it alone on
    a step that is applied."""
    p, loss = quadratic_problem()
    decayed = chain(sgd(learning_rate=1.0), add_weight_decay(0.1), scale(1.0))
    step = compile_train(loss, apply_if_finite(decayed))

    # A zero batch gives a zero gradient, so the decay term is the only thing left to move the parameter.
    step(np.zeros(2, dtype=config.floatX))

    np.testing.assert_allclose(p.get_value(), np.full(2, 0.9, dtype=config.floatX), rtol=1e-5)


def test_the_whole_spine_leaves_a_healthy_run_alone(run_training):
    """Schedule, clipping, decay, plateau policy and guard stacked on one rule. A run that never
    misbehaves has to come out the other side untouched by the machinery watching it: the network still
    trains, the guard throws nothing away, and the policy never cuts."""
    plateau_scale = scalar_state("plateau/scale", fill_value=1.0)
    total_skips = scalar_state("skip_if/total_skips")
    inner = chain(
        adam(learning_rate=cosine_schedule(1e-2, 50)),
        clip_by_global_norm(1.0),
        add_weight_decay(1e-4),
        scale(plateau_scale),
    )
    rule = apply_if_finite(
        reduce_on_plateau(inner, plateau_scale, patience=5), total_skips=total_skips
    )

    losses = run_training(rule, n_steps=50)

    assert np.all(np.isfinite(losses))
    assert losses[-1] < 0.6 * losses[0]
    assert float(total_skips.get_value()) == 0.0
    assert float(plateau_scale.get_value()) == 1.0


@pytest.mark.parametrize("condition", [None, large_step(1e6)], ids=["nonfinite", "large_step"])
def test_either_condition_guards_a_full_chain(condition):
    p, loss = quadratic_problem()
    inner = chain(sgd(learning_rate=0.1), clip_by_global_norm(1.0))
    step = compile_train(loss, skip_if(inner, condition, max_consecutive_skips=None))

    before = p.get_value().copy()
    step(BAD)

    np.testing.assert_allclose(p.get_value(), before)


def test_two_scheduled_scales_sharing_a_namespace_collide_loudly():
    """Each allocates its own clock, and optimizer state is matched by name when it is saved, so two under
    one name would alias each other on restore. The compile has to be where that is caught."""
    _, loss = quadratic_problem()
    schedule = cosine_schedule(1e-2, 50)
    rule = chain(sgd(learning_rate=1.0), scale_by_schedule(schedule), scale_by_schedule(schedule))

    with pytest.raises(ValueError, match="share the name 'scale_by_schedule/step_count'"):
        compile_train(loss, rule)

    # Naming them apart is the remedy the transform offers, and it compiles.
    apart = chain(
        sgd(learning_rate=1.0),
        scale_by_schedule(schedule, namespace="warmup"),
        scale_by_schedule(schedule, namespace="decay"),
    )
    compile_train(loss, apart)
