import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import config

from pytensor_ml.layers import BatchNorm, Linear, Sequential
from pytensor_ml.optim import (
    adam,
    apply_if_finite,
    compile_train,
    large_step,
    nonfinite,
    reduce_on_plateau,
    scalar_state,
    sgd,
    skip_if,
)
from pytensor_ml.params import trainable

GOOD = np.ones(2, dtype=config.floatX)
BAD = np.array([np.nan, 1.0], dtype=config.floatX)


def poisonable_problem():
    """A problem whose loss goes non-finite exactly when the batch handed to it does."""
    x = pt.vector("x")
    p = trainable(np.ones(2, dtype=config.floatX), name="w")
    return p, ((p * x) ** 2).sum()


def state_named(step, name):
    """Return the shared variable the compiled step writes under ``name``."""
    return next(variable for variable in step.get_shared() if variable.name == name)


@pytest.mark.parametrize("poison", [np.nan, np.inf], ids=["nan", "inf"])
def test_holds_back_parameters_and_state_on_a_nonfinite_step(poison):
    p, loss = poisonable_problem()
    step = compile_train(loss, apply_if_finite(adam(0.1)))
    first_moment = state_named(step, "w/adam/first_moment")

    step(GOOD)
    weights, moment = p.get_value().copy(), first_moment.get_value().copy()

    step(np.array([poison, 1.0], dtype=config.floatX))

    # The moment matters as much as the weights: one poisons every later step through it.
    np.testing.assert_allclose(p.get_value(), weights)
    np.testing.assert_allclose(first_moment.get_value(), moment)


def test_resumes_training_where_the_skipped_step_left_it():
    """The skipped step must be a true no-op, not a damped or partial one: two good batches around it have
    to land exactly where the same two would have landed on their own."""
    p, loss = poisonable_problem()
    guarded = compile_train(loss, apply_if_finite(sgd(0.1)))
    for batch in [GOOD, BAD, GOOD]:
        guarded(batch)

    reference_p, reference_loss = poisonable_problem()
    unguarded = compile_train(reference_loss, sgd(0.1))
    for _ in range(2):
        unguarded(GOOD)

    np.testing.assert_allclose(p.get_value(), reference_p.get_value())


def test_counts_consecutive_skips_and_resets_on_a_good_step():
    p, loss = poisonable_problem()
    consecutive = scalar_state("skip_if/consecutive_skips")
    step = compile_train(loss, apply_if_finite(sgd(0.1), consecutive_skips=consecutive))

    counts = []
    for batch in [GOOD, BAD, BAD, GOOD, BAD]:
        step(batch)
        counts.append(float(consecutive.get_value()))

    assert counts == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_counts_every_skip_in_a_total_that_never_resets():
    """The consecutive count reports only the streak in progress, so reading it at the end of a run gives
    the trailing streak rather than how often the run skipped at all."""
    p, loss = poisonable_problem()
    total = scalar_state("skip_if/total_skips")
    step = compile_train(loss, apply_if_finite(sgd(0.1), total_skips=total))

    for batch in [GOOD, BAD, BAD, GOOD, BAD, GOOD]:
        step(batch)

    assert float(total.get_value()) == 3.0


def test_raises_once_the_skips_run_past_the_tolerance():
    """A run that never recovers is a divergence, and skipping forever looks like training that has simply
    stopped learning."""
    p, loss = poisonable_problem()
    step = compile_train(loss, apply_if_finite(sgd(0.1), max_consecutive_skips=2))

    step(BAD)
    step(BAD)
    with pytest.raises(FloatingPointError, match="3 consecutive"):
        step(BAD)

    assert np.all(np.isfinite(p.get_value()))


def test_names_the_condition_in_the_error_it_raises():
    p, loss = poisonable_problem()
    step = compile_train(loss, skip_if(sgd(0.1), large_step(1e-3), max_consecutive_skips=1))

    step(GOOD)
    with pytest.raises(FloatingPointError, match=r"global norm reached 0\.001"):
        step(GOOD)


def test_skips_indefinitely_without_a_tolerance():
    p, loss = poisonable_problem()
    step = compile_train(loss, apply_if_finite(sgd(0.1), max_consecutive_skips=None))

    weights = p.get_value().copy()
    for _ in range(20):
        step(BAD)

    np.testing.assert_allclose(p.get_value(), weights)


def test_clocks_advance_through_a_skipped_step():
    """A skipped step still consumed a step, and a frozen clock would stall every schedule reading it."""
    p, loss = poisonable_problem()
    step = compile_train(loss, apply_if_finite(adam(0.1)))
    clock = state_named(step, "adam/step_count")

    step(GOOD)
    step(BAD)

    assert int(clock.get_value()) == 2


def test_large_step_holds_back_an_outsized_but_finite_step():
    """The case `nonfinite` cannot catch: a step far too big to be healthy, whose every value is finite."""
    p, loss = poisonable_problem()
    total = scalar_state("skip_if/total_skips")
    step = compile_train(
        loss,
        skip_if(sgd(0.1), large_step(0.1), max_consecutive_skips=None, total_skips=total),
    )

    weights = p.get_value().copy()
    # sgd(0.1) on a gradient of 2 * p * x**2 = 2 gives a step of norm 0.2 * sqrt(2), over the threshold.
    step(GOOD)

    assert float(total.get_value()) == 1.0
    np.testing.assert_allclose(p.get_value(), weights)


def test_large_step_also_holds_back_a_nonfinite_step():
    """A NaN norm compares False against any threshold, so the condition has to be written as a negated
    `<` rather than a `>=` or the worst step of all would sail through."""
    p, loss = poisonable_problem()
    step = compile_train(loss, skip_if(sgd(0.1), large_step(1e6), max_consecutive_skips=None))

    weights = p.get_value().copy()
    step(BAD)

    np.testing.assert_allclose(p.get_value(), weights)


def test_defaults_to_the_finiteness_condition():
    p, loss = poisonable_problem()
    step = compile_train(loss, skip_if(sgd(0.1)))

    weights = p.get_value().copy()
    step(BAD)

    np.testing.assert_allclose(p.get_value(), weights)


def test_accepts_a_condition_of_the_callers_own():
    """A bare callable carries no reason of its own, so the error falls back to a generic phrase."""
    p, loss = poisonable_problem()

    def always(updates, parameters):
        return pt.as_tensor(np.array(True))

    step = compile_train(loss, skip_if(sgd(0.1), always, max_consecutive_skips=1))

    step(GOOD)
    with pytest.raises(FloatingPointError, match="met the skip condition"):
        step(GOOD)


def test_ignores_state_a_policy_seeds_with_infinity():
    """reduce_on_plateau holds an infinite best-loss until it has seen a full window, and writes that
    sentinel straight back while one fills. Checking every value the rule writes would read that as a bad
    step and freeze a perfectly healthy run -- and raise, once a window outlasts the tolerance."""
    p, loss = poisonable_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    total = scalar_state("skip_if/total_skips")
    rule = reduce_on_plateau(adam(scale * 0.01), scale, patience=2, accumulation_size=8)
    step = compile_train(loss, apply_if_finite(rule, total_skips=total))

    weights = p.get_value().copy()
    for _ in range(8):
        step(GOOD)

    assert float(total.get_value()) == 0.0
    assert not np.allclose(p.get_value(), weights)


def test_two_guards_in_one_step_keep_separate_counters():
    """Both guards allocate a counter, and two shared variables under one name would alias each other at
    the serialization boundary."""
    x = pt.vector("x")
    first = trainable(np.ones(2, dtype=config.floatX), name="w1")
    second = trainable(np.ones(2, dtype=config.floatX), name="w2")
    loss = ((first * x) ** 2).sum() + ((second * x) ** 2).sum()

    guarded_first = apply_if_finite(sgd(0.1), namespace="guard_w1")
    guarded_second = apply_if_finite(adam(0.01), namespace="guard_w2")

    def rule(loss_or_gradients, parameters):
        return {
            **guarded_first(loss_or_gradients, [first]),
            **guarded_second(loss_or_gradients, [second]),
        }

    step = compile_train(loss, rule, parameters=[first, second])
    first_total = state_named(step, "guard_w1/total_skips")
    second_total = state_named(step, "guard_w2/total_skips")
    assert first_total is not second_total

    step(BAD)

    # Both guards see the same poisoned batch, and each records the skip in its own counter.
    assert (float(first_total.get_value()), float(second_total.get_value())) == (1.0, 1.0)
    assert np.all(np.isfinite(first.get_value()))
    assert np.all(np.isfinite(second.get_value()))


def test_the_outer_of_two_nested_guards_judges_the_corrected_step():
    """Nesting is not two independent votes on the same step: the outer guard reads what the inner one
    already decided, so a step the inner throws away reaches the outer as a step of zero."""
    p, loss = poisonable_problem()
    rule = skip_if(
        skip_if(adam(0.01), nonfinite(), max_consecutive_skips=None, namespace="inner"),
        large_step(10.0),
        max_consecutive_skips=None,
        namespace="outer",
    )
    step = compile_train(loss, rule)
    inner_total = state_named(step, "inner/total_skips")
    outer_total = state_named(step, "outer/total_skips")

    step(GOOD)
    step(BAD)

    assert float(inner_total.get_value()) == 1.0
    assert float(outer_total.get_value()) == 0.0
    assert np.all(np.isfinite(p.get_value()))


def test_a_skipped_step_still_returns_its_loss():
    """The loss is the diagnostic that says why the step was thrown away, so the guard must not swallow
    it along with the update."""
    p, loss = poisonable_problem()
    step = compile_train(loss, apply_if_finite(sgd(0.1), max_consecutive_skips=None))

    assert np.isnan(float(step(BAD)))


def test_does_not_hold_back_statistics_the_model_writes():
    """The guard covers what the rule writes. A batch-norm running statistic is written by the model and
    folded in outside the rule, so a poisoned batch reaches it even though the parameters are spared."""
    X = pt.matrix("X")
    normalization = BatchNorm("bn", n_in=2)
    network = Sequential(Linear("fc", n_in=2, n_out=2), normalization)
    loss = (network(X) ** 2).sum()
    step = compile_train(loss, apply_if_finite(sgd(0.1), max_consecutive_skips=None))

    weights = state_named(step, "fc_W")
    before = weights.get_value().copy()
    step(np.full((4, 2), np.nan, dtype=config.floatX))

    assert np.all(np.isfinite(weights.get_value()))
    np.testing.assert_allclose(weights.get_value(), before)
    assert not np.all(np.isfinite(normalization.running_mean.get_value()))


def test_rejects_a_tolerance_below_one():
    with pytest.raises(ValueError, match="must be at least 1"):
        apply_if_finite(sgd(0.1), max_consecutive_skips=0)


def test_rejects_a_non_positive_norm():
    with pytest.raises(ValueError, match="must be positive"):
        large_step(0.0)


def test_rejects_a_rule_with_nothing_to_check():
    with pytest.raises(ValueError, match="no floating-point parameters"):
        nonfinite()({}, [])


def bounded_problem(max_norm_of):
    """A problem whose sgd step is outsized but finite, with its bound built from the batch so a caller
    can hand in either a literal or a shape-derived one."""
    x = pt.vector("x")
    p = trainable(np.ones(2, dtype=config.floatX), name="w")
    total = scalar_state("skip_if/total_skips")
    step = compile_train(
        ((p * x) ** 2).sum(),
        skip_if(
            sgd(0.1), large_step(max_norm_of(x)), max_consecutive_skips=None, total_skips=total
        ),
        inputs=[x],
    )
    return p, total, step


# sgd(0.1) on a gradient of 2 * p * x**2 = 2 gives a step of norm 0.2 * sqrt(2) ~ 0.283, which the first
# two bounds sit under and the third sits over.
@pytest.mark.parametrize(
    ("max_norm_of", "skipped"),
    [
        (lambda x: 0.1, True),
        (lambda x: x.shape[0] * 0.05, True),
        (lambda x: x.shape[0] * 1.0, False),
    ],
    ids=["literal-under", "shape-derived-under", "shape-derived-over"],
)
def test_a_shape_derived_bound_decides_like_its_literal_equivalent(max_norm_of, skipped):
    """A bound written as arithmetic on a shape has no value until the step runs, so it has to reach the
    graph rather than be refused for failing a comparison that cannot be made yet."""
    p, total, step = bounded_problem(max_norm_of)
    weights = p.get_value().copy()
    step(GOOD)

    assert float(total.get_value()) == (1.0 if skipped else 0.0)
    assert np.allclose(p.get_value(), weights) == skipped


def test_a_shape_derived_bound_is_checked_when_the_step_runs():
    """The build-time check cannot fire on a bound with no value yet, so it travels with the graph. The
    jax backend drops assertions, which is why this is the best that can be offered rather than a
    guarantee."""
    _, _, step = bounded_problem(lambda x: x.shape[0] * 0.0)

    with pytest.raises(ValueError, match="max_norm is a norm to compare against"):
        step(GOOD)


def test_a_bound_that_is_not_a_single_number_is_refused():
    with pytest.raises(ValueError, match="max_norm must be a single number"):
        large_step(pt.matrix("X").shape)
