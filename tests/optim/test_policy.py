from itertools import pairwise

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor import config

from pytensor_ml.optim import adam, compile_train, reduce_on_plateau, scalar_state, sgd
from pytensor_ml.params import trainable

RTOL = 1e-6


def plateaued_problem():
    """A parameter parked at the optimum, so the loss is zero and can never improve again. Every step after
    the first is a bad one, which is what makes the policy's counting observable."""
    p = trainable(np.zeros(1, dtype=config.floatX), name="w")
    return p, 0.5 * (p**2).sum()


def run(rule, loss, scale, n_steps):
    """Return the scale after each of ``n_steps`` steps."""
    step = compile_train(loss, rule, inputs=[])
    scales = []
    for _ in range(n_steps):
        step()
        scales.append(float(scale.get_value()))
    return scales


def test_cuts_after_exactly_patience_steps_without_improvement():
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=3)

    # The first step improves on the initial infinite best, so counting starts after it, and each cut needs
    # a fresh `patience` bad steps rather than following on from the last.
    assert run(rule, loss, scale, 8) == [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25]


def test_cools_down_before_counting_again():
    """Without a cooldown the count resets on the cut and immediately starts toward the next, which is how
    a noisy loss cuts dozens of times in a short run."""
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(
        adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=3, cooldown=2
    )

    scales = run(rule, loss, scale, 11)

    # Cuts land patience + cooldown apart rather than every patience steps.
    assert scales == [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25]


def test_never_goes_below_the_floor():
    """A loss that never improves cuts forever; unfloored the scale underflows and training stops without
    any error to say so."""
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(
        adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=1, min_scale=0.2
    )

    assert min(run(rule, loss, scale, 20)) == pytest.approx(0.2, rel=RTOL)


def test_an_improvement_resets_the_count():
    """The parameter starts away from the optimum, so adam improves the loss every step and nothing ever
    counts as bad. A policy that counted regardless would cut a run that is going fine."""
    p = trainable(np.array([2.0], dtype=config.floatX), name="w")
    loss = 0.5 * (p**2).sum()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=3)

    assert set(run(rule, loss, scale, 20)) == {1.0}


def test_an_improvement_too_small_to_matter_does_not_reset_the_count():
    """`rtol` is what makes this usable on a per-batch loss: without it any improvement at all, however
    tiny, resets the count, so a loss drifting down by a rounding error never plateaus."""
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    # A relative tolerance of 1 demands the loss halve to count, which zero never does after the first step.
    rule = reduce_on_plateau(
        adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=2, rtol=1.0
    )

    assert run(rule, loss, scale, 3)[-1] == pytest.approx(0.5, rel=RTOL)


def test_a_noisy_loss_cuts_itself_into_the_ground_without_a_floor_and_a_cooldown():
    """The trajectory that made both parameters non-optional. A per-batch loss stops improving constantly,
    so with neither guard the policy cuts on almost every patience window and the scale underflows -- the run
    stops training with nothing raised to say so."""

    def noisy_run(cooldown, min_scale, n_steps=120):
        p = trainable(np.zeros(1, dtype=config.floatX), name="w")
        X = pt.tensor("X", shape=(None,))
        loss = ((p - X) ** 2).mean()
        scale = scalar_state("plateau/scale", fill_value=1.0)
        rule = reduce_on_plateau(
            adam(learning_rate=scale * 0.05),
            scale,
            factor=0.5,
            patience=3,
            cooldown=cooldown,
            min_scale=min_scale,
        )
        step = compile_train(loss, rule, inputs=[X])
        rng = np.random.default_rng(0)
        scales = []
        for _ in range(n_steps):
            step(rng.normal(size=8).astype(config.floatX))
            scales.append(float(scale.get_value()))
        cuts = sum(1 for before, after in pairwise(scales) if after < before)
        return cuts, min(scales)

    unguarded_cuts, unguarded_floor = noisy_run(cooldown=0, min_scale=0.0)
    assert unguarded_cuts > 30
    assert unguarded_floor < 1e-9  # a rate this small is not training anything

    guarded_cuts, guarded_floor = noisy_run(cooldown=10, min_scale=0.01)
    assert guarded_cuts < unguarded_cuts / 3
    assert guarded_floor == pytest.approx(0.01, rel=RTOL)


def test_rejects_precomputed_gradients():
    """The policy decides from the loss, and `loss_or_gradients` is a union, so a caller passing gradients
    would otherwise hand it a list and get a confusing failure deep in the comparison."""
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(sgd(learning_rate=scale * 0.05), scale)

    with pytest.raises(ValueError, match="needs the loss graph"):
        rule([pytensor.gradient.grad(loss, p)], [p])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"factor": 1.0}, "factor must lie in"),
        ({"factor": 0.0}, "factor must lie in"),
        ({"rtol": -0.1}, "must be non-negative"),
        ({"rtol": 2.0}, "at most 1.0"),
        ({"patience": 0}, "at least 1"),
        ({"cooldown": -1}, "non-negative"),
        ({"accumulation_size": 0}, "at least 1"),
    ],
    ids=[
        "factor_one",
        "factor_zero",
        "negative_rtol",
        "rtol_above_one",
        "no_patience",
        "negative_cooldown",
        "no_window",
    ],
)
def test_rejects_settings_that_cannot_work(kwargs, message):
    scale = scalar_state("plateau/scale", fill_value=1.0)

    with pytest.raises(ValueError, match=message):
        reduce_on_plateau(sgd(learning_rate=scale), scale, **kwargs)


def test_a_window_holds_everything_until_it_fills():
    """Nothing advances mid-window: with a window of four, a plateaued loss takes four times as many steps
    to reach a cut, because only every fourth step is a decision."""
    p, loss = plateaued_problem()
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(
        adam(learning_rate=scale * 0.05), scale, factor=0.5, patience=2, accumulation_size=4
    )

    scales = run(rule, loss, scale, 12)

    # Two decisions to fill patience, four steps each, and the first decision is the improvement on the
    # initial infinite best -- so the cut lands on the twelfth step rather than the third.
    assert scales == [1.0] * 11 + [0.5]


def test_deciding_on_a_window_mean_survives_a_noisy_loss():
    """What the window is for. The same per-batch loss that cuts itself into the ground step by step is
    judged on an average, so the noise that looked like a plateau every few steps mostly averages out."""

    def noisy_cuts(accumulation_size, n_steps=120):
        p = trainable(np.zeros(1, dtype=config.floatX), name="w")
        X = pt.tensor("X", shape=(None,))
        loss = ((p - X) ** 2).mean()
        scale = scalar_state("plateau/scale", fill_value=1.0)
        rule = reduce_on_plateau(
            adam(learning_rate=scale * 0.05),
            scale,
            factor=0.5,
            patience=3,
            accumulation_size=accumulation_size,
        )
        step = compile_train(loss, rule, inputs=[X])
        rng = np.random.default_rng(0)
        scales = []
        for _ in range(n_steps):
            step(rng.normal(size=8).astype(config.floatX))
            scales.append(float(scale.get_value()))
        return sum(1 for before, after in pairwise(scales) if after < before)

    assert noisy_cuts(accumulation_size=8) < noisy_cuts(accumulation_size=1) / 4


def test_the_decision_is_made_on_the_window_mean_not_its_last_loss():
    """The substance of the window: an average, not a sample. Given no parameters the rule moves nothing, so
    the loss is exactly what is fed and the policy's own state is all that changes."""
    loss = pt.scalar("loss")
    scale = scalar_state("plateau/scale", fill_value=1.0)
    rule = reduce_on_plateau(adam(learning_rate=scale), scale, accumulation_size=2)
    best_loss = next(v for v in rule(loss, []) if v.name == "plateau/best_loss")
    step = compile_train(loss, rule, parameters=[], inputs=[loss])

    step(np.asarray(4.0, dtype=config.floatX))
    step(np.asarray(0.0, dtype=config.floatX))

    # Deciding on the window's last loss would record 0.0 as the best seen, and every later window would
    # then have an unreachable target to beat.
    assert float(best_loss.get_value()) == pytest.approx(2.0, rel=RTOL)


@pytest.mark.parametrize(
    "kwargs, complaint",
    [
        ({"patience": 0}, "patience is a number of steps"),
        ({"patience": pt.as_tensor_variable(2) * 0}, "patience is a number of steps"),
        ({"cooldown": -1}, "cooldown is a number of steps"),
        ({"accumulation_size": 0}, "accumulation_size is a number of losses"),
    ],
    ids=["patience", "patience-folded", "cooldown", "accumulation_size"],
)
def test_a_count_that_folds_to_a_constant_is_refused_at_build_time(kwargs, complaint):
    """A count written as arithmetic on constants still folds, so it is checked like a bare one rather
    than deferred to the graph."""
    with pytest.raises(ValueError, match=complaint):
        reduce_on_plateau(adam(1e-3), scalar_state("scale", fill_value=1.0), **kwargs)


@pytest.mark.parametrize(
    ("count_name", "literal", "training_steps"),
    [("patience", 2, 9), ("accumulation_size", 4, 20)],
    ids=["patience", "accumulation_size"],
)
def test_a_count_taken_from_a_shape_behaves_like_its_literal_equivalent(
    count_name, literal, training_steps
):
    """A count is naturally written as arithmetic on a shape, which has no value until the function
    runs. Reaching the graph is not enough: it then has to decide exactly as the number it stands
    for."""
    X = pt.matrix("X")
    batch = np.zeros(
        (1, 3), dtype=config.floatX
    )  # One row, so `literal * X.shape[0]` is `literal`.

    scales = []
    for label, count in (("literal", literal), ("symbolic", literal * X.shape[0])):
        parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
        scale = scalar_state(f"{count_name}_{label}", fill_value=1.0)
        settings = {"factor": 0.1, "patience": 2}
        settings[count_name] = count
        rule = reduce_on_plateau(adam(scale * 1e-3), scale, **settings)
        step = compile_train(((parameter - X.sum()) ** 2).sum(), rule, inputs=[X])
        for _ in range(training_steps):
            step(batch)
        scales.append(float(scale.get_value()))

    assert scales[0] == scales[1]
    # Two runs that never cut would match each other while proving nothing about the count.
    assert scales[0] < 1.0


@pytest.mark.parametrize(
    ("count_name", "count_of", "complaint"),
    [
        ("patience", lambda X: X.shape[0], "patience is a number of steps"),
        ("cooldown", lambda X: X.shape[0] - 1, "cooldown is a number of steps"),
        ("accumulation_size", lambda X: X.shape[0], "accumulation_size is a number of losses"),
    ],
    ids=["patience", "cooldown", "accumulation_size"],
)
def test_a_shape_derived_count_is_checked_when_the_step_runs(count_name, count_of, complaint):
    """The build-time check cannot fire on a count with no value yet, so it travels with the graph. The
    jax backend drops assertions, which is why this is the best that can be offered rather than a
    guarantee."""
    X = pt.matrix("X")
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    scale = scalar_state(f"runtime_scale_{count_name}", fill_value=1.0)
    rule = reduce_on_plateau(adam(scale * 1e-3), scale, factor=0.1, **{count_name: count_of(X)})
    step = compile_train(((parameter - X.sum()) ** 2).sum(), rule, inputs=[X])

    with pytest.raises(ValueError, match=complaint):
        step(np.zeros((0, 3), dtype=config.floatX))


@pytest.mark.parametrize(
    "patience",
    [
        pt.matrix("X").shape,
        pt.vector("v", dtype="int64"),
        np.array([2, 3]),
        pt.matrix("m", dtype="int64"),
    ],
    ids=["shape-tuple", "vector", "numpy-array", "matrix"],
)
def test_a_count_that_is_not_a_single_number_is_refused(patience):
    """``X.shape`` for ``X.shape[0]`` is the easy slip. Left to the graph it surfaces as a bare
    ``AssertionError`` from a scalar conversion, naming neither the argument nor the mistake."""
    with pytest.raises(ValueError, match="patience must be a single number"):
        reduce_on_plateau(adam(1e-3), scalar_state("scale", fill_value=1.0), patience=patience)
