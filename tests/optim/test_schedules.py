import numpy as np
import pytest

from pytensor import config
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor import lscalar, matrix, vector

from pytensor_ml.optim import (
    chain,
    compile_train,
    constant_schedule,
    cosine_schedule,
    exponential_schedule,
    join_schedules,
    linear_onecycle_schedule,
    linear_schedule,
    polynomial_schedule,
    scale,
    sgd,
    step_decay,
)
from pytensor_ml.params import step_counter, trainable
from pytensor_ml.pytensorf import function

# Every schedule takes (learning_rate, total_steps, final_learning_rate) and owes the same contract at the
# horizon, so those properties are asserted once for all of them.
SCHEDULES = [cosine_schedule, linear_schedule, exponential_schedule, polynomial_schedule]


def evaluate_schedule(schedule, steps):
    step_count = lscalar("step_count")
    rate_at = function([step_count], schedule(step_count))
    return np.array([rate_at(step) for step in steps])


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_decreases_monotonically(schedule_factory):
    rates = evaluate_schedule(schedule_factory(0.5, 10, final_learning_rate=0.05), range(11))
    assert np.all(np.diff(rates) < 0.0)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_accepts_single_step_horizon(schedule_factory):
    rates = evaluate_schedule(schedule_factory(0.5, 1, final_learning_rate=0.05), [0, 1])
    np.testing.assert_allclose(rates, [0.5, 0.05], rtol=1e-6)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_holds_its_endpoint_past_total_steps(schedule_factory):
    rates = evaluate_schedule(schedule_factory(1.0, 4, final_learning_rate=0.25), [4, 5, 100])
    np.testing.assert_allclose(rates, 0.25, rtol=1e-6)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_reads_floatX_at_graph_build_time(schedule_factory):
    schedule = schedule_factory(1e-3, 10, 1e-5)
    with config.change_flags(floatX="float32"):
        assert schedule(lscalar("step_count")).type.dtype == "float32"
    with config.change_flags(floatX="float64"):
        assert schedule(lscalar("step_count")).type.dtype == "float64"


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_accepts_an_equal_endpoint_as_a_constant_rate(schedule_factory):
    # Equal endpoints are a flat schedule rather than a rejected input.
    rates = evaluate_schedule(schedule_factory(0.1, 4, final_learning_rate=0.1), range(6))
    np.testing.assert_allclose(rates, 0.1, rtol=1e-6)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
@pytest.mark.parametrize("total_steps", [0, -1])
def test_schedule_rejects_empty_horizon(schedule_factory, total_steps):
    with pytest.raises(ValueError, match="total_steps must be at least 1"):
        schedule_factory(1e-3, total_steps, 1e-5)


def test_cosine_schedule_hits_curve_anchor_points():
    # A non-zero floor exercises the affine map onto [final_learning_rate, learning_rate]: the half cosine
    # runs 1 -> 0.5 -> 0, so the rate runs 1.0 -> 0.625 -> 0.25.
    rates = evaluate_schedule(cosine_schedule(1.0, 4, final_learning_rate=0.25), [0, 2, 4])
    np.testing.assert_allclose(rates, [1.0, 0.625, 0.25], rtol=1e-6)


def test_linear_schedule_falls_by_a_constant_amount():
    # The constant decrement is what separates linear from every other decay: 0.75 spread over 4 steps.
    rates = evaluate_schedule(linear_schedule(1.0, 4, final_learning_rate=0.25), range(5))
    np.testing.assert_allclose(np.diff(rates), -0.1875, rtol=1e-6)
    np.testing.assert_allclose(rates[[0, -1]], [1.0, 0.25], rtol=1e-6)


def test_linear_schedule_holds_the_initial_rate_until_transition_begin():
    # total_steps is the length of the decay itself, so the floor arrives at transition_begin + total_steps.
    schedule = linear_schedule(1.0, 4, final_learning_rate=0.25, transition_begin=3)
    rates = evaluate_schedule(schedule, [0, 3, 4, 5, 7, 100])
    np.testing.assert_allclose(rates, [1.0, 1.0, 0.8125, 0.625, 0.25, 0.25], rtol=1e-6)


def test_linear_onecycle_schedule_hits_each_phase_boundary():
    schedule = linear_onecycle_schedule(1.0, total_steps=20)
    rates = evaluate_schedule(schedule, [0, 6, 17, 20, 100])
    np.testing.assert_allclose(rates, [0.04, 1.0, 0.04, 4e-6, 4e-6], rtol=1e-6)


def test_linear_onecycle_schedule_interpolates_custom_phases():
    schedule = linear_onecycle_schedule(
        1.0, total_steps=20, pct_start=0.25, pct_final=0.75, div_factor=10, final_div_factor=100
    )
    rates = evaluate_schedule(schedule, [0, 2, 5, 10, 15, 17, 20])
    np.testing.assert_allclose(rates, [0.1, 0.46, 1.0, 0.55, 0.1, 0.0604, 0.001], rtol=1e-6)


@pytest.mark.parametrize("total_steps", [0, -1])
def test_linear_onecycle_schedule_rejects_empty_horizon(total_steps):
    with pytest.raises(ValueError, match="total_steps must be at least 1"):
        linear_onecycle_schedule(1.0, total_steps)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pct_start": 0.0},
        {"pct_start": 0.9, "pct_final": 0.8},
        {"pct_final": 1.0},
    ],
)
def test_linear_onecycle_schedule_rejects_invalid_phase_fractions(kwargs):
    with pytest.raises(ValueError, match="0 < pct_start < pct_final < 1"):
        linear_onecycle_schedule(1.0, 20, **kwargs)


@pytest.mark.parametrize("name", ["div_factor", "final_div_factor"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan])
def test_linear_onecycle_schedule_rejects_non_positive_divisors(name, value):
    with pytest.raises(ValueError, match=rf"{name} must be positive"):
        linear_onecycle_schedule(1.0, 20, **{name: value})


@pytest.mark.parametrize("kwargs", [{}, {"pct_start": 0.31, "pct_final": 0.39}])
def test_linear_onecycle_schedule_rejects_empty_rounded_phases(kwargs):
    with pytest.raises(ValueError, match="three phases of at least one step after rounding"):
        linear_onecycle_schedule(1.0, 2 if not kwargs else 10, **kwargs)


@pytest.mark.parametrize(
    "schedule_factory", [linear_schedule, exponential_schedule, polynomial_schedule]
)
def test_schedule_holds_the_initial_rate_until_transition_begin(schedule_factory):
    """Each schedule has to pass its own `transition_begin` through, so the shared clamp being right is not
    enough — asserted as a property because the curve between B and B + T differs per schedule."""
    rates = evaluate_schedule(
        schedule_factory(1.0, 4, 0.25, transition_begin=3), [0, 1, 3, 4, 7, 100]
    )

    np.testing.assert_allclose(rates[:3], 1.0, rtol=1e-6)  # held through step B
    assert rates[3] < 1.0  # decaying afterwards
    np.testing.assert_allclose(rates[4:], 0.25, rtol=1e-6)  # floor from step B + T


def test_polynomial_schedule_bends_with_the_power():
    # (1 - p) ** 2 over four steps: quick drop, then a long flat tail.
    rates = evaluate_schedule(polynomial_schedule(1.0, 4, power=2.0), range(5))
    np.testing.assert_allclose(rates, [1.0, 0.5625, 0.25, 0.0625, 0.0], rtol=1e-6)


def test_polynomial_schedule_at_power_one_matches_linear_schedule():
    """`linear_schedule` is the `power=1` case, and the two are separate implementations, so pin the identity
    rather than trusting that they stay in step."""
    steps = range(6)
    polynomial = evaluate_schedule(
        polynomial_schedule(1.0, 4, final_learning_rate=0.25, power=1.0), steps
    )
    linear = evaluate_schedule(linear_schedule(1.0, 4, final_learning_rate=0.25), steps)
    np.testing.assert_allclose(polynomial, linear, rtol=1e-6)


def test_polynomial_schedule_reaches_the_floor_at_a_fractional_power():
    # A power below one holds the rate high and then drops, and raises exactly zero to a fractional
    # exponent at the horizon, which is where a NaN would appear.
    rates = evaluate_schedule(polynomial_schedule(1.0, 4, 0.25, power=0.5), [0, 2, 4, 100])
    np.testing.assert_allclose(rates, [1.0, 0.75 * 2**-0.5 + 0.25, 0.25, 0.25], rtol=1e-6)
    assert rates[1] > 0.625  # above the straight line, which passes through 0.625 at the midpoint


@pytest.mark.parametrize(
    "schedule_factory", [linear_schedule, exponential_schedule, polynomial_schedule]
)
def test_delayed_schedules_agree_on_their_positional_arguments(schedule_factory):
    """Copying a call from one schedule to another must not change what it means, so the fourth positional
    argument is `transition_begin` in every schedule that takes a delay — not, say, a curve parameter."""
    steps = [0, 3, 4, 7]
    positional = evaluate_schedule(schedule_factory(1.0, 4, 0.25, 3), steps)
    keyword = evaluate_schedule(schedule_factory(1.0, 4, 0.25, transition_begin=3), steps)
    np.testing.assert_allclose(positional, keyword, rtol=1e-6)


@pytest.mark.parametrize("power", [0.0, -1.0])
def test_polynomial_schedule_rejects_a_non_positive_power(power):
    with pytest.raises(ValueError, match="power must be positive"):
        polynomial_schedule(0.1, 4, power=power)


def test_exponential_schedule_falls_by_a_constant_factor():
    # The constant ratio is what separates geometric decay from linear: 1.0 -> 0.0625 over 4 steps halves.
    rates = evaluate_schedule(exponential_schedule(1.0, 4, 0.0625), range(5))
    np.testing.assert_allclose(rates[1:] / rates[:-1], 0.5, rtol=1e-6)
    np.testing.assert_allclose(rates[[0, -1]], [1.0, 0.0625], rtol=1e-6)


@pytest.mark.parametrize(
    ("learning_rate", "final_learning_rate"),
    [(0.1, 0.0), (0.0, 0.0)],
    ids=["zero_floor", "zero_initial"],
)
def test_exponential_schedule_rejects_a_non_positive_rate(learning_rate, final_learning_rate):
    with pytest.raises(ValueError, match="needs positive rates"):
        exponential_schedule(learning_rate, 4, final_learning_rate)


@pytest.mark.parametrize(
    "schedule_factory", [linear_schedule, exponential_schedule, polynomial_schedule]
)
def test_schedule_rejects_negative_transition_begin(schedule_factory):
    with pytest.raises(ValueError, match="transition_begin must not be negative"):
        schedule_factory(1e-3, 10, 1e-5, transition_begin=-1)


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_schedule_drives_training_through_the_learning_rate_union(schedule_factory):
    """Passing a schedule as `learning_rate` reads it off the clock the rule counts on, which is the path a
    new schedule is most likely to miss."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p, so the unit-rate base step is -p
    rule = sgd(learning_rate=schedule_factory(0.1, 2, 0.01))

    compile_train(loss, rule)()  # every schedule starts at its initial rate, so p = 2 - 0.1 * 2
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)


# Geometric decay never reaches zero, so the schedule that cannot hit a zero floor sits this one out.
@pytest.mark.parametrize("schedule_factory", [cosine_schedule, linear_schedule])
def test_schedule_drives_training_through_a_scaled_chain(schedule_factory):
    # Decaying to zero over two steps, every curve passes through the same midpoint, so one set of expected
    # values covers them: 0.1 -> 0.05 -> 0.
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p, so the unit-rate base step is -p
    clock = step_counter()
    rule = chain(sgd(learning_rate=1.0), scale(schedule_factory(0.1, 2)(clock)))
    step = compile_train(loss, rule)

    step()  # step 0 -> lr = 0.1, p = 2 - 0.1 * 2 = 1.8
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)
    step()  # step 1 -> lr = 0.05, p = 1.8 - 0.05 * 1.8 = 1.71
    np.testing.assert_allclose(p.get_value(), [1.71], rtol=1e-6)
    step()  # step 2 -> lr = 0, p unchanged
    np.testing.assert_allclose(p.get_value(), [1.71], rtol=1e-6)


def test_schedule_drives_training_when_the_caller_assembles_the_updates():
    """A caller who builds the updates by hand and compiles them with :func:`function`, rather than going
    through :func:`compile_train`, still gets a clock that advances -- otherwise every step reads the
    initial rate, and under a warmup that rate is zero, so the parameter never moves at all."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p, so the unit-rate base step is -p
    clock = step_counter(name="schedule/step_count")
    warmup = linear_schedule(0.0, total_steps=2, final_learning_rate=0.1)
    updates = scale(warmup(clock))(sgd(learning_rate=1.0)(loss, [p]), [p])

    step = function([], loss, updates=updates)

    step()  # step 0 -> lr = 0, the warmup holds the parameter still
    np.testing.assert_allclose(p.get_value(), [2.0], rtol=1e-6)
    step()  # step 1 -> lr = 0.05, p = 2 - 0.05 * 2 = 1.9
    np.testing.assert_allclose(p.get_value(), [1.9], rtol=1e-6)
    assert int(clock.get_value()) == 2


@pytest.mark.parametrize(
    "schedule_factory",
    [cosine_schedule, linear_schedule, exponential_schedule, polynomial_schedule],
)
def test_schedule_ramps_up_when_the_endpoint_is_higher(schedule_factory):
    """The endpoint is an endpoint, not a floor, so the same schedules express warmup. This is why there is
    no separate warmup function."""
    rates = evaluate_schedule(schedule_factory(1e-4, 4, 1e-2), [0, 1, 2, 3, 4, 100])

    # Endpoints two orders of magnitude apart: pytensor refactors the interpolation while fusing it, so
    # under floatX=float32 the ends land within rounding rather than exactly.
    np.testing.assert_allclose(rates[0], 1e-4, rtol=1e-5)
    assert np.all(np.diff(rates[:5]) > 0.0)
    np.testing.assert_allclose(rates[4:], 1e-2, rtol=1e-5)


def test_join_schedules_gives_each_segment_its_own_step_count():
    """Warmup into cosine decay, the standard recipe: the cosine has to start from its own step zero at the
    boundary rather than partway down its curve."""
    recipe = join_schedules([linear_schedule(0.0, 3, 1.0), cosine_schedule(1.0, 6)], [3])
    rates = evaluate_schedule(recipe, range(10))

    np.testing.assert_allclose(
        rates[:4], [0.0, 1 / 3, 2 / 3, 1.0], rtol=1e-6
    )  # ramp, peaking at the handoff
    np.testing.assert_allclose(rates[3], 1.0, rtol=1e-6)  # cosine restarts at its initial rate
    np.testing.assert_allclose(rates[9], 0.0, rtol=1e-6)  # and reaches its endpoint 6 steps later
    assert np.all(np.diff(rates[3:10]) < 0.0)


def test_join_schedules_switches_at_every_boundary():
    joined = join_schedules(
        [constant_schedule(1.0), constant_schedule(0.5), constant_schedule(0.25)], [2, 5]
    )
    rates = evaluate_schedule(joined, range(7))
    np.testing.assert_allclose(rates, [1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25], rtol=1e-6)


def test_join_schedules_with_one_schedule_is_that_schedule():
    steps = range(6)
    joined = evaluate_schedule(join_schedules([linear_schedule(1.0, 4)], []), steps)
    alone = evaluate_schedule(linear_schedule(1.0, 4), steps)
    np.testing.assert_allclose(joined, alone, rtol=1e-6)


@pytest.mark.parametrize(
    "schedule",
    [
        *(factory(0.5, 4, 0.05) for factory in SCHEDULES),
        step_decay(0.5, decay_every=2),
        constant_schedule(0.5),
    ],
    ids=["cosine", "linear", "exponential", "polynomial", "step", "constant"],
)
def test_schedule_tolerates_a_negative_step_count(schedule):
    """`join_schedules` evaluates every segment and selects, so a later schedule is called with a negative
    count before its boundary. Each one has to return something finite rather than a NaN."""
    rates = evaluate_schedule(schedule, [-100, -1])
    assert np.all(np.isfinite(rates))
    np.testing.assert_allclose(rates, rates[0], rtol=1e-6)  # held at the pre-horizon value


def test_join_schedules_drives_training_through_the_learning_rate_union():
    """The composed schedule has to survive substitution into a rule, which is how anyone would actually
    use warmup followed by decay."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()  # grad = p, so the unit-rate base step is -p
    recipe = join_schedules([constant_schedule(0.1), linear_schedule(0.1, 2)], [2])
    step = compile_train(loss, sgd(learning_rate=recipe))

    step()  # constant segment: p = 2 - 0.1 * 2
    np.testing.assert_allclose(p.get_value(), [1.8], rtol=1e-6)
    step()
    step()  # boundary passed, linear segment at its own step 0, still 0.1
    np.testing.assert_allclose(p.get_value(), [1.8 * 0.9 * 0.9], rtol=1e-6)


def test_constant_schedule_holds_its_rate():
    rates = evaluate_schedule(constant_schedule(0.25), [0, 1, 10_000])
    np.testing.assert_allclose(rates, 0.25, rtol=1e-6)


@pytest.mark.parametrize(
    ("schedules", "boundaries", "message"),
    [
        ([], [], "at least one schedule"),
        ([constant_schedule(1.0), constant_schedule(0.5)], [], "one fewer boundary"),
        ([constant_schedule(1.0), constant_schedule(0.5)], [0], "must be positive"),
        (
            [constant_schedule(1.0), constant_schedule(0.5), constant_schedule(0.25)],
            [5, 2],
            "strictly increasing",
        ),
    ],
    ids=["empty", "boundary_count", "non_positive", "not_increasing"],
)
def test_join_schedules_rejects_a_malformed_sequence(schedules, boundaries, message):
    with pytest.raises(ValueError, match=message):
        join_schedules(schedules, boundaries)


# step_decay decays indefinitely instead of over a horizon, so it takes (decay_every, decay_factor) rather
# than (total_steps, min_learning_rate) positionally and sits outside the shared contract tests above.
def test_step_decay_drops_by_a_factor_at_each_interval():
    rates = evaluate_schedule(step_decay(1.0, decay_every=3, decay_factor=0.5), range(10))
    expected = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.125]
    np.testing.assert_allclose(rates, expected, rtol=1e-6)


def test_step_decay_stops_at_the_floor():
    schedule = step_decay(1.0, decay_every=3, decay_factor=0.5, min_learning_rate=0.3)
    rates = evaluate_schedule(schedule, [3, 6, 9, 100])
    np.testing.assert_allclose(rates, [0.5, 0.3, 0.3, 0.3], rtol=1e-6)


def test_step_decay_delays_the_first_drop_until_transition_begin():
    schedule = step_decay(1.0, decay_every=3, decay_factor=0.5, transition_begin=2)
    rates = evaluate_schedule(schedule, [0, 4, 5, 8])
    np.testing.assert_allclose(rates, [1.0, 1.0, 0.5, 0.25], rtol=1e-6)


def test_step_decay_underflows_to_zero_without_a_floor():
    # A long run multiplies the factor thousands of times; the result has to reach zero rather than a NaN.
    rate = evaluate_schedule(step_decay(1.0, decay_every=1, decay_factor=0.5), [10_000])
    np.testing.assert_array_equal(rate, [0.0])


def test_step_decay_requires_keyword_arguments():
    """The positional slots mean something different here than in the horizon-based schedules, so passing
    them positionally is an error rather than a silent reinterpretation."""
    with pytest.raises(TypeError, match="positional"):
        step_decay(1.0, 3)  # type: ignore[misc]


def test_step_decay_reads_floatX_at_graph_build_time():
    # Not covered by the shared contract test, and this is the one schedule with an integer intermediate
    # (the drop count), so it is the likeliest to leak a float64 rate into a float32 graph.
    schedule = step_decay(1e-3, decay_every=10, min_learning_rate=1e-5)
    with config.change_flags(floatX="float32"):
        assert schedule(lscalar("step_count")).type.dtype == "float32"
    with config.change_flags(floatX="float64"):
        assert schedule(lscalar("step_count")).type.dtype == "float64"


def test_step_decay_accepts_a_unit_factor_as_a_constant_rate():
    # The factor check admits 1.0, which is a flat schedule rather than a rejected input.
    rates = evaluate_schedule(step_decay(0.1, decay_every=3, decay_factor=1.0), [0, 3, 30])
    np.testing.assert_allclose(rates, 0.1, rtol=1e-6)


@pytest.mark.parametrize("decay_every", [0, -1])
def test_step_decay_rejects_a_non_positive_interval(decay_every):
    with pytest.raises(ValueError, match="decay_every must be at least 1"):
        step_decay(1.0, decay_every=decay_every)


@pytest.mark.parametrize("decay_factor", [0.0, -0.5, 1.5])
def test_step_decay_rejects_a_factor_outside_the_unit_interval(decay_factor):
    with pytest.raises(ValueError, match=r"decay_factor must be in \(0, 1]"):
        step_decay(1.0, decay_every=3, decay_factor=decay_factor)


SHAPE_SOURCE = matrix("shape_source")
HORIZON = 10
HORIZON_STEPS = [0, 3, 7, 10, 20]


# A horizon is naturally written as arithmetic on a shape -- `X.shape[0]` for one epoch -- which has no
# value until the function runs, so it has to reach the graph rather than be refused for failing a
# comparison that cannot be made yet.
def compile_shape_derived_schedule(schedule):
    # `on_unused_input`: a schedule whose count folds away leaves SHAPE_SOURCE unconsumed, and the
    # resulting UnusedInputError would look nothing like the mistake that caused it.
    step_count = lscalar("step_count")
    return function([step_count, SHAPE_SOURCE], schedule(step_count), on_unused_input="ignore")


def evaluate_shape_derived_schedule(schedule, steps, rows):
    rate_at = compile_shape_derived_schedule(schedule)
    batch = np.zeros((rows, 2), dtype=config.floatX)
    return np.array([rate_at(step, batch) for step in steps])


@pytest.mark.parametrize("schedule_factory", SCHEDULES)
def test_shape_derived_horizon_matches_its_literal_equivalent(schedule_factory):
    literal = evaluate_schedule(
        schedule_factory(0.5, HORIZON, final_learning_rate=0.05), HORIZON_STEPS
    )
    symbolic = evaluate_shape_derived_schedule(
        schedule_factory(0.5, SHAPE_SOURCE.shape[0], final_learning_rate=0.05),
        HORIZON_STEPS,
        rows=HORIZON,
    )
    np.testing.assert_allclose(literal, symbolic, rtol=1e-6)
    # Both sides come from the same code path, so a rate that never moves would match itself.
    assert np.ptp(symbolic) > 0.0


def test_shape_derived_transition_begin_matches_its_literal_equivalent():
    literal = evaluate_schedule(
        linear_schedule(0.5, HORIZON, final_learning_rate=0.05, transition_begin=3), HORIZON_STEPS
    )
    symbolic = evaluate_shape_derived_schedule(
        linear_schedule(
            0.5, HORIZON, final_learning_rate=0.05, transition_begin=SHAPE_SOURCE.shape[0] - 7
        ),
        HORIZON_STEPS,
        rows=HORIZON,
    )
    np.testing.assert_allclose(literal, symbolic, rtol=1e-6)


def test_shape_derived_decay_interval_matches_its_literal_equivalent():
    literal = evaluate_schedule(
        step_decay(1.0, decay_every=HORIZON, decay_factor=0.5), HORIZON_STEPS
    )
    symbolic = evaluate_shape_derived_schedule(
        step_decay(1.0, decay_every=SHAPE_SOURCE.shape[0], decay_factor=0.5),
        HORIZON_STEPS,
        rows=HORIZON,
    )
    np.testing.assert_allclose(literal, symbolic, rtol=1e-6)


def test_shape_derived_boundary_matches_its_literal_equivalent():
    segments = [constant_schedule(1.0), constant_schedule(0.25)]
    literal = evaluate_schedule(join_schedules(segments, boundaries=[HORIZON]), HORIZON_STEPS)
    symbolic = evaluate_shape_derived_schedule(
        join_schedules(segments, boundaries=[SHAPE_SOURCE.shape[0]]), HORIZON_STEPS, rows=HORIZON
    )
    np.testing.assert_allclose(literal, symbolic, rtol=1e-6)


@pytest.mark.parametrize(
    ("build_schedule", "rows", "message"),
    [
        (lambda: linear_schedule(0.5, SHAPE_SOURCE.shape[0]), 0, "total_steps must be at least 1"),
        (
            lambda: linear_schedule(0.5, 10, transition_begin=SHAPE_SOURCE.shape[0] - 5),
            0,
            "transition_begin must not be negative",
        ),
        (
            lambda: step_decay(1.0, decay_every=SHAPE_SOURCE.shape[0]),
            0,
            "decay_every must be at least 1",
        ),
        (
            lambda: join_schedules(
                [constant_schedule(1.0), constant_schedule(0.25)],
                boundaries=[SHAPE_SOURCE.shape[0]],
            ),
            0,
            r"boundaries\[0\] must be positive",
        ),
        (
            lambda: join_schedules(
                [constant_schedule(1.0), constant_schedule(0.5), constant_schedule(0.25)],
                boundaries=[5, SHAPE_SOURCE.shape[0]],
            ),
            2,
            "boundaries must be strictly increasing",
        ),
    ],
    ids=["total_steps", "transition_begin", "decay_every", "boundary_sign", "boundary_order"],
)
def test_a_shape_derived_count_is_checked_when_the_schedule_runs(build_schedule, rows, message):
    """The build-time check cannot fire on a count with no value yet, so it travels with the graph. The
    jax backend drops assertions, which is why this is the best that can be offered rather than a
    guarantee."""
    schedule = (
        build_schedule()
    )  # Built outside the block: a build-time raise is not what is claimed.

    with pytest.raises(ValueError, match=message):
        evaluate_shape_derived_schedule(schedule, [3], rows=rows)


@pytest.mark.parametrize(
    "total_steps",
    [matrix("X").shape, vector("v", dtype="int64"), np.array([2, 3])],
    ids=["shape-tuple", "vector", "numpy-array"],
)
def test_a_horizon_that_is_not_a_single_number_is_refused(total_steps):
    """``X.shape`` for ``X.shape[0]`` is the easy slip, and left to the graph it surfaces as a bare
    ``AssertionError`` naming neither the argument nor the mistake."""
    with pytest.raises(ValueError, match="total_steps must be a single number"):
        linear_schedule(0.5, total_steps)


def test_a_literal_horizon_leaves_no_runtime_check_in_the_graph():
    """Folding the comparison at build time is what keeps a literal schedule as cheap to compile as it
    was; a check attached unconditionally would burden every graph that never needed one."""

    def carries_a_check(schedule):
        compiled = compile_shape_derived_schedule(schedule)
        return any(isinstance(node.op, CheckAndRaise) for node in compiled.maker.fgraph.apply_nodes)

    assert not carries_a_check(linear_schedule(0.5, HORIZON))
    assert carries_a_check(linear_schedule(0.5, SHAPE_SOURCE.shape[0]))
