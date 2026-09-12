from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import Schedule
from pytensor_ml.optim.checks import checked_scalar


def _checked_total_steps(total_steps: int | TensorVariable) -> TensorVariable:
    """Return the horizon length, refusing an empty one wherever that can be established."""
    return checked_scalar(
        pt.as_tensor_variable(total_steps),
        name="total_steps",
        complaint="must be at least 1",
        condition_fn=lambda value: value >= 1,
    )


def _checked_transition_begin(transition_begin: int | TensorVariable) -> TensorVariable:
    """Return the step the horizon starts at, refusing a negative one wherever that can be established."""
    return checked_scalar(
        pt.as_tensor_variable(transition_begin),
        name="transition_begin",
        complaint="must not be negative",
        condition_fn=lambda value: value >= 0,
    )


def _clamped_progress(
    step_count: TensorVariable,
    total_steps: TensorVariable,
    transition_begin: TensorVariable | int = 0,
) -> TensorVariable:
    """Return the fraction of the horizon completed at ``step_count``, held at 0 before it starts and 1
    after it ends, so every schedule flattens outside its horizon rather than running past it."""
    floatX = config.floatX
    step_limit = total_steps.astype(floatX)
    begin = pt.as_tensor_variable(transition_begin).astype(floatX)
    return pt.clip(step_count.astype(floatX) - begin, 0.0, step_limit) / step_limit


def cosine_schedule(
    learning_rate: float,
    total_steps: int | TensorVariable,
    final_learning_rate: float = 0.0,
) -> Schedule:
    r"""
    Move the learning rate from ``learning_rate`` to ``final_learning_rate`` along a half cosine.

    At step :math:`t` of :math:`T` total steps,

    .. math::

        \eta_t = \eta_f + \frac{1}{2} (\eta_0 - \eta_f)
                 \left(1 + \cos\left(\pi \frac{\min(t, T)}{T}\right)\right)

    so the rate leaves :math:`\eta_0` slowly, moves fastest at the midpoint, and flattens into
    :math:`\eta_f`, where it stays for any step past :math:`T`. A ``final_learning_rate`` above
    ``learning_rate`` ramps up instead of decaying.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at step zero.
    total_steps : int or TensorVariable
        Number of steps :math:`T` over which the rate reaches its endpoint. Must be at least one.
    final_learning_rate : float, optional
        Endpoint :math:`\eta_f` reached at step ``total_steps``, above or below ``learning_rate``.
        Default 0.0.

    Returns
    -------
    schedule : Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.

    Examples
    --------
    Hand the schedule to an optimizer in place of a rate:

    .. code-block:: python

        from pytensor_ml.optim import adam, cosine_schedule

        rule = adam(learning_rate=cosine_schedule(3e-4, 10_000))
    """
    total_steps = _checked_total_steps(total_steps)

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(final_learning_rate, dtype=floatX)

        progress = _clamped_progress(step_count, total_steps)
        # Weighting both endpoints keeps each one exact: at weight 0 or 1 the other term vanishes, where
        # `final + (initial - final) * factor` recovers `initial` by cancellation instead.
        weight = 0.5 * (1.0 - pt.cos(np.pi * progress))
        return initial_rate * (1.0 - weight) + final_rate * weight

    return schedule


def linear_schedule(
    learning_rate: float,
    total_steps: int | TensorVariable,
    final_learning_rate: float = 0.0,
    transition_begin: int | TensorVariable = 0,
) -> Schedule:
    r"""
    Move the learning rate from ``learning_rate`` to ``final_learning_rate`` at a constant rate.

    At step :math:`t`, decaying over :math:`T` steps starting from step :math:`B`,

    .. math::

        \eta_t = \eta_0 + (\eta_f - \eta_0)
                 \frac{\min(\max(t - B, 0),\; T)}{T}

    so the rate holds at :math:`\eta_0` through step :math:`B`, falls by the same amount every step for
    :math:`T` steps, then holds at :math:`\eta_f` from step :math:`B + T` on.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at every step up to ``transition_begin``.
    total_steps : int or TensorVariable
        Number of steps :math:`T` the decay itself spans. Must be at least one.
    final_learning_rate : float, optional
        Endpoint :math:`\eta_f` reached at step ``transition_begin + total_steps``, above or below
        ``learning_rate``. Default 0.0.
    transition_begin : int or TensorVariable, optional
        Number of steps :math:`B` to hold the initial rate before decaying. Must not be negative.
        Default 0.

    Returns
    -------
    schedule : Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.

    Examples
    --------
    Move the rate along a straight line, the usual choice for a warmup or a linear decay to zero:

    .. code-block:: python

        from pytensor_ml.optim import adam, linear_schedule

        rule = adam(learning_rate=linear_schedule(3e-4, total_steps=10_000, final_learning_rate=1e-5))
    """
    total_steps = _checked_total_steps(total_steps)
    transition_begin = _checked_transition_begin(transition_begin)

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(final_learning_rate, dtype=floatX)

        progress = _clamped_progress(step_count, total_steps, transition_begin)
        return initial_rate * (1.0 - progress) + final_rate * progress

    return schedule


def linear_onecycle_schedule(
    peak_value: float,
    total_steps: int,
    pct_start: float = 0.3,
    pct_final: float = 0.85,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> Schedule:
    r"""Move the learning rate through three linear phases over one cycle.

    The rate rises from its initial value to ``peak_value`` during the first phase, returns to
    the initial value during the second, and falls to its final value during the third.

    For :math:`T =` ``total_steps``, phase boundaries :math:`s_1` and :math:`s_2`, initial rate
    :math:`\eta_0`, peak rate :math:`\eta_p`, and final rate :math:`\eta_f`, the schedule is

    .. math::

        \eta_t = \begin{cases}
            \eta_0 + (\eta_p - \eta_0)t / s_1 & t \leq s_1 \\
            \eta_p + (\eta_0 - \eta_p)(t - s_1) / (s_2 - s_1) & s_1 < t \leq s_2 \\
            \eta_0 + (\eta_f - \eta_0)(t - s_2) / (T - s_2) & s_2 < t \leq T
        \end{cases}

    Parameters
    ----------
    peak_value : float
        Maximum learning rate, reached at the end of the first phase.
    total_steps : int
        Number of steps in the complete cycle. Must be positive and give every rounded phase at least
        one step.
    pct_start : float, optional
        Fraction of the cycle spent increasing to ``peak_value``. Must be between zero and
        ``pct_final``. Default 0.3.
    pct_final : float, optional
        Fraction of the cycle spent increasing and returning to the initial value. Must be between
        ``pct_start`` and one. Default 0.85.
    div_factor : float, optional
        Positive factor by which ``peak_value`` is divided to obtain the initial value. Default 25.0.
    final_div_factor : float, optional
        Positive factor by which the initial value is divided to obtain the final value. Default
        10000.0.

    Returns
    -------
    schedule : Schedule
        A callable mapping a symbolic step count to a scalar learning rate.

    Examples
    --------
    Hand a one-cycle schedule directly to an optimizer:

    .. code-block:: python

        from pytensor_ml.optim import adam, linear_onecycle_schedule

        rule = adam(learning_rate=linear_onecycle_schedule(8e-3, total_steps=10_000))
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}.")
    if not 0 < pct_start < pct_final < 1:
        raise ValueError(
            "pct_start and pct_final must satisfy 0 < pct_start < pct_final < 1, "
            f"got pct_start={pct_start} and pct_final={pct_final}."
        )
    if not div_factor > 0:
        raise ValueError(f"div_factor must be positive, got {div_factor}.")
    if not final_div_factor > 0:
        raise ValueError(f"final_div_factor must be positive, got {final_div_factor}.")
    peak_step = int(pct_start * total_steps)
    final_phase_step = int(pct_final * total_steps)
    phase_steps = (peak_step, final_phase_step - peak_step, total_steps - final_phase_step)
    if min(phase_steps) < 1:
        raise ValueError(
            "pct_start and pct_final must produce three phases of at least one step after rounding; "
            f"got phase lengths {phase_steps} for total_steps={total_steps}."
        )
    initial_value = peak_value / div_factor

    return join_schedules(
        [
            linear_schedule(initial_value, peak_step, peak_value),
            linear_schedule(peak_value, final_phase_step - peak_step, initial_value),
            linear_schedule(
                initial_value,
                total_steps - final_phase_step,
                initial_value / final_div_factor,
            ),
        ],
        [peak_step, final_phase_step],
    )


def exponential_schedule(
    learning_rate: float,
    total_steps: int | TensorVariable,
    final_learning_rate: float,
    transition_begin: int | TensorVariable = 0,
) -> Schedule:
    r"""
    Move the learning rate from ``learning_rate`` to ``final_learning_rate`` by a constant factor per step.

    At step :math:`t`, decaying over :math:`T` steps starting from step :math:`B`, and writing
    :math:`p = \min(\max(t - B, 0),\; T) / T`,

    .. math::

        \eta_t = \eta_0 \left(\frac{\eta_f}{\eta_0}\right)^{p}

    so the rate is multiplied by the same factor every step — falling fastest in absolute terms early on —
    and holds at :math:`\eta_f` from step :math:`B + T` on.

    ``final_learning_rate`` is required and must be positive: a geometric path approaches zero without
    ever reaching it, so there is no endpoint of zero to move to.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at every step up to ``transition_begin``. Must be positive.
    total_steps : int or TensorVariable
        Number of steps :math:`T` the decay itself spans. Must be at least one.
    final_learning_rate : float
        Endpoint :math:`\eta_f` reached at step ``transition_begin + total_steps``. Must be positive,
        and may be above ``learning_rate`` to ramp up geometrically.
    transition_begin : int or TensorVariable, optional
        Number of steps :math:`B` to hold the initial rate before decaying. Must not be negative.
        Default 0.

    Returns
    -------
    schedule : Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.

    Examples
    --------
    Decay by a constant factor per step, so the rate falls fast early and flattens out. Both endpoints
    must be positive, since no finite number of multiplications reaches zero:

    .. code-block:: python

        from pytensor_ml.optim import adam, exponential_schedule

        rule = adam(learning_rate=exponential_schedule(3e-4, total_steps=10_000, final_learning_rate=1e-6))
    """
    total_steps = _checked_total_steps(total_steps)
    transition_begin = _checked_transition_begin(transition_begin)
    if final_learning_rate <= 0.0 or learning_rate <= 0.0:
        raise ValueError(
            f"exponential_schedule needs positive rates, got learning_rate={learning_rate} and "
            f"final_learning_rate={final_learning_rate}. A geometric path never reaches zero, so pick an "
            f"endpoint you are willing to train at."
        )

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        # Taken in float64 on the host, so a small ratio keeps its precision under floatX=float32.
        log_decay = np.asarray(np.log(final_learning_rate / learning_rate), dtype=floatX)

        progress = _clamped_progress(step_count, total_steps, transition_begin)
        return initial_rate * pt.exp(log_decay * progress)

    return schedule


def polynomial_schedule(
    learning_rate: float,
    total_steps: int | TensorVariable,
    final_learning_rate: float = 0.0,
    transition_begin: int | TensorVariable = 0,
    power: float = 1.0,
) -> Schedule:
    r"""
    Move the learning rate from ``learning_rate`` to ``final_learning_rate`` along a power curve.

    At step :math:`t`, decaying over :math:`T` steps starting from step :math:`B`, and writing
    :math:`p = \min(\max(t - B, 0),\; T) / T`,

    .. math::

        \eta_t = \eta_f + (\eta_0 - \eta_f) (1 - p)^{\gamma}

    so :math:`\gamma > 1` moves the rate quickly and then flattens, :math:`\gamma < 1` holds it near
    :math:`\eta_0` and then moves, and :math:`\gamma = 1` is a straight line. The rate holds at
    :math:`\eta_f` from step :math:`B + T` on.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned at every step up to ``transition_begin``.
    total_steps : int or TensorVariable
        Number of steps :math:`T` the decay itself spans. Must be at least one.
    final_learning_rate : float, optional
        Endpoint :math:`\eta_f` reached at step ``transition_begin + total_steps``, above or below
        ``learning_rate``. Default 0.0.
    transition_begin : int or TensorVariable, optional
        Number of steps :math:`B` to hold the initial rate before decaying. Must not be negative.
        Default 0.
    power : float, optional
        Exponent :math:`\gamma` applied to the remaining fraction of the horizon. Must be positive.
        Default 1.0, which decays linearly.

    Returns
    -------
    schedule : Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.

    Examples
    --------
    Bend the path between the endpoints: ``power=1.0`` is linear, higher powers hold the initial rate
    longer before dropping away:

    .. code-block:: python

        from pytensor_ml.optim import adam, polynomial_schedule

        rule = adam(learning_rate=polynomial_schedule(3e-4, total_steps=10_000, power=2.0))
    """
    total_steps = _checked_total_steps(total_steps)
    transition_begin = _checked_transition_begin(transition_begin)
    if power <= 0.0:
        raise ValueError(f"power must be positive, got {power}.")

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        final_rate = np.asarray(final_learning_rate, dtype=floatX)
        exponent = np.asarray(power, dtype=floatX)

        remaining = 1.0 - _clamped_progress(step_count, total_steps, transition_begin)
        weight = 1.0 - remaining**exponent
        return initial_rate * (1.0 - weight) + final_rate * weight

    return schedule


def step_decay(
    learning_rate: float,
    *,
    decay_every: int | TensorVariable,
    decay_factor: float = 0.1,
    min_learning_rate: float = 0.0,
    transition_begin: int | TensorVariable = 0,
) -> Schedule:
    r"""
    Multiply the learning rate by ``decay_factor`` every ``decay_every`` steps.

    At step :math:`t`, dropping by :math:`\gamma` every :math:`E` steps and starting from step :math:`B`,

    .. math::

        \eta_t = \max\left(\eta_0\, \gamma^{\lfloor \max(t - B,\, 0) / E \rfloor},\; \eta_{\min}\right)

    so the rate is a staircase rather than a curve, and it decays indefinitely rather than over a fixed
    horizon. Every argument after the rate is keyword-only, because with no horizon its positional slots
    would not line up with the other schedules'.

    Parameters
    ----------
    learning_rate : float
        Initial rate :math:`\eta_0`, returned until the first drop.
    decay_every : int or TensorVariable
        Number of steps :math:`E` between drops. Must be at least one.
    decay_factor : float, optional
        Multiplier :math:`\gamma` applied at each drop. Must be in ``(0, 1]``. Default 0.1.
    min_learning_rate : float, optional
        Floor :math:`\eta_{\min}` the staircase never goes below. Default 0.0.
    transition_begin : int or TensorVariable, optional
        Number of steps :math:`B` to hold the initial rate before the first drop can occur. Must not be
        negative. Default 0.

    Returns
    -------
    schedule : Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.

    Examples
    --------
    Cut the rate by a factor on a fixed cadence, holding it flat in between -- the staircase familiar
    from torch's ``StepLR``:

    .. code-block:: python

        from pytensor_ml.optim import adam, step_decay

        rule = adam(learning_rate=step_decay(3e-4, decay_every=2_000, decay_factor=0.5))
    """
    if not 0.0 < decay_factor <= 1.0:
        raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}.")
    decay_every = checked_scalar(
        pt.as_tensor_variable(decay_every),
        name="decay_every",
        complaint="must be at least 1",
        condition_fn=lambda value: value >= 1,
    )
    transition_begin = _checked_transition_begin(transition_begin)

    def schedule(step_count: TensorVariable) -> TensorVariable:
        floatX = config.floatX
        initial_rate = np.asarray(learning_rate, dtype=floatX)
        floor_rate = np.asarray(min_learning_rate, dtype=floatX)
        factor = np.asarray(decay_factor, dtype=floatX)

        drops = pt.maximum(step_count - transition_begin, 0) // decay_every
        return pt.maximum(initial_rate * factor ** drops.astype(floatX), floor_rate)

    return schedule


def constant_schedule(learning_rate: float) -> Schedule:
    r"""
    Hold the learning rate at ``learning_rate`` forever.

    Useful as a segment of :func:`join_schedules`, where a constant stretch before or between decays is
    otherwise awkward to express.

    Parameters
    ----------
    learning_rate : float
        The rate :math:`\eta_0` returned at every step.

    Returns
    -------
    schedule : Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.

    Examples
    --------
    Hold one rate for the whole run. Useful as a segment of :func:`join_schedules`, where every other
    segment is a schedule too:

    .. code-block:: python

        from pytensor_ml.optim import adam, constant_schedule

        rule = adam(learning_rate=constant_schedule(3e-4))
    """

    def schedule(step_count: TensorVariable) -> TensorVariable:
        # The rate has to stay a function of `step_count`: callers compile the schedule against the
        # counter, and a graph that ignores its input raises UnusedInputError.
        return pt.zeros_like(step_count, dtype=config.floatX) + np.asarray(
            learning_rate, dtype=config.floatX
        )

    return schedule


def join_schedules(
    schedules: Sequence[Schedule], boundaries: Sequence[int | TensorVariable]
) -> Schedule:
    r"""
    Run ``schedules`` one after another, switching at ``boundaries``.

    Each schedule after the first receives the step count *since its boundary*, so a decay placed second
    starts from its own step zero rather than partway down its curve. That is what makes warmup followed by
    decay a composition rather than a special case.

    Every segment is evaluated at every step and the result selected, so a later schedule is called with a
    negative count before its boundary. The schedules in this module clamp that to zero; a custom schedule
    must tolerate it.

    Parameters
    ----------
    schedules : sequence of Schedule
        The schedules to run in order. Must not be empty.
    boundaries : sequence of int or TensorVariable
        Steps at which to hand over to the next schedule, one fewer than ``schedules``, strictly
        increasing and positive.

    Returns
    -------
    schedule : Schedule
        A callable mapping a symbolic step count to a scalar learning rate, ready to hand to a rule as
        its ``learning_rate``.

    Examples
    --------
    Run schedules back to back, switching at each boundary step. A linear warmup into a cosine decay is
    the standard recipe for a transformer:

    .. code-block:: python

        from pytensor_ml.optim import adam, cosine_schedule, join_schedules, linear_schedule

        warmup = linear_schedule(0.0, total_steps=1_000, final_learning_rate=3e-4)
        decay = cosine_schedule(3e-4, total_steps=9_000)

        rule = adam(learning_rate=join_schedules([warmup, decay], boundaries=[1_000]))
    """
    if not schedules:
        raise ValueError("join_schedules needs at least one schedule.")
    if len(boundaries) != len(schedules) - 1:
        raise ValueError(
            f"join_schedules needs one fewer boundary than schedules, got {len(boundaries)} "
            f"boundaries for {len(schedules)} schedules."
        )
    positive_boundaries = [
        checked_scalar(
            pt.as_tensor_variable(boundary),
            name=f"boundaries[{index}]",
            complaint="must be positive",
            condition_fn=lambda value: value >= 1,
        )
        for index, boundary in enumerate(boundaries)
    ]
    checked_boundaries = positive_boundaries[:1] + [
        checked_scalar(
            later,
            name="boundaries",
            complaint="must be strictly increasing",
            condition_fn=lambda value: value > earlier,
        )
        for earlier, later in pairwise(positive_boundaries)
    ]

    def schedule(step_count: TensorVariable) -> TensorVariable:
        rate = schedules[0](step_count)
        for boundary, next_schedule in zip(checked_boundaries, schedules[1:]):
            rate = pt.where(step_count < boundary, rate, next_schedule(step_count - boundary))
        return rate

    return schedule
