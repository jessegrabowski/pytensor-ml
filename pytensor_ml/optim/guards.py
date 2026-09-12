from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytensor.tensor as pt

from pytensor.raise_op import CheckAndRaise
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import (
    LossGradientsOrUpdates,
    Parameter,
    Transform,
    Updates,
    reuses_state,
    scalar_state,
)
from pytensor_ml.optim.checks import checked_scalar
from pytensor_ml.params import StepCounter

type Decision = Callable[[Updates, Sequence[Parameter]], TensorVariable]
"""
Reads the step a rule has proposed and returns a scalar boolean graph: True to throw the step away.

Examples
--------
Pass one straight to :func:`skip_if` when the reason a raised error names does not matter, or wrap it in
a :class:`SkipCondition` when it does:

.. code-block:: python

    import numpy as np
    import pytensor.tensor as pt

    from pytensor_ml.layers import Input, Linear
    from pytensor_ml.loss import SquaredError, supervised_loss
    from pytensor_ml.optim import adam, compile_train, skip_if


    def loss_is_huge(updates, parameters):
        return pt.max([pt.abs(updates[parameter]).max() for parameter in parameters]) > 1e6


    X = Input("X", shape=(None, 4))
    loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

    step = compile_train(loss, skip_if(adam(1e-3), loss_is_huge))
    loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
"""


@dataclass(frozen=True)
class SkipCondition:
    """
    A rule for throwing a training step away, paired with the phrase naming why.

    Attributes
    ----------
    decide : callable
        ``(updates, parameters) -> scalar bool graph``, True for a step to throw away.
    reason : str
        Phrase naming why this condition skips, completing the sentence "the optimizer ..." in the error a
        run of consecutive skips raises. Default names no particular cause.

    Examples
    --------
    Pair a decision graph with the reason a raised error should name, for a condition of your own:

    .. code-block:: python

        import numpy as np
        import pytensor.tensor as pt

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import SkipCondition, adam, compile_train, skip_if

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        any_step_huge = SkipCondition(
            decide=lambda updates, parameters: pt.max([pt.abs(step).max() for step in updates.values()]) > 1e3,
            reason="took a step with an implausibly large coordinate",
        )

        step = compile_train(loss, skip_if(adam(1e-3), any_step_huge))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    decide: Decision
    reason: str = "met the skip condition"

    def __call__(self, updates: Updates, parameters: Sequence[Parameter]) -> TensorVariable:
        return self.decide(updates, parameters)


def nonfinite() -> SkipCondition:
    """
    Throw the step away when any parameter the rule would write is inf or NaN.

    The condition behind :func:`apply_if_finite`, and the one to reach for when there is no scale to
    threshold on. It fires only once a value has already gone non-finite, which on a diverging run is a
    lagging alarm: the step before is typically finite and enormous. :func:`large_step` catches that one.

    Only the parameters are checked. A policy is free to keep a sentinel among its own state --
    :func:`~pytensor_ml.optim.policy.reduce_on_plateau` holds an infinite best-loss until it has seen a full
    window -- and that is not a step to throw away. Optimizer state cannot hide a NaN for long in any case,
    since a poisoned moment reaches its parameter on the very next step.

    Examples
    --------
    The condition behind :func:`apply_if_finite`, useful when you want it alongside a different skip budget
    than that helper's default:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, compile_train, nonfinite, skip_if

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, skip_if(adam(1e-3), nonfinite(), max_consecutive_skips=None))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    def decide(updates: Updates, parameters: Sequence[Parameter]) -> TensorVariable:
        checked_values = [
            new_value
            for new_value in (updates[parameter] for parameter in parameters)
            if new_value.dtype.startswith("float")
        ]
        if not checked_values:
            raise ValueError(
                "There are no floating-point parameters to check for finiteness. A guarded rule updates "
                "parameters; check that `parameters` is not empty."
            )
        return ~pt.all(pt.stack([pt.all(pt.isfinite(value)) for value in checked_values]))

    return SkipCondition(decide, "produced non-finite updates")


def large_step(max_norm: float | TensorVariable) -> SkipCondition:
    r"""
    Throw the step away when the global L2 norm of the steps reaches ``max_norm``.

    A strict superset of :func:`nonfinite`: an inf or NaN anywhere makes the norm inf or NaN, and the
    comparison is written so that either is thrown away. Thresholding on scale catches a divergence while
    its numbers are still finite, several steps before anything turns into a NaN, which is what makes this
    the condition to choose when a plausible bound on a healthy step is known.

    Distinct from :func:`~pytensor_ml.optim.clipping.clip_by_global_norm`, which rescales an outsized step
    down and applies it. This throws it away, and an inf rescaled by :math:`\text{max\_norm} / \infty` is a
    NaN rather than a bounded step, so clipping alone does not survive one.

    Parameters
    ----------
    max_norm : float or TensorVariable
        Norm at which a step is thrown away rather than applied.

    Examples
    --------
    Skip a step whose global norm exceeds a bound. Unlike :func:`clip_by_global_norm`, which rescales an
    outsized step and applies it, this one discards it entirely:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, compile_train, large_step, skip_if

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, skip_if(adam(1e-3), large_step(10.0)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    bound = checked_scalar(
        pt.as_tensor_variable(max_norm),
        name="max_norm",
        complaint="is a norm to compare against and must be positive",
        condition_fn=lambda value: value > 0.0,
    )

    def decide(updates: Updates, parameters: Sequence[Parameter]) -> TensorVariable:
        steps = [updates[parameter] - parameter for parameter in parameters]
        global_norm = pt.sqrt(sum(pt.sum(step**2) for step in steps))
        # Negated `<` rather than `>=`: every comparison against a NaN is False, so this throws away a NaN
        # norm where `>=` would apply it.
        return ~(global_norm < bound)

    described = max_norm if isinstance(max_norm, int | float) else "its max_norm bound"
    return SkipCondition(decide, f"produced a step whose global norm reached {described}")


def _counter_or_new(given: Parameter | None, name: str) -> Parameter:
    """Return the caller's own counter, or a fresh scalar state slot under ``name``."""
    return scalar_state(name) if given is None else given


def skip_if(
    rule: Transform,
    condition: SkipCondition | Decision | None = None,
    *,
    max_consecutive_skips: int | None = 5,
    consecutive_skips: Parameter | None = None,
    total_skips: Parameter | None = None,
    namespace: str = "skip_if",
) -> Transform:
    """
    Throw away any step ``condition`` rejects, leaving the parameters and the optimizer state as they were.

    One inf or NaN reaching a parameter poisons it for the rest of the run, and one reaching an Adam moment
    poisons every later step through it, so a single bad batch -- an overflowing exponential, a log of zero,
    a corrupted example -- ends the run. Throwing the step away costs one step instead.

    The guard covers what ``rule`` writes and nothing else: a batch-norm running statistic, written by the
    model and folded in by :func:`~pytensor_ml.optim.train.compile_train` outside the rule, is not held back
    with the step, and the step still returns the loss that produced the skip. Training clocks are exempt
    too -- a skipped step still consumed a step, so the schedules reading them advance as usual.

    .. code-block:: python

        step = compile_train(loss, skip_if(adam(1e-3), large_step(10.0)))

    Parameters
    ----------
    rule : Transform
        The rule to guard, e.g. ``adam(1e-3)`` or a rule already wrapped in a policy.
    condition : SkipCondition or callable, optional
        What to throw a step away for, as a :class:`SkipCondition` such as :func:`nonfinite` or
        :func:`large_step`, or as a bare ``(updates, parameters) -> scalar bool graph`` callable, which the
        raised error names no cause for. Default :func:`nonfinite`.
    max_consecutive_skips : int, optional
        Consecutive skipped steps to tolerate before raising a ``FloatingPointError``. A run that never
        recovers is a divergence rather than a bad batch, and skipping forever looks like training that has
        simply stopped learning. Pass None to skip indefinitely and never raise. Default 5.
    consecutive_skips : shared tensor variable, optional
        Scalar counting consecutive skipped steps, reset to zero by any step that applies. Pass one built by
        :func:`~pytensor_ml.optim.base.scalar_state` to read it from Python between steps; the guard
        allocates its own as ``"{namespace}/consecutive_skips"`` when omitted.
    total_skips : shared tensor variable, optional
        Scalar counting every skipped step, never reset. This is the one to watch to learn how often a run
        is skipping at all, since ``consecutive_skips`` reports only the streak in progress. Allocated as
        ``"{namespace}/total_skips"`` when omitted.
    namespace : str
        Prefix for the state slots this guard allocates, so two guards in one graph keep separate counters
        rather than colliding under one name at the serialization boundary. Default ``"skip_if"``.

    Returns
    -------
    guarded_rule : Transform
        The guarded rule, which also writes both skip counters.

    Examples
    --------
    Wrap a rule to throw away any step meeting the condition, leaving the parameters where they were. It
    raises once too many steps in a row are skipped, so a divergence surfaces instead of looking like
    training that quietly stopped learning:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, compile_train, large_step, skip_if

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        rule = skip_if(adam(1e-3), large_step(10.0), max_consecutive_skips=3)

        step = compile_train(loss, rule)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    if max_consecutive_skips is not None and max_consecutive_skips < 1:
        raise ValueError(
            f"max_consecutive_skips is a number of steps to tolerate and must be at least 1, got "
            f"{max_consecutive_skips}. Pass None to never raise."
        )

    if condition is None:
        condition = nonfinite()
    elif not isinstance(condition, SkipCondition):
        condition = SkipCondition(condition)

    @reuses_state
    def guarded(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        updates = rule(loss_gradients_or_updates, parameters).copy()
        # Snapshotted before the counters are added, since those are the one thing a skipped step still has
        # to write: freeze them along with everything else and the guard can never count its way to the
        # error, which is a silent failure rather than a loud one.
        held_back = {
            variable: new_value
            for variable, new_value in updates.items()
            if not isinstance(variable, StepCounter)
        }

        skipping = condition(updates, parameters)
        consecutive = _counter_or_new(consecutive_skips, f"{namespace}/consecutive_skips")
        total = _counter_or_new(total_skips, f"{namespace}/total_skips")

        next_consecutive = pt.where(skipping, consecutive + 1, 0.0)
        if max_consecutive_skips is not None:
            next_consecutive = CheckAndRaise(
                FloatingPointError,
                f"The optimizer {condition.reason} on {max_consecutive_skips + 1} consecutive steps. "
                "Each was thrown away, so the parameters are the ones from before the first of them, and "
                "training has made no progress since. Lower the learning rate, clip the gradients, or look "
                "for a term in the loss that can overflow; raise `max_consecutive_skips`, or set it to "
                "None, to keep skipping instead.",
            )(next_consecutive, next_consecutive <= max_consecutive_skips)

        updates[consecutive] = next_consecutive.astype(consecutive.dtype)
        updates[total] = pt.where(skipping, total + 1, total).astype(total.dtype)

        for variable, new_value in held_back.items():
            updates[variable] = pt.where(skipping, variable, new_value)

        return updates

    return guarded


def apply_if_finite(
    rule: Transform,
    *,
    max_consecutive_skips: int | None = 5,
    consecutive_skips: Parameter | None = None,
    total_skips: Parameter | None = None,
    namespace: str = "skip_if",
) -> Transform:
    """
    Skip any step that would write a non-finite parameter, leaving the parameters and the optimizer state
    as they were.

    :func:`skip_if` under :func:`nonfinite`, which is the common case and the name optax gives it. Takes the
    same keyword arguments; see :func:`skip_if` for what they mean and what the guard does and does not
    cover.

    Examples
    --------
    The common case of :func:`skip_if`: drop any step carrying a NaN or an infinity, which would otherwise
    poison every parameter it touches and every step after it:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, apply_if_finite, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, apply_if_finite(adam(1e-3)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    return skip_if(
        rule,
        nonfinite(),
        max_consecutive_skips=max_consecutive_skips,
        consecutive_skips=consecutive_skips,
        total_skips=total_skips,
        namespace=namespace,
    )
