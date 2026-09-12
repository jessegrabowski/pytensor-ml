from collections.abc import Sequence

import numpy as np
import pytensor.tensor as pt

from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import (
    Gradients,
    LossGradientsOrUpdates,
    Parameter,
    Steps,
    Transform,
    Updates,
    reuses_state,
    scalar_state,
)
from pytensor_ml.optim.checks import checked_scalar


def reduce_on_plateau(
    rule: Transform,
    scale: Parameter,
    *,
    factor: float = 0.1,
    patience: int | TensorVariable = 10,
    cooldown: int | TensorVariable = 0,
    rtol: float = 1e-4,
    atol: float = 0.0,
    min_scale: float = 0.0,
    accumulation_size: int | TensorVariable = 1,
    namespace: str = "plateau",
) -> Transform:
    r"""
    Cut ``scale`` by ``factor`` once the loss has stopped improving.

    Wraps a rule whose rate is built from ``scale``, so the policy owns a multiplier rather than the rate
    itself and composes with any schedule:

    .. code-block:: python

        scale = scalar_state("plateau/scale", fill_value=1.0)
        rule = reduce_on_plateau(adam(learning_rate=scale * 1e-3), scale, patience=5)

    Unlike torch's, this runs once per training step rather than once per epoch on a validation metric, so
    the loss it sees is whatever expression the rule is given. A per-batch loss is a noisy signal for it;
    ``accumulation_size`` is the answer, deciding on the mean of a window rather than on one batch. Set it
    to the number of steps in an epoch to get torch's cadence, on a better estimate than a single batch.
    That count may be shape-derived, so ``X.shape[0]`` works as readily as a literal.

    It decides from the loss itself rather than from an updates dict, which is the one thing in this module
    a :func:`~pytensor_ml.optim.base.chain` cannot hand along. So it wraps a chain rather than sitting
    inside one -- ``reduce_on_plateau(chain(clip_by_global_norm(1.0), adam(rate)), rate)`` -- and raises
    rather than deciding on something else if placed where the loss no longer reaches it.

    Parameters
    ----------
    rule : Transform
        The rule to wrap. Its rate must be built from ``scale`` for the cuts to reach the step.
    scale : shared tensor variable
        The multiplier this policy owns. Nothing else may write it.
    factor : float
        Multiplier applied on a cut, in the open interval (0, 1). Default 0.1.
    patience : int or TensorVariable
        Steps without improvement before a cut. Default 10.
    cooldown : int or TensorVariable
        Steps to wait after a cut before counting again, during which no step counts as bad. Without one,
        the counter resets on the cut and immediately starts toward the next. Default 0.
    rtol : float
        Relative improvement needed to count, as ``loss < (1 - rtol) * best - atol``. Default 1e-4.
    atol : float
        Absolute improvement needed to count. Default 0.0.
    min_scale : float
        Floor the scale cannot go below. Without one a noisy loss cuts repeatedly and underflows to zero.
        Default 0.0.
    namespace : str
        Prefix for the history this policy allocates, as ``"{namespace}/best_loss"`` and so on. Give two
        policies in one graph different namespaces so their histories stay distinct at the serialization
        boundary. Default ``"plateau"``.
    accumulation_size : int or TensorVariable
        Losses to average before deciding anything. Nothing advances mid-window -- not the count, not the
        cooldown, not the best seen. Default 1, which decides on every step from that step's loss.

    Returns
    -------
    wrapped_rule : Transform
        The wrapped rule, which also writes the scale and the policy's own history.

    Examples
    --------
    Own a scale the rule's rate is built from, and the policy cuts it once the loss stops improving. It
    decides once per step rather than once per epoch, so widen ``accumulation_size`` to judge on a window
    of batches rather than on a single noisy one:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, compile_train, reduce_on_plateau, scalar_state

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        scale = scalar_state("plateau/scale", fill_value=1.0)
        rule = reduce_on_plateau(adam(learning_rate=scale * 1e-3), scale, patience=5, accumulation_size=50)

        step = compile_train(loss, rule)
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    if not 0.0 < factor < 1.0:
        raise ValueError(f"factor must lie in (0, 1), got {factor}.")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError(f"rtol and atol must be non-negative, got rtol={rtol}, atol={atol}.")
    if rtol > 1.0:
        raise ValueError(f"rtol is a relative tolerance and must be at most 1.0, got {rtol}.")
    patience = checked_scalar(
        pt.as_tensor_variable(patience),
        name="patience",
        complaint="is a number of steps and must be at least 1",
        condition_fn=lambda value: value >= 1,
    )
    cooldown = checked_scalar(
        pt.as_tensor_variable(cooldown),
        name="cooldown",
        complaint="is a number of steps and must be non-negative",
        condition_fn=lambda value: value >= 0,
    )
    accumulation_size = checked_scalar(
        pt.as_tensor_variable(accumulation_size),
        name="accumulation_size",
        complaint="is a number of losses to average and must be at least 1",
        condition_fn=lambda value: value >= 1,
    )

    @reuses_state
    def policy(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        if isinstance(loss_gradients_or_updates, list | tuple | dict):
            raise ValueError(
                "reduce_on_plateau decides from the loss, so it needs the loss graph rather than "
                "gradients or an updates dict. It therefore has to come first in a chain, where the loss "
                "still reaches it -- wrap the whole chain in it instead of placing it inside one, or drop "
                "the policy."
            )

        loss = loss_gradients_or_updates
        result = rule(loss, parameters)
        if isinstance(result, Gradients):
            raise ValueError(
                "The wrapped rule returned gradients rather than the steps to take, so every parameter "
                "would move uphill. Put an optimizer such as `adam(rate)` inside the policy."
            )
        updates = Steps(result)

        best_loss = scalar_state(f"{namespace}/best_loss", fill_value=np.inf)
        waited = scalar_state(f"{namespace}/wait")
        cooling = scalar_state(f"{namespace}/cooldown")
        observed = scalar_state(f"{namespace}/observed")
        mean_loss = scalar_state(f"{namespace}/mean_loss")

        # Everything below is gated on `deciding`, so a window that is still filling advances nothing. At the
        # default size of one the window is a single step and the mean is that step's loss.
        seen = observed + 1
        running_mean = (observed * mean_loss + loss) / seen
        deciding = seen >= accumulation_size

        improved = deciding & (running_mean < (1 - rtol) * best_loss - atol)
        counted = pt.where(deciding, pt.where(improved, 0.0, waited + 1), waited)

        # Cooling down zeroes the count rather than pausing it, so the steps immediately after a cut cannot
        # add up to the next one before the network has had a chance to respond to the rate it just got.
        in_cooldown = cooling > 0
        cutting = deciding & ~in_cooldown & (counted >= patience)
        next_cooling = pt.where(in_cooldown, cooling - 1, pt.where(cutting, cooldown, 0.0))

        updates[scale] = pt.where(cutting, pt.maximum(scale * factor, min_scale), scale).astype(
            scale.dtype
        )
        updates[best_loss] = pt.where(improved, running_mean, best_loss).astype(best_loss.dtype)
        updates[waited] = pt.where(deciding & (in_cooldown | cutting), 0.0, counted).astype(
            waited.dtype
        )
        updates[cooling] = pt.where(deciding, next_cooling, cooling).astype(cooling.dtype)
        updates[observed] = pt.where(deciding, 0.0, seen).astype(observed.dtype)
        updates[mean_loss] = pt.where(deciding, 0.0, running_mean).astype(mean_loss.dtype)

        return updates

    return policy
