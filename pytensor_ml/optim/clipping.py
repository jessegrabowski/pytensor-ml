from collections.abc import Sequence

import pytensor.tensor as pt

from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import (
    LossGradientsOrUpdates,
    Parameter,
    Transform,
    Updates,
    steps_of,
    to_updates,
)
from pytensor_ml.optim.checks import checked_scalar


def clip_by_global_norm(max_norm: float | TensorVariable = 1.0) -> Transform:
    r"""
    Rescale everything by a single factor so its global L2 norm does not exceed ``max_norm``.

    With :math:`\|s\|` the norm of the concatenated values, every one is multiplied by
    :math:`\min(1, \text{max\_norm} / (\|s\| + \epsilon))`, preserving the direction while bounding the
    magnitude.

    Position in a :func:`~pytensor_ml.optim.base.chain` decides what is bounded. Ahead of a rule this
    clips the gradients, so a spike never reaches the moment estimates; behind one it clips the step the
    rule produced, which an adaptive rule has already normalized to roughly its learning rate however
    large the gradient was. The first is what stops an exploding gradient.

    Parameters
    ----------
    max_norm : float or TensorVariable
        Maximum allowed global norm. Default 1.0.

    Returns
    -------
    transform : Transform
        A transform that clips by global norm, in whichever space it is placed.

    Examples
    --------
    Bound the gradient before the rule sees it, so one exploding batch cannot poison the moment estimates
    for every step after it:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, chain, clip_by_global_norm, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, chain(clip_by_global_norm(1.0), adam(1e-3)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))

    Place it after the rule instead to cap how far a single step may move the parameters, whatever the
    rule asked for:

    .. code-block:: python

        from pytensor_ml.optim import adam, chain, clip_by_global_norm

        rule = chain(adam(1e-3), clip_by_global_norm(0.01))
    """

    bound = checked_scalar(
        pt.as_tensor_variable(max_norm),
        name="max_norm",
        complaint="is a norm to clip to and must be positive",
        condition_fn=lambda value: value > 0.0,
    )

    def transform(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        updates = to_updates(loss_gradients_or_updates, parameters)
        steps = steps_of(updates, parameters)
        global_norm = pt.sqrt(sum(pt.sum(step**2) for step in steps))
        clip_scale = pt.minimum(1.0, bound / (global_norm + 1e-8))
        return updates.replacing(
            {parameter: parameter + clip_scale * step for parameter, step in zip(parameters, steps)}
        )

    return transform


def clip_by_value(
    min_value: float | TensorVariable = -1.0, max_value: float | TensorVariable = 1.0
) -> Transform:
    """
    Clamp every value element-wise into ``[min_value, max_value]``.

    Position in a :func:`~pytensor_ml.optim.base.chain` decides what is clamped: gradients ahead of a
    rule, the rule's step behind it. See :func:`clip_by_global_norm` for what that choice costs.

    Parameters
    ----------
    min_value : float or TensorVariable
        Lower bound. Default -1.0.
    max_value : float or TensorVariable
        Upper bound. Default 1.0.

    Returns
    -------
    transform : Transform
        A transform that clips element-wise, in whichever space it is placed.

    Examples
    --------
    Clip each coordinate on its own, which bounds the magnitude but tilts the direction whenever only
    some coordinates are clipped:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, chain, clip_by_value, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, chain(clip_by_value(-1.0, 1.0), adam(1e-3)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    # The lower bound is established as a scalar first, because comparing the two while either could
    # still be an array gives the ordering check a non-scalar condition to carry.
    lower = checked_scalar(pt.as_tensor_variable(min_value), name="min_value")
    upper = checked_scalar(
        pt.as_tensor_variable(max_value),
        name="max_value",
        complaint="is the upper bound and must not be below min_value",
        condition_fn=lambda value: value >= lower,
    )

    def transform(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        updates = to_updates(loss_gradients_or_updates, parameters)
        return updates.replacing(
            {
                parameter: parameter + pt.clip(step, lower, upper)
                for parameter, step in zip(parameters, steps_of(updates, parameters))
            }
        )

    return transform
