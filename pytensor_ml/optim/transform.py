from collections.abc import Callable, Sequence

from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import (
    Gradients,
    LossGradientsOrUpdates,
    Parameter,
    Rate,
    Schedule,
    Transform,
    Updates,
    counter,
    reuses_state,
    state_for,
    steps_of,
    to_updates,
)


def _reject_gradients(updates: Updates, what: str) -> None:
    """Raise unless ``updates`` carries steps, for a transform that is a silent no-op on gradients."""
    if isinstance(updates, Gradients):
        raise ValueError(
            f"{what} has no effect on gradients ahead of a scale-invariant rule such as adam, which "
            "normalizes whatever magnitude it is given, so the chain would train as though it were absent. "
            "Move it after the rule to scale the step, or pass the rate to the rule as its "
            "`learning_rate`."
        )


def trace(decay: float = 0.9, nesterov: bool = False, *, namespace: str = "trace") -> Transform:
    r"""
    Accumulate into a velocity buffer (classical or Nesterov momentum).

    Reading :math:`s = \text{updates}[p] - p`, the velocity is :math:`v \leftarrow \rho v + s`, and the
    new value is :math:`v` (classical) or :math:`s + \rho v` (Nesterov lookahead).

    Position in a :func:`~pytensor_ml.optim.base.chain` decides what is accumulated. Ahead of a rule this
    is momentum on the gradients, which is what heavy-ball descent means; behind one it smooths the step
    a rule already decided on, on top of whatever momentum that rule keeps of its own.

    Parameters
    ----------
    decay : float
        Momentum coefficient :math:`\rho`. Default 0.9.
    nesterov : bool
        Apply the Nesterov lookahead correction. Default False.
    namespace : str
        Prefix for the velocity this transform allocates, as ``"{parameter}/{namespace}/velocity"``. Give
        two traces in one chain different namespaces so their velocities stay distinct at the
        serialization boundary. Default ``"trace"``.

    Returns
    -------
    transform : Transform
        A transform that folds momentum into the updates dict.

    Examples
    --------
    Fold momentum into whatever produced the steps, giving a rule that has none of its own:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import chain, compile_train, sgd, trace

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, chain(sgd(0.1), trace(decay=0.9, nesterov=True)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    @reuses_state
    def transform(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        updates = to_updates(loss_gradients_or_updates, parameters)
        next_updates = updates.copy()
        for parameter in parameters:
            step = updates[parameter] - parameter
            velocity = state_for(parameter, f"{namespace}/velocity")
            new_velocity = decay * velocity + step
            next_updates[velocity] = new_velocity
            next_updates[parameter] = parameter + (
                step + decay * new_velocity if nesterov else new_velocity
            )
        return next_updates

    return transform


def scale(factor: Rate) -> Transform:
    """
    Scale each step by a constant factor.

    Typically the terminal transform in a chain, used to apply the learning rate after a unit-rate base rule.

    Belongs behind a rule, and raises ahead of one: an adaptive rule normalizes whatever magnitude it is
    handed, so scaling its input changes nothing about what it does.

    Parameters
    ----------
    factor : float or shared tensor variable
        Multiplier applied to every step.

    Returns
    -------
    transform : Transform
        A transform that rescales the updates dict.

    Examples
    --------
    Multiply every step by a constant, most often to shrink a rule's steps without rebuilding it at a
    different rate:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, chain, compile_train, scale

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, chain(adam(1e-3), scale(0.5)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    def transform(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        updates = to_updates(loss_gradients_or_updates, parameters)
        _reject_gradients(updates, f"scale({factor})")
        return updates.replacing(
            {
                parameter: parameter + factor * step
                for parameter, step in zip(parameters, steps_of(updates, parameters))
            }
        )

    return transform


def scale_by_schedule(schedule: Schedule, *, namespace: str = "scale_by_schedule") -> Transform:
    """
    Scale each step by a schedule read off a training clock of its own.

    The terminal transform for scheduling a rate *after* a rule rather than inside it. The two are different
    graphs whenever anything sits between them: a clip placed before this one bounds a step at unit rate, so
    its bound stays in gradient units instead of moving with the rate, while a rule given the schedule as its
    ``learning_rate`` has already applied the rate by the time the clip sees the step.

    .. code-block:: python

        schedule = cosine_schedule(3e-4, 10_000)
        rule = chain(adam(1.0), clip_by_global_norm(1.0), scale_by_schedule(schedule))

    The clock advances on its own. It is a :class:`~pytensor_ml.params.StepCounter`, so
    :func:`~pytensor_ml.pytensorf.collect_clock_updates` finds it in the graph and writes its advance into
    the compiled step, and every other clock the step reads counts the same steps as this one.

    Parameters
    ----------
    schedule : Schedule
        A ``(step_count) -> rate`` callable such as :func:`~pytensor_ml.optim.schedules.cosine_schedule`,
        applied to the clock and multiplied into every step.
    namespace : str
        Prefix for the clock this transform allocates, as ``"{namespace}/step_count"``. Give two scheduled
        scales in one graph different namespaces so their clocks stay distinct at the serialization
        boundary. Default ``"scale_by_schedule"``.

    Returns
    -------
    transform : Transform
        A transform that rescales the updates dict by the rate its clock currently reads.

    Examples
    --------
    Apply the rate at the end of a chain rather than inside the rule, so a clip placed before it bounds
    the step in gradient units instead of units that move with the rate:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, chain, clip_by_global_norm, compile_train, cosine_schedule, scale_by_schedule

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, chain(adam(1.0), clip_by_global_norm(1.0), scale_by_schedule(cosine_schedule(3e-4, 10_000))))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    @reuses_state
    def transform(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        updates = to_updates(loss_gradients_or_updates, parameters)
        _reject_gradients(updates, "scale_by_schedule")
        return scale(schedule(counter(f"{namespace}/step_count")))(updates, parameters)

    return transform


def add_weight_decay(
    weight_decay: float = 0.01,
    mask: Callable[[Parameter], bool] | None = None,
) -> Transform:
    r"""
    Subtract a weight-decay term :math:`\lambda p`.

    Place this behind a rule and before a terminal :func:`scale` so the final update is
    :math:`p \leftarrow p + \eta (s - \lambda p)`, giving decay that scales with the learning rate but is
    decoupled from any adaptive rescaling earlier in the chain -- the AdamW form.

    Ahead of a rule it instead adds the term to the gradient, which is the coupled L2 penalty the decay
    was named after. That form is a different optimizer, not a placement detail: an adaptive rule divides
    it through by the second moment along with everything else.

    Parameters
    ----------
    weight_decay : float
        Decay coefficient :math:`\lambda`. Default 0.01.
    mask : callable, optional
        Predicate ``(parameter) -> bool`` selecting which parameters receive decay. Decay is applied to every
        parameter when omitted.

    Returns
    -------
    transform : Transform
        A transform that folds weight decay into the updates dict.

    Examples
    --------
    Pull weights towards zero by a fixed fraction of themselves each step, independently of the
    gradient. The mask keeps biases and norm scales out of it:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, add_weight_decay, chain, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, chain(adam(1e-3), add_weight_decay(0.01, mask=lambda parameter: parameter.ndim > 1)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    def transform(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        updates = to_updates(loss_gradients_or_updates, parameters)
        in_gradient_space = isinstance(updates, Gradients)

        def decayed(parameter: Parameter, step: TensorVariable) -> TensorVariable:
            if mask is not None and not mask(parameter):
                return step
            penalty = weight_decay * parameter
            # A rule negates what it reads, so the term that pulls a weight towards zero enters a
            # gradient with the opposite sign to the one it enters a step with.
            return (step + penalty) if in_gradient_space else (step - penalty)

        return updates.replacing(
            {
                parameter: parameter + decayed(parameter, step)
                for parameter, step in zip(parameters, steps_of(updates, parameters))
            }
        )

    return transform
