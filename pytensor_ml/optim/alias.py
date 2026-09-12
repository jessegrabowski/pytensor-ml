from collections.abc import Callable, Sequence

from pytensor_ml.optim.base import (
    LearningRate,
    LossGradientsOrUpdates,
    Parameter,
    Rate,
    Transform,
    Updates,
    counter,
    reuses_state,
)
from pytensor_ml.optim.rules import (
    _require_numeric_learning_rate,
    adadelta_updates,
    adagrad_updates,
    adam_updates,
    adamax_updates,
    adamw_updates,
    nadam_updates,
    rmsprop_updates,
    rprop_updates,
    sgd_updates,
)
from pytensor_ml.optim.transform import scale, trace


def _at_learning_rate(
    learning_rate: LearningRate,
    name: str,
    build_updates: Callable[[Rate], Updates],
) -> Updates:
    """Build updates at ``learning_rate``, reading a schedule off the clock the rule counts its own steps on.
    Both reach that clock through :func:`counter` under ``"{name}/step_count"``, so a rule that keeps a step
    count hands the schedule the same variable instead of a second one measuring the same time."""
    if not callable(learning_rate):
        return build_updates(learning_rate)

    return build_updates(learning_rate(counter(f"{name}/step_count")))


def sgd(
    learning_rate: LearningRate = 0.01,
    momentum: float = 0.0,
    nesterov: bool = False,
    *,
    namespace: str = "sgd",
) -> Transform:
    """
    Stochastic gradient descent, optionally with momentum.

    Parameters
    ----------
    learning_rate : float, shared tensor variable, symbolic scalar, or Schedule
        Step size. A float is baked into the graph; a scalar shared variable can be steered from Python with
        ``set_value``; any scalar graph is used as the rate directly, as in
        ``cosine_schedule(3e-4, 10_000)(step_counter())``; an unapplied schedule is read off the clock the
        rule counts its own steps on. Default 0.01.
    momentum : float
        Momentum coefficient. A value of 0 (the default) gives plain SGD.
    nesterov : bool
        Use Nesterov momentum. Ignored when ``momentum`` is 0. Default False.

    namespace : str
        Prefix for the state this rule allocates, so two of them in one step keep separate step counters
        rather than colliding on one name. A composed transform such as the momentum ``trace`` names its
        own state. Default is the rule's own name.

    Examples
    --------
    Plain gradient descent by default. Momentum carries a running average of past steps, and
    ``nesterov`` measures the gradient after that carry rather than before:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import compile_train, sgd

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, sgd(learning_rate=0.1, momentum=0.9, nesterov=True))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    # Built once here rather than per invocation, so the velocity it owns is the same buffer on every
    # step compiled from this rule instead of a fresh one each time.
    momentum_trace = trace(momentum, nesterov) if momentum else None

    @reuses_state
    def rule(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        def build_updates(rate: Rate) -> Updates:
            if momentum_trace is None:
                return sgd_updates(
                    loss_gradients_or_updates,
                    parameters,
                    learning_rate=rate,
                    namespace=namespace,
                )
            updates = sgd_updates(
                loss_gradients_or_updates,
                parameters,
                learning_rate=1.0,
                namespace=namespace,
            )
            updates = momentum_trace(updates, parameters)
            return scale(rate)(updates, parameters)

        return _at_learning_rate(learning_rate, namespace, build_updates)

    return rule


def adam(
    learning_rate: LearningRate = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    amsgrad: bool = False,
    *,
    namespace: str = "adam",
) -> Transform:
    """
    Adam optimizer. See :func:`~pytensor_ml.optim.rules.adam_updates` for the update rule.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule, and
    ``namespace`` prefixes the state this rule allocates; see :func:`sgd`.

    Examples
    --------
    The usual first choice: a per-parameter rate adapted from the first and second gradient moments,
    both bias-corrected, so the earliest steps are not damped towards zero:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, adam(learning_rate=1e-3))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    @reuses_state
    def rule(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        return _at_learning_rate(
            learning_rate,
            namespace,
            lambda rate: adam_updates(
                loss_gradients_or_updates,
                parameters,
                learning_rate=rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                amsgrad=amsgrad,
                namespace=namespace,
            ),
        )

    return rule


def adamw(
    learning_rate: LearningRate = 1e-3,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    amsgrad: bool = False,
    mask: Callable[[Parameter], bool] | None = None,
    *,
    namespace: str = "adamw",
) -> Transform:
    """
    AdamW optimizer (Adam with decoupled weight decay). See
    :func:`~pytensor_ml.optim.rules.adamw_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule, and
    ``namespace`` prefixes the state this rule allocates; see :func:`sgd`.

    Examples
    --------
    Adam entangles weight decay with its adaptive rate; this one subtracts the decay from the weights
    directly. A ``mask`` keeps biases and norm scales out of it, which is almost always what you want:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adamw, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, adamw(learning_rate=1e-3, mask=lambda parameter: parameter.ndim > 1))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    @reuses_state
    def rule(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        return _at_learning_rate(
            learning_rate,
            namespace,
            lambda rate: adamw_updates(
                loss_gradients_or_updates,
                parameters,
                learning_rate=rate,
                weight_decay=weight_decay,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                amsgrad=amsgrad,
                mask=mask,
                namespace=namespace,
            ),
        )

    return rule


def nadam(
    learning_rate: LearningRate = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    *,
    namespace: str = "nadam",
) -> Transform:
    """
    Nadam optimizer (Adam with Nesterov momentum). See
    :func:`~pytensor_ml.optim.rules.nadam_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule, and
    ``namespace`` prefixes the state this rule allocates; see :func:`sgd`.

    Examples
    --------
    Adam with Nesterov's look-ahead folded into the first moment, which turns corners a little faster
    than plain Adam on the same rate:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import compile_train, nadam

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, nadam(learning_rate=2e-3))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    @reuses_state
    def rule(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        return _at_learning_rate(
            learning_rate,
            namespace,
            lambda rate: nadam_updates(
                loss_gradients_or_updates,
                parameters,
                learning_rate=rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                namespace=namespace,
            ),
        )

    return rule


def adamax(
    learning_rate: LearningRate = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    *,
    namespace: str = "adamax",
) -> Transform:
    """
    AdaMax optimizer (Adam with an infinity-norm denominator). See
    :func:`~pytensor_ml.optim.rules.adamax_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule, and
    ``namespace`` prefixes the state this rule allocates; see :func:`sgd`.

    Examples
    --------
    Adam's second moment replaced by a running infinity norm, so one outsized gradient cannot shrink
    every step for many iterations afterwards:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adamax, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, adamax(learning_rate=2e-3))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    @reuses_state
    def rule(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        return _at_learning_rate(
            learning_rate,
            namespace,
            lambda rate: adamax_updates(
                loss_gradients_or_updates,
                parameters,
                learning_rate=rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                namespace=namespace,
            ),
        )

    return rule


def rprop(
    learning_rate: float = 1e-2,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    step_min: float = 1e-6,
    step_max: float = 50.0,
    *,
    namespace: str = "rprop",
) -> Transform:
    """
    Rprop optimizer (resilient backpropagation). See :func:`~pytensor_ml.optim.rules.rprop_updates`.

    Unlike the other rules, ``learning_rate`` must be a plain number: it initializes the per-parameter
    step sizes Rprop then adapts, so it never enters the graph and cannot be scheduled or steered.

    ``namespace`` prefixes the state this rule allocates; see :func:`sgd`.

    Examples
    --------
    Steps by the sign of the gradient alone, growing or shrinking a per-parameter step size. It reads a
    sign change as overshoot, so minibatch noise misleads it -- keep it to full-batch objectives:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import compile_train, rprop

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, rprop(learning_rate=1e-2, eta_plus=1.2, eta_minus=0.5))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    _require_numeric_learning_rate(learning_rate)

    @reuses_state
    def rule(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        return rprop_updates(
            loss_gradients_or_updates,
            parameters,
            learning_rate=learning_rate,
            eta_minus=eta_minus,
            eta_plus=eta_plus,
            step_min=step_min,
            step_max=step_max,
            namespace=namespace,
        )

    return rule


def rmsprop(
    learning_rate: LearningRate = 1e-2,
    rho: float = 0.9,
    momentum: float = 0.0,
    epsilon: float = 1e-8,
    centered: bool = False,
    *,
    namespace: str = "rmsprop",
) -> Transform:
    """
    RMSProp optimizer. See :func:`~pytensor_ml.optim.rules.rmsprop_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule, and
    ``namespace`` prefixes the state this rule allocates; see :func:`sgd`.

    Examples
    --------
    Scales each step by a decaying average of squared gradients. Setting ``centered`` subtracts the
    mean gradient first, which estimates variance rather than raw magnitude:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import compile_train, rmsprop

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, rmsprop(learning_rate=1e-2, centered=True))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    @reuses_state
    def rule(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        return _at_learning_rate(
            learning_rate,
            namespace,
            lambda rate: rmsprop_updates(
                loss_gradients_or_updates,
                parameters,
                learning_rate=rate,
                rho=rho,
                momentum=momentum,
                epsilon=epsilon,
                centered=centered,
                namespace=namespace,
            ),
        )

    return rule


def adagrad(
    learning_rate: LearningRate = 0.01,
    epsilon: float = 1e-8,
    *,
    namespace: str = "adagrad",
) -> Transform:
    """
    AdaGrad optimizer. See :func:`~pytensor_ml.optim.rules.adagrad_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule, and
    ``namespace`` prefixes the state this rule allocates; see :func:`sgd`.

    Examples
    --------
    Accumulates every squared gradient it has seen, so the effective rate only ever decreases. That
    suits sparse features and stalls on long runs:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adagrad, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, adagrad(learning_rate=1e-2))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    @reuses_state
    def rule(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        return _at_learning_rate(
            learning_rate,
            namespace,
            lambda rate: adagrad_updates(
                loss_gradients_or_updates,
                parameters,
                learning_rate=rate,
                epsilon=epsilon,
                namespace=namespace,
            ),
        )

    return rule


def adadelta(
    learning_rate: LearningRate = 1.0,
    rho: float = 0.9,
    epsilon: float = 1e-8,
    *,
    namespace: str = "adadelta",
) -> Transform:
    """
    AdaDelta optimizer. See :func:`~pytensor_ml.optim.rules.adadelta_updates`.

    ``learning_rate`` accepts a float, a scalar shared variable, any scalar graph, or a schedule, and
    ``namespace`` prefixes the state this rule allocates; see :func:`sgd`.

    Examples
    --------
    Tracks a window of squared updates alongside squared gradients, so their ratio sets the scale and
    the learning rate stays at its default of 1.0:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adadelta, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, adadelta())
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """

    @reuses_state
    def rule(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        return _at_learning_rate(
            learning_rate,
            namespace,
            lambda rate: adadelta_updates(
                loss_gradients_or_updates,
                parameters,
                learning_rate=rate,
                rho=rho,
                epsilon=epsilon,
                namespace=namespace,
            ),
        )

    return rule
