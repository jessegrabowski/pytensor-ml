from typing import TYPE_CHECKING

import numpy as np

from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorVariable

if TYPE_CHECKING:
    from pytensor_ml.state import Initializer


class TrainableParameter(TensorSharedVariable):
    """
    Marker class for trainable parameters (weights, biases).

    Attributes
    ----------
    initializer : Initializer or None
        The law this parameter's value is drawn from, which :func:`~pytensor_ml.state.initialize_params`
        redraws it from. Every layer declares one for each parameter it builds. None means a redraw has
        nothing to go on and raises.
    layer_name : str or None
        The name of the layer that built this parameter, which is the part a checkpoint numbers when
        several layers share a name. None for a parameter no layer owns.

    Examples
    --------
    The shared-variable class :func:`trainable` produces. Graph traversal tells parameters apart by type,
    which is how an optimizer finds exactly the weights it may write:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.params import TrainableParameter, trainable
        from pytensor_ml.pytensorf import collect_trainable_params
        from pytensor_ml.state import ZeroInitializer

        W = trainable(np.zeros((4, 4)), "W", initializer=ZeroInitializer())

        is_trainable = isinstance(W, TrainableParameter)
        found = collect_trainable_params(W.sum())
    """

    initializer: "Initializer | None" = None
    layer_name: str | None = None


class NonTrainableParameter(TensorSharedVariable):
    """
    Marker class for non-trainable state (running mean/var in BatchNorm).

    Attributes
    ----------
    layer_name : str or None
        The name of the layer that built this state, which is the part a checkpoint numbers when several
        layers share a name. None for state no layer owns.

    Examples
    --------
    The shared-variable class :func:`non_trainable` produces. The separate type is what keeps a running
    statistic out of the set an optimizer differentiates and writes:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.params import NonTrainableParameter, non_trainable

        running_mean = non_trainable(np.zeros(32), "bn_running_mean")

        is_non_trainable = isinstance(running_mean, NonTrainableParameter)
    """

    layer_name: str | None = None


class StepCounter(TensorSharedVariable):
    """
    Training time: an integer scalar counting training steps, whose transition is :meth:`advance`.

    Examples
    --------
    The shared-variable class :func:`step_counter` produces. Its type is how
    :func:`~pytensor_ml.pytensorf.collect_clock_updates` finds every clock in a graph and writes each one's
    advance into the compiled step:

    .. code-block:: python

        from pytensor_ml.params import StepCounter, step_counter

        clock = step_counter("step_count")

        is_clock = isinstance(clock, StepCounter)
    """

    def advance(self) -> TensorVariable:
        """Return the expression for this counter's value on the next training step."""
        return self + 1


def _make_parameter[T: TensorSharedVariable](
    parameter_type: type[T], value, name, shape, strict, **kwargs
) -> T:
    value = np.asarray(value)
    if shape is None:
        shape = value.shape
    ttype = TensorType(dtype=str(value.dtype), shape=shape)
    return parameter_type(name=name, type=ttype, value=value, strict=strict, **kwargs)


def trainable(
    value,
    name=None,
    shape=None,
    strict=False,
    initializer: "Initializer | None" = None,
    layer_name: str | None = None,
    **kwargs,
) -> TrainableParameter:
    """
    Create a shared variable marked as a trainable parameter.

    The marker class lets graph traversal tell parameters apart from other shared state, so that an
    optimizer updates exactly these. A parameter also declares the initializer it is drawn from, which is
    what lets one seed reproduce every value in a network.

    Parameters
    ----------
    value : array-like
        Initial value for the parameter.
    name : str, optional
        Name for the parameter. Optimizer state and checkpoints are matched by name, so prefer giving one.
    shape : tuple, optional
        Static shape for the variable. Defaults to the concrete shape of ``value``; pass a tuple with None
        entries for dynamic dimensions, e.g. ``(None, None)`` for a fully dynamic matrix.
    strict : bool, optional
        If True, the value must exactly match the dtype.
    initializer : Initializer, optional
        The law to redraw this parameter from, whether that is a unit scale, a zero bias, or a fan-scaled
        draw. Default None, which leaves the parameter with no law and raises on a redraw.
    layer_name : str, optional
        Name of the layer building this parameter, which a checkpoint numbers when several layers share
        a name. Default None, for a parameter no layer owns.
    **kwargs
        Additional arguments passed to the SharedVariable constructor.

    Examples
    --------
    Declare a weight an optimizer may write. Name it, since optimizer state and checkpoints are matched by
    name, and give it the initializer it should be redrawn from:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.params import trainable
        from pytensor_ml.state import XavierUniformInitializer

        W = trainable(np.zeros((64, 32)), "fc_W", initializer=XavierUniformInitializer())
    """
    parameter = _make_parameter(TrainableParameter, value, name, shape, strict, **kwargs)
    parameter.initializer = initializer
    parameter.layer_name = layer_name
    return parameter


def trainable_parameter(
    name: str, shape: tuple[int, ...], initializer=None, default=None, layer_name: str | None = None
) -> TrainableParameter:
    """
    Build a trainable parameter of ``shape``, drawn by ``initializer``, or by ``default`` if None.

    The layers all resolve a constructor keyword against the draw they would otherwise declare, so this
    is the one place that rule lives.

    Parameters
    ----------
    name : str
        Name for the parameter.
    shape : tuple of int
        Shape to draw.
    initializer : Initializer, optional
        What the caller asked for. ``default`` is used when omitted.
    default : Initializer
        What the layer declares when the caller asks for nothing.
    layer_name : str, optional
        Name of the layer building this parameter, which a checkpoint numbers when several layers share
        a name. Default None, for a parameter no layer owns.
    """
    chosen = default if initializer is None else initializer
    return trainable(chosen.initial_value(shape), name, initializer=chosen, layer_name=layer_name)


def non_trainable(
    value, name=None, shape=None, strict=False, layer_name: str | None = None, **kwargs
) -> NonTrainableParameter:
    """
    Create a shared variable marked as non-trainable state, such as batch norm's running statistics.

    Takes the same arguments as :func:`trainable`, ``layer_name`` included; only the marker class differs,
    which is what keeps these out of the set an optimizer updates.

    Examples
    --------
    Declare state the model owns but no optimizer may write, which is what a batch-norm running mean is.
    It still travels in a checkpoint and still gets restored:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.params import non_trainable

        running_mean = non_trainable(np.zeros(32), "bn_running_mean")
    """
    parameter = _make_parameter(NonTrainableParameter, value, name, shape, strict, **kwargs)
    parameter.layer_name = layer_name
    return parameter


def step_counter(name: str = "step_count") -> StepCounter:
    """
    Create a training clock: an integer scalar counting training steps, starting at zero.

    Schedules and policies read the clock to place themselves in time, and
    :func:`~pytensor_ml.pytensorf.collect_clock_updates` advances it once per step however many of them read
    it. Hold the returned object and pass it where it is needed: readers share a clock by referring to the
    same variable, never by name.

    Parameters
    ----------
    name : str, optional
        Name for the counter. Training state is matched by name at serialization boundaries. Default
        'step_count'.

    Examples
    --------
    Build the clock a schedule reads. Every clock in a graph counts the same steps, and the compiled
    function advances them, so a schedule moves through time rather than reading step zero forever:

    .. code-block:: python

        from pytensor_ml.optim import cosine_schedule
        from pytensor_ml.params import step_counter

        clock = step_counter("my_schedule/step_count")
        rate = cosine_schedule(3e-4, total_steps=10_000)(clock)
    """
    return _make_parameter(
        StepCounter, np.asarray(0, dtype="int64"), name, shape=None, strict=False
    )
