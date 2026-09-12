from collections.abc import Container, Mapping, Sequence

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.gradient import DisconnectedGrad, ZeroGrad
from pytensor.graph import graph_inputs
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.op import io_connection_pattern
from pytensor.graph.traversal import ancestors
from pytensor.tensor import TensorVariable

from pytensor_ml.base import StatefulOp, update_chain_root
from pytensor_ml.params import NonTrainableParameter, StepCounter, TrainableParameter


def as_output_list(outputs: Variable | Sequence[Variable]) -> list[Variable]:
    """
    Normalize one output, or a sequence of them, to a list.

    Examples
    --------
    Normalize an output argument that may be one variable or a sequence of them, so the code after it has
    one shape to handle:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.pytensorf import as_output_list

        X = Input("X", shape=(None, 64))
        activations = Linear("fc", n_in=64, n_out=32)(X)

        outputs = as_output_list(activations)
    """
    return [outputs] if isinstance(outputs, Variable) else list(outputs)


def _collect_inputs_of_type[T: Variable](
    outputs: Variable | Sequence[Variable],
    variable_type: type[T],
    blockers: Sequence[Variable] | None = None,
) -> list[T]:
    return [
        variable
        for variable in graph_inputs(as_output_list(outputs), blockers=blockers)
        if isinstance(variable, variable_type)
    ]


def collect_graph_inputs(outputs: Variable | Sequence[Variable]) -> list[Variable]:
    """
    Collect the graph inputs that carry data -- everything that is neither a Constant nor shared.

    Examples
    --------
    The placeholders a caller has to supply, in the order the graph puts them. Weights, running
    statistics and RNGs are shared, so they are left out -- the compiled function carries those itself.
    :func:`collect_data_inputs` is the same function under the name the training and serialization
    boundaries use:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import BatchNorm, Dropout, Input, Linear, Sequential
        from pytensor_ml.pytensorf import collect_graph_inputs

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            BatchNorm("bn", n_in=32),
            ReLU(),
            Dropout(p=0.5, random_state=0),
        )
        activations = network(X)

        inputs = collect_graph_inputs(activations)
    """
    return [
        variable
        for variable in graph_inputs(as_output_list(outputs))
        if not isinstance(variable, Constant | SharedVariable)
    ]


# Same set, named for how the training and serialization boundaries talk about it.
collect_data_inputs = collect_graph_inputs


def collect_shared_variables(outputs: Variable | Sequence[Variable]) -> list[SharedVariable]:
    """
    Collect every SharedVariable the graph reads, parameters and RNGs alike.

    Examples
    --------
    Every shared variable in the graph, whichever kind: parameters, running statistics, RNGs and
    training clocks together. A rule's own state is not in the graph, so a checkpoint that resumes a run
    adds :func:`collect_optimizer_state` to this:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import BatchNorm, Dropout, Input, Linear, Sequential
        from pytensor_ml.pytensorf import collect_shared_variables

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            BatchNorm("bn", n_in=32),
            ReLU(),
            Dropout(p=0.5, random_state=0),
        )
        activations = network(X)

        shared = collect_shared_variables(activations)
    """
    return _collect_inputs_of_type(outputs, SharedVariable)


def collect_optimizer_state(
    updates: Mapping[SharedVariable, Variable], parameters: Sequence[SharedVariable]
) -> list[SharedVariable]:
    """
    Collect the state a rule keeps between steps, which no walk of the graph reaches.

    A rule's moments and its step counter are created when the rule is applied and appear only in the
    updates it returns, so a checkpoint assembled from the graph alone saves the parameters and leaves
    the optimizer at step zero.

    Parameters
    ----------
    updates : mapping of SharedVariable to Variable
        What a rule returned, keyed by the variable each expression writes to.
    parameters : sequence of SharedVariable
        The parameters the rule was given, which the result excludes.

    Returns
    -------
    state : list of SharedVariable
        The updated variables that are not in ``parameters``, in the order ``updates`` gives them.

    Examples
    --------
    A checkpoint that resumes a run rather than restarting its optimizer needs both halves, since
    reloading the parameters alone leaves the moments at zero and takes a full-rate step from a
    converged point:

    .. code-block:: python

        import pytensor.tensor as pt

        from pytensor_ml import save_state
        from pytensor_ml.layers import Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam
        from pytensor_ml.pytensorf import (
            collect_optimizer_state,
            collect_shared_variables,
            collect_trainable_params,
        )

        X = pt.tensor("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())
        parameters = collect_trainable_params(loss)
        updates = adam(1e-2)(loss, parameters)

        shared = collect_shared_variables(loss)
        save_state([*shared, *collect_optimizer_state(updates, parameters)], "ckpt.safetensors")
    """
    already_collected = set(parameters)
    return [variable for variable in updates if variable not in already_collected]


def collect_trainable_params(outputs: Variable | Sequence[Variable]) -> list[TrainableParameter]:
    """
    Collect the parameters an optimizer should update.

    Examples
    --------
    The weights an optimizer is allowed to write. This is what a rule differentiates with respect to,
    and what :meth:`~pytensor_ml.model.Model.initialize` redraws:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import BatchNorm, Dropout, Input, Linear, Sequential
        from pytensor_ml.pytensorf import collect_trainable_params

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            BatchNorm("bn", n_in=32),
            ReLU(),
            Dropout(p=0.5, random_state=0),
        )
        activations = network(X)

        parameters = collect_trainable_params(activations)
    """
    return _collect_inputs_of_type(outputs, TrainableParameter)


def collect_differentiable_params(
    outputs: Variable | Sequence[Variable],
) -> list[TrainableParameter]:
    """
    Collect the trainable parameters an optimizer can take gradients of ``outputs`` with respect to.

    Two things disqualify a parameter, and neither subsumes the other. A stop-gradient marker on the only
    path to it -- :func:`pytensor.gradient.disconnected_grad` or :func:`pytensor.gradient.zero_grad` --
    though a parameter reached on both a detached and a live path stays, since the live path still carries
    gradient. Or no connection to the gradient at all, which no marker expresses: an additive constant
    survives in ``outputs`` as written and vanishes once they differentiate, the shape a physics-informed
    loss hits when an output bias is gone by the second derivative.

    Both checks are needed because ``zero_grad`` yields a gradient that is connected and zero, which the
    connection pattern reports as present.

    Parameters
    ----------
    outputs : Variable or sequence of Variable
        One or more graph outputs to trace back from, typically a scalar loss.

    Returns
    -------
    parameters : list of TrainableParameter
        The differentiable parameters, in graph-input order.

    Examples
    --------
    The trainable parameters the loss actually depends on. A parameter reached only through a
    non-differentiable path is left out, since asking for its gradient would raise:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import BatchNorm, Dropout, Input, Linear, Sequential
        from pytensor_ml.pytensorf import collect_differentiable_params

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            BatchNorm("bn", n_in=32),
            ReLU(),
            Dropout(p=0.5, random_state=0),
        )
        activations = network(X)

        parameters = collect_differentiable_params(activations)
    """
    output_list = as_output_list(outputs)
    stop_gradient_outputs = [
        variable
        for variable in ancestors(output_list)
        if variable.owner is not None and isinstance(variable.owner.op, DisconnectedGrad | ZeroGrad)
    ]
    undetached = _collect_inputs_of_type(
        outputs, TrainableParameter, blockers=stop_gradient_outputs
    )
    if not undetached:
        return undetached

    connection_pattern = io_connection_pattern(list(undetached), output_list)
    return [
        parameter
        for parameter, to_outputs in zip(undetached, connection_pattern)
        if any(to_outputs)
    ]


def collect_non_trainable_params(
    outputs: Variable | Sequence[Variable],
) -> list[NonTrainableParameter]:
    """
    Collect the state that training updates without gradients, such as batch-norm running statistics.

    Examples
    --------
    State the model owns but no optimizer may write, such as a batch-norm running mean. The model
    updates these from its own forward pass:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import BatchNorm, Dropout, Input, Linear, Sequential
        from pytensor_ml.pytensorf import collect_non_trainable_params

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            BatchNorm("bn", n_in=32),
            ReLU(),
            Dropout(p=0.5, random_state=0),
        )
        activations = network(X)

        statistics = collect_non_trainable_params(activations)
    """
    return _collect_inputs_of_type(outputs, NonTrainableParameter)


def collect_step_counters(outputs: Variable | Sequence[Variable]) -> list[StepCounter]:
    """
    Collect the training clocks the graph reads.

    Examples
    --------
    The training clocks the graph reads. Each counts steps for a schedule, and every one in a graph
    counts the same steps:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import BatchNorm, Dropout, Input, Linear, Sequential
        from pytensor_ml.pytensorf import collect_step_counters

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            BatchNorm("bn", n_in=32),
            ReLU(),
            Dropout(p=0.5, random_state=0),
        )
        activations = network(X)

        clocks = collect_step_counters(activations)
    """
    return _collect_inputs_of_type(outputs, StepCounter)


def collect_clock_updates(
    outputs: Variable | Sequence[Variable],
    already_written: Container[SharedVariable] = (),
) -> dict[StepCounter, TensorVariable]:
    """
    Collect the advance for every training clock the graph reads, ready to pass as a function's ``updates``.

    A clock advances once per step however many schedules, policies, and diagnostics read it, because
    readers share a clock by referring to the same variable rather than by name.

    Parameters
    ----------
    outputs
        One or more graph outputs to trace back from.
    already_written : container of shared variable, optional
        The variables the caller writes themselves, typically the ``updates`` mapping itself. Any clock
        among them is left out of the result and exempt from the rule that every clock agrees on its step
        count, since a clock this step does not advance is not counting its steps. Default is empty, which
        collects and checks every clock the graph reads.

    Returns
    -------
    clock_updates : dict
        Mapping from each clock the graph reads to the expression for its next value.

    Examples
    --------
    The one-step advance for each training clock in the graph, which is what makes a schedule move rather
    than read step zero forever. :func:`function` threads these in automatically:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.optim import cosine_schedule, get_gradients, scale_by_schedule
        from pytensor_ml.pytensorf import collect_clock_updates, collect_trainable_params

        X = Input("X", shape=(None, 64))
        activations = Linear("fc", n_in=64, n_out=32)(X)

        parameters = collect_trainable_params(activations)
        updates = scale_by_schedule(cosine_schedule(1e-3, 1_000))(
            dict(zip(parameters, get_gradients(activations.sum(), parameters))), parameters
        )

        clock_updates = collect_clock_updates(list(updates.values()))
    """
    counters = [
        counter for counter in collect_step_counters(outputs) if counter not in already_written
    ]
    step_counts = {int(counter.get_value()) for counter in counters}
    if len(step_counts) > 1:
        raise ValueError(
            f"Training clocks {sorted(str(counter.name) for counter in counters)} hold different step "
            f"counts {sorted(step_counts)}. They all count training steps, so this usually means a "
            "checkpoint restored some of them and not the others. Restore all of them, or set them to the "
            "same count before compiling."
        )
    return {counter: counter.advance() for counter in counters}


def collect_non_trainable_updates(
    outputs: Variable | Sequence[Variable],
    already_written: Container[SharedVariable] = (),
) -> dict[NonTrainableParameter, TensorVariable]:
    """
    Collect the write-backs that stateful ops declare, ready to pass as a function's ``updates``.

    A batch-norm statistic is written by the op rather than by any rule, so a graph that reads one has to
    write it back or the model trains against batch statistics and then predicts against the initial ones.
    :func:`function` threads these in automatically; a graph specialized by
    :func:`~pytensor_ml.pytensorf.rewrite.rewrite_for_prediction` no longer holds the op and so declares
    nothing to write.

    Parameters
    ----------
    outputs
        One or more graph outputs to trace back from.
    already_written : container of shared variable, optional
        The variables the caller writes themselves, typically the ``updates`` mapping itself. Any of them
        is left out of the result, so a caller who pins a statistic keeps their own expression for it.
        Default is empty, which collects every write-back the graph declares.

    Returns
    -------
    non_trainable_updates : dict
        Mapping from each NonTrainableParameter to its new value, for every update an op declares through
        :meth:`~pytensor_ml.base.StatefulOp.update_map`.

    Examples
    --------
    The writes the model makes on its own, such as a batch-norm statistic, mapped to their next values.
    :func:`function` threads these into every step it compiles, so reach for this directly only when
    assembling updates for something else:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import BatchNorm, Dropout, Input, Linear, Sequential
        from pytensor_ml.pytensorf import collect_non_trainable_updates

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            BatchNorm("bn", n_in=32),
            ReLU(),
            Dropout(p=0.5, random_state=0),
        )
        activations = network(X)

        updates = collect_non_trainable_updates(activations)
    """
    updates: dict[NonTrainableParameter, TensorVariable] = {}
    depth_of: dict[NonTrainableParameter, int] = {}
    for ancestor in ancestors(as_output_list(outputs)):
        node = ancestor.owner
        if node is None:
            continue
        if not isinstance(node.op, StatefulOp):
            continue
        for output_index, input_index in node.op.update_map().items():
            chain = update_chain_root(node.inputs[input_index])
            if chain is None:
                continue
            parameter, depth = chain
            if not isinstance(parameter, NonTrainableParameter):
                continue
            if parameter in already_written or depth <= depth_of.get(parameter, -1):
                continue
            depth_of[parameter] = depth
            updates[parameter] = node.outputs[output_index]

    return updates


__all__ = [
    "as_output_list",
    "collect_clock_updates",
    "collect_data_inputs",
    "collect_differentiable_params",
    "collect_graph_inputs",
    "collect_non_trainable_params",
    "collect_non_trainable_updates",
    "collect_shared_variables",
    "collect_step_counters",
    "collect_trainable_params",
]
