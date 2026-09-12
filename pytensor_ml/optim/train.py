from collections.abc import Sequence

from pytensor.compile import Function
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Variable, equal_computations
from pytensor.tensor import TensorVariable

from pytensor_ml.optim.base import (
    Gradients,
    Parameter,
    Steps,
    Transform,
    Updates,
    require_unique_state_names,
)
from pytensor_ml.pytensorf import (
    collect_clock_updates,
    collect_data_inputs,
    collect_differentiable_params,
    collect_non_trainable_updates,
    function,
)


def _inconsistent_update(updates: Updates, variable: SharedVariable, new_value: Variable) -> bool:
    """
    Report whether ``updates`` already writes ``variable`` with an expression other than ``new_value``.

    Two components writing one variable is only a problem when they disagree: a plain merge keeps one write
    and drops the other, silently, leaving something that looks configured and is inert. Writers that agree
    structurally are fine, which is how several components share one quantity.
    """
    written = updates.get(variable)
    return written is not None and not equal_computations([written], [new_value])


def compile_train(
    loss: TensorVariable,
    rule: Transform,
    *,
    parameters: Sequence[Parameter] | None = None,
    inputs: Sequence[Variable] | None = None,
    extra_outputs: Sequence[Variable] | None = None,
    extra_updates: dict[SharedVariable, TensorVariable] | None = None,
    compile_kwargs: dict | None = None,
) -> Function:
    """
    Compile a one-step training function from a loss graph and an update rule.

    Differentiates the loss via ``rule``, applies the resulting updates, folds in any non-trainable state
    updates (such as batch-norm running statistics) and any given in ``extra_updates``, advances every
    training clock the step reads, and compiles. The parameters and data inputs are collected from ``loss``
    unless given explicitly.

    Parameters
    ----------
    loss : TensorVariable
        Scalar loss to minimize.
    rule : Transform
        A configured optimizer ``(loss_gradients_or_updates, parameters) -> Updates``, e.g. ``adam(1e-3)``.
    parameters : sequence of shared tensor variable, optional
        Parameters to optimize. Collected from ``loss`` with :func:`collect_differentiable_params` when
        omitted, so parameters the loss detaches with a stop-gradient are left alone. A detached parameter
        is still initialized and checkpointed, since :func:`collect_trainable_params` reaches it; only the
        optimizer skips it.
    inputs : sequence of Variable, optional
        Data inputs of the compiled function, in call order. Collected from ``loss``, ``extra_outputs`` and
        ``extra_updates`` with :func:`collect_data_inputs` when omitted; pass them explicitly when call order
        matters (e.g. features before targets).
    extra_outputs : sequence of Variable, optional
        Diagnostics to return alongside the loss, such as gradient norms or a batch accuracy. Evaluated in
        the same pass as the gradients, so they see the pre-update parameter values, and they add no
        non-trainable state updates of their own. A random node reached only through an extra output does
        still advance its generator, since :func:`~pytensor_ml.pytensorf.function` threads the next-RNG
        update for every generator the outputs draw from.
    extra_updates : dict, optional
        State the step should write that no gradient produces -- a target-network sync, a Polyak average,
        replay priorities. Mapping from shared variable to its next value, folded in alongside the rule's own
        updates. Raise if the rule or the model already writes one of these variables, since two writes to one
        variable cannot both take effect. An expression here is part of the step like any other, so a clock it
        reads still advances once, and a generator it draws from still advances.
    compile_kwargs : dict, optional
        Extra keyword arguments forwarded to the function compiler. An ``updates`` entry is taken as
        ``extra_updates`` rather than forwarded, since the compiler's own ``updates`` carries the whole
        assembled step. Raise if a variable is given in both.

    Returns
    -------
    step : Function
        The compiled one-step training function, applying every update in place. Returns the loss alone, or
        ``(loss, *extra_outputs)`` when diagnostics were requested.

    Examples
    --------
    Hand it any scalar loss graph and a rule, and it differentiates the loss, applies the rule, and folds in
    every update the graph carries -- optimizer state, batch-norm statistics, RNGs, training clocks. Pass
    ``extra_outputs`` to read a value out alongside the loss:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, compile_train

        X = Input("X", shape=(None, 4))
        prediction = Linear("fc", n_in=4, n_out=1)(X)
        loss, target = supervised_loss(prediction, SquaredError())

        step = compile_train(loss, adam(1e-3), inputs=[X, target], extra_outputs=[prediction])
        loss_value, predictions = step(np.zeros((8, 4)), np.zeros((8, 1)))
    """
    extra_outputs = list(extra_outputs or [])
    extra_updates = dict(extra_updates or {})
    # Copied rather than mutated, so popping the caller's `updates` out does not empty their dict.
    compile_kwargs = dict(compile_kwargs or {})

    # `updates` is pytensor's own name for this, so a caller who puts one in compile_kwargs is asking for
    # what extra_updates does; take it as one instead of letting it collide with this function's own call.
    for variable, new_value in compile_kwargs.pop("updates", {}).items():
        if variable in extra_updates:
            raise ValueError(
                f"The update for {variable.name!r} is given twice, once in `extra_updates` and once in "
                "`compile_kwargs['updates']`. Both are folded into the training step, so keep whichever one "
                "is right and drop the other."
            )
        extra_updates[variable] = new_value

    if parameters is None:
        parameters = collect_differentiable_params(loss)
    if inputs is None:
        inputs = collect_data_inputs([loss, *extra_outputs, *extra_updates.values()])

    result = rule(loss, parameters)
    if isinstance(result, Gradients):
        raise ValueError(
            "The rule returned gradients rather than the steps to take, so every parameter would move "
            "along its gradient -- uphill, away from a minimum. Put an optimizer such as `adam(1e-3)` in "
            "the chain, or `scale(-learning_rate)` to descend along the gradients yourself."
        )
    updates: Updates = Steps(result)

    # Assigned per key rather than merged: SupportsKeysAndGetItem is invariant in its key type, so
    # dict.update rejects the narrower NonTrainableParameter keys.
    for parameter, new_value in collect_non_trainable_updates(loss).items():
        if _inconsistent_update(updates, parameter, new_value):
            raise ValueError(
                f"The model writes {parameter.name!r} from a stateful op, and the rule writes it too, so "
                "the two writes cannot both take effect. A batch-norm statistic is the model's to update; "
                "leave it out of the rule."
            )
        updates[parameter] = new_value

    # Folded in after the rule's and the model's, so a collision with either is caught rather than deciding
    # by insertion order which of the two writes survives.
    for variable, new_value in extra_updates.items():
        if _inconsistent_update(updates, variable, new_value):
            raise ValueError(
                f"The extra update for {variable.name!r} writes a variable the training step already "
                "writes differently, so the two writes cannot both take effect. Optimizer state and "
                "batch-norm statistics are written by the step itself; drop this one, or fold what it does "
                "into the expression that already writes the variable."
            )
        updates[variable] = new_value

    # Collected from the assembled updates rather than from the loss alone: a clock is read by a schedule
    # or a policy, which live in the updates, where an RNG or a running statistic is read by the model.
    for clock, next_count in collect_clock_updates(
        [loss, *extra_outputs, *updates.values()]
    ).items():
        written = updates.get(clock)
        if written is None:
            updates[clock] = next_count
        elif equal_computations([written], [next_count]):
            pass  # the rule advances this clock itself, identically, so its own write stands
        else:
            raise ValueError(
                f"The training clock {clock.name!r} is already advanced by an expression that is not the "
                "one-step advance. A clock advances once per step, so the two writes cannot both take "
                "effect; drop yours, or write it as `clock + 1` if it is the same advance spelled "
                "differently."
            )

    require_unique_state_names(updates)

    outputs = [loss, *extra_outputs] if extra_outputs else loss

    return function(list(inputs), outputs, updates=updates, **compile_kwargs)
