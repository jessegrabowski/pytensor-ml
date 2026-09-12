from collections.abc import Callable, Sequence
from contextvars import ContextVar
from functools import wraps

import numpy as np
import pytensor

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.gradient import DisconnectedInputError, grad
from pytensor.graph.basic import Variable
from pytensor.graph.op import io_connection_pattern
from pytensor.tensor import TensorVariable
from pytensor.tensor.sharedvar import TensorSharedVariable

from pytensor_ml.params import step_counter
from pytensor_ml.pytensorf import rewrite_pregrad

type Parameter = TensorSharedVariable

# What every rule accepts first: either a scalar loss to differentiate, or gradients already computed.
type LossOrGradients = TensorVariable | Sequence[TensorVariable]


class Updates(dict[SharedVariable, TensorVariable]):
    """
    Pytensor's native ``updates`` contract, and the single currency every transform here speaks.

    Carries the next parameter values *and* the next optimizer-state values in one identity-keyed
    mapping, so a step and the momentum that produced it travel together. Every transform reads what it
    needs as ``updates[parameter] - parameter`` and writes back a new value for the parameter, which is
    what lets one be written without knowing what produced its input.

    What that difference *means* depends on where in a chain the transform sits, so the two positions are
    distinguished by :class:`Gradients` and :class:`Steps` rather than by a bare mapping. Write a result
    with :meth:`replacing` rather than ``|``, which returns a bare ``dict`` and would silently widen a
    transform's own output back to an unplaced mapping.
    """

    def replacing(self, changes: dict[SharedVariable, TensorVariable]) -> "Updates":
        """
        Return these updates with ``changes`` written over them, in the same space.

        Parameters
        ----------
        changes : dict mapping shared variable to TensorVariable
            New values to write, overriding any entry already present for the same variable.

        Returns
        -------
        updates : Updates
            A new updates dict of the same class, so a transform's output stays placed.
        """
        return type(self)({**self, **changes})

    def copy(self) -> "Updates":
        return type(self)(self)


class Gradients(Updates):
    r"""
    Updates carrying gradients: ``updates[parameter] - parameter`` is the gradient :math:`g` itself.

    What :func:`to_updates` produces from a loss, and what everything ahead of the first rule in a chain
    sees. A clip placed here bounds the gradient itself, so a spike never reaches the moment estimates.
    """


class Steps(Updates):
    """
    Updates carrying steps: ``updates[parameter] - parameter`` is the move a rule decided on.

    What every rule returns, and what everything after it in a chain sees. A clip placed here bounds the
    step an adaptive rule already normalized, which is a different and usually weaker guarantee.
    """


# What every transform accepts first. A loss or gradients seed a fresh `Gradients`; an updates dict from
# an earlier stage passes through as whatever it already is. A bare dict is admitted because a
# hand-written transform is free to build one.
type LossGradientsOrUpdates = LossOrGradients | dict[SharedVariable, TensorVariable]

Transform = Callable[[LossGradientsOrUpdates, Sequence[Parameter]], Updates]
"""
What every optimizer, clip, and schedule in this module is: a callable taking a loss, gradients, or an
updates dict, along with the parameters, and returning the updates dict that moves them.

One type covers all of them: ``adam(1e-3)`` and ``clip_by_global_norm(1.0)`` share this signature and so
compose in either order, and :func:`chain` folds them left to right. What distinguishes them is only what
each does to the difference it reads, and position decides whether that difference is a gradient or a
step.

Examples
--------
Write one as a plain function and :func:`chain` accepts it wherever a built-in transform goes. The
updates dict also carries optimizer state and training clocks, so touch only the entries for
``parameters`` -- rewriting the rest would halve a clock's advance as readily as a step.

One that keeps state of its own allocates it with :func:`state_for` and takes a ``namespace``, so that
two of them in one chain write to separate buffers rather than colliding at the serialization boundary.
Wrap it in :func:`reuses_state` to hold those buffers across invocations when it is used outside a chain;
:func:`chain` gives each of its own members a frame already.


.. code-block:: python

    import numpy as np

    from pytensor_ml.layers import Input, Linear
    from pytensor_ml.loss import SquaredError, supervised_loss
    from pytensor_ml.optim import adam, chain, compile_train, to_updates


    def halve_every_step(loss_gradients_or_updates, parameters):
        updates = to_updates(loss_gradients_or_updates, parameters)
        halved = updates.copy()
        for parameter in parameters:
            halved[parameter] = parameter + 0.5 * (updates[parameter] - parameter)
        return halved


    X = Input("X", shape=(None, 4))
    loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

    step = compile_train(loss, chain(adam(1e-3), halve_every_step))
    loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))
"""

type Schedule = Callable[[TensorVariable], TensorVariable]
"""
A learning-rate schedule: symbolic step count in, scalar learning rate out.

Examples
--------
The built-in schedules return one, and any callable of the same shape works in their place:

.. code-block:: python

    import pytensor.tensor as pt

    from pytensor_ml.optim import adam


    def inverse_square_root(step_count):
        return 3e-4 / pt.sqrt(pt.maximum(step_count, 1))


    rule = adam(learning_rate=inverse_square_root)
"""

type Rate = float | Parameter | TensorVariable
"""
A rate a rule multiplies into its step.

Either a baked-in constant, a shared variable to steer from Python with ``set_value`` or to substitute a
schedule into, or any scalar graph, which is what a schedule reading a training clock produces.

Examples
--------
A shared scalar is the form to reach for when the rate has to change mid-run without recompiling:

.. code-block:: python

    import numpy as np

    from pytensor_ml.layers import Input, Linear
    from pytensor_ml.loss import SquaredError, supervised_loss
    from pytensor_ml.optim import compile_train, scalar_state, sgd

    rate = scalar_state("rate", fill_value=0.1)

    X = Input("X", shape=(None, 4))
    loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

    step = compile_train(loss, sgd(learning_rate=rate))
    loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))

    rate.set_value(np.array(0.01, dtype=rate.dtype))
"""

type LearningRate = Rate | Schedule
"""
What an optimizer alias accepts as its rate, adding a schedule that drives it on-graph.

Examples
--------
Every alias takes either form, so a constant can be swapped for a schedule without touching anything
else:

.. code-block:: python

    from pytensor_ml.optim import adam, cosine_schedule

    fixed = adam(learning_rate=3e-4)
    scheduled = adam(learning_rate=cosine_schedule(3e-4, total_steps=10_000))
"""


def to_floatx(value: Rate) -> Rate:
    """
    Return ``value`` at the current ``floatX``, casting only a variable stored at something else.

    A shared variable carries whatever dtype it was allocated with, which need not be the ``floatX`` the
    graph is built under -- restoring a checkpoint into a differently configured session is the ordinary
    way to get there. A learning rate is where it bites: a float64 rate in a float32 graph makes an update
    pytensor refuses, and the error names the parameter rather than the rate behind it.

    A plain number is left alone rather than made into an array, which is where this differs from pymc's
    ``floatX``: a rule given a float literal must build exactly the graph it built before. ``astype``
    already returns the variable itself when the dtype matches, so a well typed graph is untouched too.

    Parameters
    ----------
    value : float or TensorVariable
        A scalar a rule is about to build into its step.

    Examples
    --------
    Cast a rate to ``floatX`` so it cannot silently upcast a float32 graph to float64. A symbolic rate, such
    as one a schedule produced, passes through unchanged:

    .. code-block:: python

        from pytensor_ml.optim import to_floatx

        rate = to_floatx(1e-3)
    """
    return value.astype(pytensor.config.floatX) if isinstance(value, Variable) else value


def get_gradients(
    loss_or_gradients: LossOrGradients,
    parameters: Sequence[Parameter],
) -> list[TensorVariable]:
    """
    Return gradients of the loss with respect to ``parameters``, or pass through precomputed gradients.

    Parameters
    ----------
    loss_or_gradients : TensorVariable or sequence of TensorVariable
        Either a scalar loss to differentiate, or an already-computed list of gradients, one per parameter.
    parameters : sequence of shared tensor variable
        Parameters to differentiate with respect to.

    Returns
    -------
    gradients : list of TensorVariable
        One gradient per parameter, in the order of ``parameters``.

    Examples
    --------
    Take gradients of a loss with respect to the parameters, or pass gradients straight through. Every rule
    calls it first, so a rule can be handed gradients you computed yourself:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import get_gradients
        from pytensor_ml.pytensorf import collect_trainable_params

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        parameters = collect_trainable_params(loss)
        gradients = get_gradients(loss, parameters)
    """
    if isinstance(loss_or_gradients, list | tuple):
        gradients = list(loss_or_gradients)
        if len(gradients) != len(parameters):
            raise ValueError(f"Got {len(gradients)} gradients for {len(parameters)} parameters.")
        return gradients

    loss = rewrite_pregrad(loss_or_gradients)
    try:
        return grad(loss, list(parameters))  # type: ignore[return-value]
    except DisconnectedInputError as error:
        unreachable = _unreachable_parameter_names(loss, parameters)
        if not unreachable:
            raise
        raise DisconnectedInputError(
            f"The loss has no gradient with respect to {unreachable}. Leave them out of `parameters`, or "
            "check that the loss is meant to depend on them: a term that differentiates away, such as an "
            "output bias under a second derivative, is the usual cause."
        ) from error


def to_updates(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
) -> Updates:
    r"""
    Return ``loss_gradients_or_updates`` as an updates dict, differentiating a loss if that is what it is.

    The first line of every transform, which is what lets one accept a loss, gradients, or an earlier
    stage's output through a single argument. A loss or a list of gradients seeds a fresh
    :class:`Gradients` as :math:`\{p: p + g\}`, so the gradient :math:`g` is recoverable as
    ``updates[parameter] - parameter`` by exactly the arithmetic a transform already does to read a step.

    An updates dict is returned as the *same object*, not a copy, so a transform must write its result
    with :meth:`Updates.replacing` or into a :meth:`Updates.copy` rather than assigning into what this
    returns -- mutating it in place would reach back into the dict the previous stage still holds.

    The sign is positive rather than negative so that a bound written for a step means the same thing
    written for a gradient: ``clip_by_value(-0.1, 0.1)`` clips :math:`g` into that interval, not
    :math:`-g`.

    Parameters
    ----------
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        A scalar loss to differentiate, precomputed gradients one per parameter, or an updates dict an
        earlier transform produced.
    parameters : sequence of shared tensor variable
        Parameters the updates are keyed by, in the order gradients are given in.

    Returns
    -------
    updates : Updates
        A loss or gradients as a new :class:`Gradients`, a bare dict as an unplaced :class:`Updates`, and
        an updates dict as itself, keeping whichever space it already carries.

    Examples
    --------
    Open a hand-written transform with it and the transform composes in any position, reading gradients
    at the front of a chain and steps behind a rule, with no branch of its own:

    .. code-block:: python

        from pytensor_ml.optim import to_updates


        def halve(loss_gradients_or_updates, parameters):
            updates = to_updates(loss_gradients_or_updates, parameters)
            halved = updates.copy()
            for parameter in parameters:
                halved[parameter] = parameter + 0.5 * (updates[parameter] - parameter)
            return halved
    """
    if isinstance(loss_gradients_or_updates, Updates):
        return loss_gradients_or_updates
    if isinstance(loss_gradients_or_updates, dict):
        # A hand-written transform is free to build a bare dict, which says nothing about where it sits.
        # Leave it unplaced rather than guessing: the checks that read the space reject a definite mismatch
        # only, so an unplaced mapping passes every one of them rather than tripping the wrong one.
        return Updates(loss_gradients_or_updates)

    gradients = get_gradients(loss_gradients_or_updates, parameters)
    return Gradients(
        {parameter: parameter + gradient for parameter, gradient in zip(parameters, gradients)}
    )


def gradients_to_descend(
    loss_gradients_or_updates: LossGradientsOrUpdates,
    parameters: Sequence[Parameter],
    rule_name: str,
) -> tuple[Updates, list[TensorVariable]]:
    """
    Return the updates a rule was handed and the gradients it descends along.

    The opening line of every rule. Raises when handed :class:`Steps`, which a rule cannot use: it negates
    what it reads, so descending along a step another rule already chose would move the parameters uphill.

    Parameters
    ----------
    loss_gradients_or_updates : TensorVariable, sequence of TensorVariable, or Updates
        Whatever the rule was called with.
    parameters : sequence of shared tensor variable
        Parameters to read gradients for, in the order the result is returned in.
    rule_name : str
        The rule's own name, used to say which one was misplaced.

    Returns
    -------
    incoming : Updates
        The input as an updates dict, carrying any optimizer state an earlier transform wrote.
    gradients : list of TensorVariable
        One gradient per parameter, in the order of ``parameters``.
    """
    incoming = to_updates(loss_gradients_or_updates, parameters)
    if isinstance(incoming, Steps):
        raise ValueError(
            f"{rule_name} was given the step another rule already produced, rather than gradients. A rule "
            "descends along what it reads, so it would negate that step and move the parameters uphill. "
            "Keep one rule in a chain and shape its step with `scale`, `trace`, or a clip after it."
        )
    return incoming, steps_of(incoming, parameters)


def steps_of(updates: Updates, parameters: Sequence[Parameter]) -> list[TensorVariable]:
    """
    Return the amount each parameter's entry moves it by.

    A gradient or a step according to which space ``updates`` carries; see :class:`Gradients` and
    :class:`Steps`.

    Parameters
    ----------
    updates : Updates
        The updates dict to read.
    parameters : sequence of shared tensor variable
        Parameters to read, in the order the result is returned in.

    Returns
    -------
    steps : list of TensorVariable
        ``updates[parameter] - parameter``, one per parameter.
    """
    return [updates[parameter] - parameter for parameter in parameters]


def _unreachable_parameter_names(
    loss: TensorVariable, parameters: Sequence[Parameter]
) -> list[str]:
    """Name the parameters the loss carries no gradient signal to, which pytensor's own error omits."""
    connection_pattern = io_connection_pattern(list(parameters), [loss])
    return [
        parameter.name or str(parameter)
        for parameter, to_the_loss in zip(parameters, connection_pattern)
        if not any(to_the_loss)
    ]


# A per-parameter slot, or the bare name of a rule-wide counter.
type _StateKey = tuple[Parameter, str] | str

# Bound by reuses_state for the duration of one rule invocation; None means "allocate fresh".
_state_buffers: ContextVar[dict[_StateKey, Parameter] | None] = ContextVar(
    "optimizer_state_buffers", default=None
)

# The per-parameter slots claimed so far in the current invocation. Reuse *across* invocations is the
# whole point of the buffers, but two claims on one slot within a single invocation are two components
# allocating over each other, which is otherwise invisible.
_claimed_slots: ContextVar[set[_StateKey] | None] = ContextVar(
    "optimizer_claimed_slots", default=None
)


def reuses_state[**P, R](builds_updates: Callable[P, R]) -> Callable[P, R]:
    """
    Give ``builds_updates`` a private set of optimizer-state buffers, reused on every invocation.

    A configured rule such as ``adam(1e-3)`` reads as a value, so it is natural to compile two training
    functions from one. Without this, each invocation allocates fresh momentum under the *same* derived
    name: the two steps then share parameters but not optimizer state, which is silently wrong at runtime
    and raises only later when both are checkpointed together. The same holds for a composed transform,
    whose state is likewise allocated per call.

    The buffers are keyed per wrapped callable rather than globally so two independently configured
    optimizers stay independent. They are bound dynamically because :func:`state_for` is reached several
    call layers below, and threading a cache down would touch every one of them. Nesting is safe: an inner
    scope restores the outer one on exit, so a rule's own buffers and its enclosing chain's coexist.

    Parameters
    ----------
    builds_updates : callable
        A rule or transform to wrap. Its buffers live as long as the wrapper does.

    Returns
    -------
    with_persistent_state : callable
        ``builds_updates`` with a buffer scope of its own, matching its signature.

    Examples
    --------
    Wrap a hand-written transform that allocates state, so two functions compiled from it drive the same
    buffers rather than each getting fresh ones. :func:`chain` already does this for its own members:

    .. code-block:: python

        from pytensor_ml.optim import reuses_state, state_for, to_updates


        def smooth(decay, namespace="smooth"):
            @reuses_state
            def transform(loss_gradients_or_updates, parameters):
                updates = to_updates(loss_gradients_or_updates, parameters)
                smoothed = updates.copy()
                for parameter in parameters:
                    velocity = state_for(parameter, f"{namespace}/velocity")
                    smoothed[velocity] = decay * velocity + (updates[parameter] - parameter)
                    smoothed[parameter] = parameter + smoothed[velocity]
                return smoothed

            return transform
    """
    buffers: dict[_StateKey, Parameter] = {}

    @wraps(builds_updates)
    def with_persistent_state(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _state_buffers.set(buffers)
        claimed_token = _claimed_slots.set(set())
        try:
            return builds_updates(*args, **kwargs)
        finally:
            _state_buffers.reset(token)
            _claimed_slots.reset(claimed_token)

    return with_persistent_state


def _reuse_or_allocate(key: _StateKey, allocate: Callable[[], Parameter]) -> Parameter:
    buffers = _state_buffers.get()
    if buffers is None:
        return allocate()
    if key not in buffers:
        buffers[key] = allocate()
    return buffers[key]


def state_for(parameter: Parameter, slot: str, fill_value: float = 0.0) -> Parameter:
    """
    Return the optimizer-state shared variable shaped and typed like ``parameter``.

    The variable is named ``"{parameter.name}/{slot}"`` and carries the parameter's layer, so a checkpoint
    numbers it where it numbers the parameter. The name is never used to *find* the variable at runtime --
    callers hold the returned object directly, and reuse within a rule is keyed on the parameter object, so
    two same-named parameters still get distinct buffers rather than silently sharing one.

    Allocates unless the enclosing rule was wrapped in :func:`reuses_state` and already holds this slot.
    Within one invocation a slot belongs to one component: a second claim on it raises rather than handing
    two components the same buffer, which only the later writer's updates would survive.

    Parameters
    ----------
    parameter : shared tensor variable
        The parameter this state accompanies. Its value's shape and dtype define the state's.
    slot : str
        A short role tag for the slot, e.g. ``"adam/first_moment"`` or ``"trace/velocity"``.
    fill_value : float
        Constant to initialize the state with. Default 0.0.

    Returns
    -------
    state : shared tensor variable
        The buffer for this slot, allocated on the first claim and returned again on later ones.

    Examples
    --------
    Allocate the buffer a stateful transform keeps between steps. The slot's namespace is what keeps two
    transforms of the same kind from writing to one buffer:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.optim import state_for
        from pytensor_ml.params import trainable

        weight = trainable(np.zeros(4), name="fc/W")
        velocity = state_for(weight, "trace/velocity")
    """
    if parameter.name is None:
        raise ValueError(
            f"Cannot allocate optimizer state {slot!r} for an unnamed parameter. Stateful optimizers rely on "
            "parameter names to identify their state at serialization boundaries; give the parameter a name."
        )

    key = (parameter, slot)
    claimed = _claimed_slots.get()
    if claimed is not None:
        if key in claimed:
            raise ValueError(
                f"Two components asked for the {slot!r} state of {parameter.name!r} in one step, so the "
                "second would allocate over the first and only its writes would survive. Give one of them "
                "a `namespace` of its own, or wrap it in `reuses_state` so it keeps its own buffers."
            )
        claimed.add(key)

    def allocate() -> Parameter:
        value = parameter.get_value(borrow=True)
        state = pytensor.shared(np.full_like(value, fill_value), name=f"{parameter.name}/{slot}")
        # Keeps `Linear_1_W` and `Linear_1_W/adam/first_moment` numbered onto the same layer.
        state.layer_name = getattr(parameter, "layer_name", None)
        return state

    return _reuse_or_allocate(key, allocate)


def counter(name: str) -> Parameter:
    """
    Return the training clock a component counts its own steps on.

    Reused across invocations of a rule wrapped in :func:`reuses_state`, so the count keeps advancing. A
    :class:`~pytensor_ml.params.StepCounter` rather than a plain shared variable, so a schedule can read
    the same notion of time the rule uses, and :func:`~pytensor_ml.pytensorf.collect_clock_updates` advances
    it for a caller who does not write the advance themselves.

    Parameters
    ----------
    name : str
        Name of the clock, used to match it at serialization boundaries. Two components given the same
        name share one clock, which is how a rule and the schedule driving it count the same steps.

    Returns
    -------
    clock : StepCounter
        The step counter under ``name``, allocated on first use and returned again after that.

    Examples
    --------
    Read a schedule off a clock of your own, which is what a transform does when it applies a rate after
    the rule rather than inside it:

    .. code-block:: python

        from pytensor_ml.optim import cosine_schedule, counter

        rate = cosine_schedule(3e-4, total_steps=10_000)(counter("my_transform/step_count"))
    """
    return _reuse_or_allocate(name, lambda: step_counter(name))


def scalar_state(name: str, fill_value: float = 0.0) -> Parameter:
    """
    Return a floatX scalar shared variable, reused across invocations of a rule wrapped in
    :func:`reuses_state`.

    Parameters
    ----------
    name : str
        Name of the variable, used to match it at serialization boundaries.
    fill_value : float
        Value to initialize it with. Default 0.0.

    Examples
    --------
    Build the scalar a rule or policy keeps between steps. Naming it makes it findable in a checkpoint and
    in a printed graph:

    .. code-block:: python

        from pytensor_ml.optim import scalar_state

        scale = scalar_state("plateau/scale", fill_value=1.0)
    """
    return _reuse_or_allocate(
        name,
        lambda: pytensor.shared(np.asarray(fill_value, dtype=pytensor.config.floatX), name=name),
    )


def require_unique_state_names(updates: Updates) -> None:
    """
    Raise if two distinct shared variables in ``updates`` share a name.

    Optimizer state is matched by name at serialization boundaries, so two buffers with the same name would
    silently alias each other on save or restore. Runtime is unaffected — the updates dict is keyed by object
    identity — so this guards only the serialization contract.

    Parameters
    ----------
    updates : Updates
        The assembled updates dict whose shared-variable keys are checked.
    """
    seen: set[str] = set()
    for variable in updates:
        name = variable.name
        if name is None:
            continue
        if name in seen:
            raise ValueError(
                f"Two distinct shared variables share the name {name!r}. Optimizer state is matched by "
                "name at serialization boundaries, so the two would collide there. Two transforms of the "
                "same kind in one chain are the usual cause: give one of them a `namespace` of its own. "
                "Otherwise two parameters share a name, and one of them needs a different one."
            )
        seen.add(name)


def chain(*transforms: Transform) -> Transform:
    """
    Compose transforms left to right, each reading what the one before it produced.

    Every argument has the same type, so a clip composes ahead of a rule as readily as behind it, and the
    two mean different things. Ahead of the rule the clip sees gradients, so a spike is bounded before it
    reaches the moment estimates; behind it the clip sees the step the rule already decided on, which an
    adaptive rule has normalized to roughly its learning rate whatever the gradient was.

    .. code-block:: python

        stop_the_spike = chain(clip_by_global_norm(1.0), adam(1e-3))
        bound_the_move = chain(adam(1e-3), clip_by_global_norm(1.0))

    A chain is itself a transform, so one composes into another and the result is flat.

    The composed callable owns one set of optimizer-state buffers however many times it is invoked, so two
    training functions compiled from one chain share its momentum rather than each allocating their own.

    Parameters
    ----------
    *transforms : Transform
        Applied in order. The first reads whatever the chain is called with -- a loss, gradients, or an
        updates dict -- and each one after it reads the previous one's output.

    Returns
    -------
    chained : Transform
        A transform applying every argument in sequence.

    Examples
    --------
    Clip the gradients before the rule sees them, which is what bounds an exploding gradient rather than
    the step it produced:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.loss import SquaredError, supervised_loss
        from pytensor_ml.optim import adam, chain, clip_by_global_norm, compile_train

        X = Input("X", shape=(None, 4))
        loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

        step = compile_train(loss, chain(clip_by_global_norm(1.0), adam(1e-3)))
        loss_value = step(np.zeros((8, 4)), np.zeros((8, 1)))

    Put a transform after the rule to act on the step instead, which is where a rate or a decay belongs:

    .. code-block:: python

        from pytensor_ml.optim import adam, chain, clip_by_global_norm, scale

        rule = chain(clip_by_global_norm(1.0), adam(1.0), scale(1e-3))
    """
    if not transforms:
        raise ValueError("chain needs at least one transform.")

    # Each member gets a buffer frame of its own, made once here. Without it a transform that allocates
    # state without wrapping itself falls through to the chain's frame, where a second such transform
    # would claim the same slot and quietly take it over.
    staged = tuple(reuses_state(transform) for transform in transforms)

    @reuses_state
    def combined(
        loss_gradients_or_updates: LossGradientsOrUpdates, parameters: Sequence[Parameter]
    ) -> Updates:
        updates = staged[0](loss_gradients_or_updates, parameters)
        for transform in staged[1:]:
            updates = transform(updates, parameters)
        return updates

    return combined
