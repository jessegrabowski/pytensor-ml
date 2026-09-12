from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import pytensor.tensor as pt

from pytensor.compile.builders import SymbolicOp
from pytensor.graph.basic import Variable
from pytensor.tensor.variable import TensorVariable


def _check_input_rank(X: TensorVariable, name: str, n_spatial: int) -> None:
    """Reject an input whose rank is not batch, one axis per spatial dimension, then channels."""
    if X.ndim != n_spatial + 2:
        raise ValueError(
            f"{name} takes an input of shape (batch, {', '.join(['spatial'] * n_spatial)}, channels), "
            f"so it needs a {n_spatial + 2}-dimensional input; got a {X.ndim}-dimensional one."
        )


def _resolve_layer_name(
    name: object, default: str, suggested_hyperparameter: str | None = None
) -> str:
    """
    Return the layer's name, rejecting a hyperparameter passed where the name belongs.

    Parameters
    ----------
    name : str or None
        The name the caller passed, annotated as ``object`` because validating it is the point.
        None selects ``default``.
    default : str
        The layer's own name, used when none is given and quoted in the error.
    suggested_hyperparameter : str, optional
        The parameter the error tells the caller to pass by keyword instead. Layers with no
        hyperparameter a name could displace omit it, and the error only reports the bad name.

    Returns
    -------
    resolved : str
        ``name`` when it is a non-empty string, otherwise ``default``.
    """
    if name is not None and not isinstance(name, str):
        message = f"{default}'s `name` must be a string, but got {type(name).__name__} {name!r}."
        if suggested_hyperparameter is not None:
            message += (
                f" Hyperparameters are keyword-only, so pass it by name: "
                f"{default}({suggested_hyperparameter}={name!r})."
            )
        raise TypeError(message)
    return name if name else default


class Layer(ABC):
    """
    Base class for the objects that build layer graphs. Defined here, not in ``pytensor_ml.layers``, so
    that ``pytensor_ml.activations`` can subclass it without a circular import.

    Examples
    --------
    Subclass it when a layer owns parameters or needs a marker op of its own; for anything stateless a
    plain function is enough. The constructor builds the parameters once, and ``__call__`` builds the graph:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.base import Layer
        from pytensor_ml.layers import Input
        from pytensor_ml.params import trainable
        from pytensor_ml.state import ZeroInitializer


        class Bias(Layer):
            def __init__(self, name, n_in):
                self.b = trainable(np.zeros(n_in), f"{name}_b", initializer=ZeroInitializer())

            def __call__(self, X):
                return X + self.b


        activations = Bias("bias", n_in=4)(Input("X", shape=(None, 4)))
    """

    @abstractmethod
    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable: ...


class VariadicLayer(ABC):
    """
    Base class for graph-building layers that consume multiple tensors.

    Examples
    --------
    Declare a layer whose inputs are supplied together:

    .. code-block:: python

        import pytensor.tensor as pt

        from pytensor_ml.base import VariadicLayer

        class Add(VariadicLayer):
            def __call__(self, *inputs: pt.TensorLike) -> pt.TensorVariable:
                left, right = inputs
                return pt.as_tensor(left) + pt.as_tensor(right)

        total = Add()(pt.vector("left"), pt.vector("right"))
    """

    @abstractmethod
    def __call__(self, *inputs: pt.TensorLike) -> pt.TensorVariable: ...


class LayerOp(SymbolicOp):
    """Base class for the library's neural-network ops.

    A ``SymbolicOp`` is an ``OpFromGraph`` whose inner graph is rebuilt from its ``__props__`` and input
    types by :meth:`build_inner_graph`, so equal props with equal inputs yield an identical op. Basing the
    layers on it (rather than plain ``OpFromGraph``) is what lets the numba backend optimize each inner
    graph: ``SymbolicOp`` restores the fgraph-aware ``__eq__``/``__hash__`` that a props-carrying
    ``OpFromGraph`` would otherwise lose, so the ``ofg_inner_graph`` rewrite keeps its optimized inner
    graph instead of discarding it as unchanged.
    """

    __props__: tuple[str, ...] = ()


class UnaryLayerOp(LayerOp):
    """A ``LayerOp`` with exactly one output, typed as such.

    ``SymbolicOp.__call__`` is annotated ``Variable | list[Variable]`` because an op may produce many
    outputs; a unary layer op produces one, so narrow the result to ``TensorVariable``. The ``isinstance``
    guard narrows for the type checker without a cast and asserts the invariant at runtime.
    """

    def __call__(self, *inputs, **kwargs) -> TensorVariable:
        out = super().__call__(*inputs, **kwargs)
        assert isinstance(out, TensorVariable), f"{type(self).__name__} produced multiple outputs"
        return out


@runtime_checkable
class StatefulOp(Protocol):
    """An op that writes some of its outputs back to shared variables it takes as inputs, the way batch
    norm updates its running statistics. Defining :meth:`update_map` is what marks an op stateful -- the
    check is structural, so there is nothing to register."""

    def update_map(self) -> dict[int, int]:
        """Map each output index to the index of the input that output updates."""


def update_chain_root(variable: Variable) -> tuple[Variable, int] | None:
    """
    Trace a stateful op's update input back to the value it ultimately writes, and count the applications
    behind it.

    Applying one layer object again feeds it what the previous application produced, so the input is the
    written value itself only at the head of that chain. The depth orders the chain, and the deepest
    application is the one holding every earlier contribution.

    Parameters
    ----------
    variable : Variable
        The input a stateful op declares itself to update.

    Returns
    -------
    root : Variable or None
        The value written at the head of the chain, or None if the chain breaks at an op that declares no
        update for the output it reaches.
    depth : int
        How many applications separate ``variable`` from ``root``.
    """
    depth = 0
    while True:
        node = variable.owner
        if node is None or not isinstance(node.op, StatefulOp):
            return variable, depth
        input_index = node.op.update_map().get(node.outputs.index(variable))
        if input_index is None:
            return None
        variable = node.inputs[input_index]
        depth += 1


__all__ = ["Layer", "LayerOp", "StatefulOp", "UnaryLayerOp", "VariadicLayer", "update_chain_root"]
