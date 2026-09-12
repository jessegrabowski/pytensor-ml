import importlib

from collections.abc import Callable, Sequence
from functools import singledispatch
from typing import Any

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.op import Op
from pytensor.graph.traversal import io_toposort
from pytensor.graph.type import Type
from pytensor.scalar.basic import ScalarType
from pytensor.scalar.basic import constant as scalar_constant
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomGeneratorType
from pytensor.tensor.type import TensorType
from pytensor.tensor.type_other import NoneConst, NoneTypeT

# Ordered rather than keyed by class, because a Type is selected with isinstance and so a subclass must be
# offered its own handler before any supertype's. Newest registration wins (see register_type), which is what
# lets a caller override a built-in for a Type they subclass.
#
# The handler signatures are erased to Any on the varying side because the table is heterogeneous; each
# entry's own types are still checked at its register_type call, which is generic over the Type.
_TYPE_TO_JSON: list[tuple[type[Type], Callable[[Any], dict]]] = []
_TYPE_FROM_JSON: dict[str, Callable[[dict], Any]] = {}


def register_type[T: Type](
    kind: str,
    graph_type: type[T],
    to_json: Callable[[T], dict],
    from_json: Callable[[dict], T],
) -> None:
    """
    Register the handlers that encode and decode one pytensor Type.

    Parameters
    ----------
    kind : str
        Tag written into the JSON, and the key its decoder is found under.
    graph_type : type of Type
        Matched with ``isinstance``. Registering last wins, so a subclass of an already-registered Type
        takes precedence over it -- the built-ins register at import time, so appending instead would make
        them impossible to override.
    to_json : callable
        Maps an instance of ``graph_type`` to a JSON-native dict carrying its ``"kind"``.
    from_json : callable
        Rebuilds the Type from that dict.
    """
    _TYPE_TO_JSON.insert(0, (graph_type, to_json))
    _TYPE_FROM_JSON[kind] = from_json


def type_to_json(graph_type: Type) -> dict:
    for candidate, to_json in _TYPE_TO_JSON:
        if isinstance(graph_type, candidate):
            return to_json(graph_type)
    raise TypeError(f"Unserializable type: {graph_type!r}")


def type_from_json(type_dict: dict) -> Any:
    """Rebuild a pytensor Type from its JSON dict. Returns ``Any`` rather than ``Type`` because the concrete
    subclass depends on the registered ``kind``, and callers read subclass attributes such as ``dtype``."""
    kind = type_dict["kind"]
    if kind not in _TYPE_FROM_JSON:
        raise ValueError(f"Unknown type kind: {kind!r}")
    return _TYPE_FROM_JSON[kind](type_dict)


register_type(
    "tensor",
    TensorType,
    lambda graph_type: {
        "kind": "tensor",
        "dtype": graph_type.dtype,
        "shape": list(graph_type.shape),
    },
    lambda type_dict: TensorType(type_dict["dtype"], tuple(type_dict["shape"])),
)
register_type(
    "scalar",
    ScalarType,
    lambda graph_type: {"kind": "scalar", "dtype": graph_type.dtype},
    lambda type_dict: ScalarType(type_dict["dtype"]),
)
register_type(
    "random_generator",
    RandomGeneratorType,
    lambda graph_type: {"kind": "random_generator"},
    lambda type_dict: RandomGeneratorType(),
)
register_type(
    "none",
    NoneTypeT,
    lambda graph_type: {"kind": "none"},
    lambda type_dict: NoneTypeT(),
)


def prop_to_json(value):
    if isinstance(value, tuple):
        return {"__tuple__": [prop_to_json(element) for element in value]}
    if isinstance(value, slice):
        return {"__slice__": [prop_to_json(part) for part in (value.start, value.stop, value.step)]}
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    raise TypeError(f"Unserializable op prop: {value!r} ({type(value).__name__})")


def prop_from_json(value):
    if isinstance(value, dict):
        if "__tuple__" in value:
            return tuple(prop_from_json(element) for element in value["__tuple__"])
        if "__slice__" in value:
            return slice(*(prop_from_json(part) for part in value["__slice__"]))
    return value


def _encode_nonfinite(value):
    """Replace inf/-inf/nan floats with sentinels. JSON has no literals for them, so a constant such as
    the causal mask's -inf would serialize to a non-standard ``-Infinity`` token that strict, portable
    JSON parsers reject."""
    if isinstance(value, list):
        return [_encode_nonfinite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"__float__": "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")}
    return value


def _decode_nonfinite(value):
    if isinstance(value, list):
        return [_decode_nonfinite(item) for item in value]
    if isinstance(value, dict) and "__float__" in value:
        return float(value["__float__"])
    return value


def const_to_json(constant: Constant) -> dict:
    if isinstance(constant.type, NoneTypeT):
        return {"type": {"kind": "none"}}
    value = _encode_nonfinite(np.asarray(constant.data).tolist())
    return {"type": type_to_json(constant.type), "value": value}


def const_from_json(const_dict: dict):
    graph_type = type_from_json(const_dict["type"])
    if isinstance(graph_type, NoneTypeT):
        return NoneConst
    value = np.asarray(_decode_nonfinite(const_dict["value"]), dtype=graph_type.dtype)
    # Use the type-specific constant wrappers: a raw Constant holds an unhashable ndarray and breaks the
    # FrozenApply interning that reconstruction relies on.
    if isinstance(graph_type, ScalarType):
        return scalar_constant(value.item(), dtype=graph_type.dtype)
    return pt.constant(value, dtype=graph_type.dtype)


def qualname(op: object) -> str:
    """Return the import path of an object's class. Note this makes a class's module part of the on-disk
    format: moving it to another module changes what :func:`resolve_class` must find. Takes an instance
    rather than a class, and anything carrying ``__props__`` rather than ops alone."""
    return f"{type(op).__module__}.{type(op).__name__}"


def resolve_class(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def props_to_json(op: object) -> dict:
    """Encode an object's ``__props__`` values, or an empty dict for one that declares none."""
    return {name: prop_to_json(getattr(op, name)) for name in getattr(op, "__props__", ())}


def props_from_json(props_dict: dict) -> dict:
    """Decode a props dict into keyword arguments for the op's constructor."""
    return {name: prop_from_json(value) for name, value in props_dict.items()}


def leaf_to_json(op: Op) -> dict:
    """Encode an op fully described by its class and ``__props__``, needing no inner graph."""
    return {"family": "leaf", "type": qualname(op), "props": props_to_json(op)}


@singledispatch
def op_to_json(op: Op) -> dict:
    """Serialize an op to a JSON dict, dispatching on op type. The default treats the op as a leaf;
    structural ops register their own rules in ``pytensor_ml.serialize``."""
    return leaf_to_json(op)


_OP_FROM_JSON: dict[str, Callable[[dict], Op]] = {}


def register_from_json(family: str) -> Callable[[Callable[[dict], Op]], Callable[[dict], Op]]:
    """Register the handler that rebuilds an op from a JSON dict tagged with ``family``."""

    def register(handler: Callable[[dict], Op]) -> Callable[[dict], Op]:
        _OP_FROM_JSON[family] = handler
        return handler

    return register


def op_from_json(op_dict: dict) -> Op:
    """Rebuild an op from its JSON dict, dispatching on the ``family`` tag."""
    return _OP_FROM_JSON[op_dict["family"]](op_dict)


@register_from_json("leaf")
def _leaf_from_json(op_dict: dict):
    return resolve_class(op_dict["type"])(**props_from_json(op_dict["props"]))


def graph_to_json(inputs: Sequence[Variable], outputs: Sequence[Variable]) -> dict:
    """Serialize a graph to a dict of input types, op nodes, and output references."""
    nodes = io_toposort(inputs, outputs)
    # Keyed by id() rather than by the variable: a Constant wraps an ndarray and is not reliably hashable,
    # so it cannot be a dict key. Safe here because every variable stays alive for the call.
    reference_by_id: dict[int, dict] = {
        id(inp): {"input": index} for index, inp in enumerate(inputs)
    }
    for node_index, node in enumerate(nodes):
        for output_index, out in enumerate(node.outputs):
            reference_by_id[id(out)] = {"node": node_index, "out": output_index}

    def make_ref(variable: Variable) -> dict:
        existing = reference_by_id.get(id(variable))
        if existing is not None:
            return existing
        if isinstance(variable, Constant):
            return {"const": const_to_json(variable)}
        raise ValueError(f"Unresolved variable while serializing graph: {variable!r}")

    return {
        "inputs": [type_to_json(inp.type) for inp in inputs],
        "nodes": [
            {
                "op": op_to_json(node.op),
                "inputs": [make_ref(inp) for inp in node.inputs],
                "outputs": [type_to_json(out.type) for out in node.outputs],
            }
            for node in nodes
        ],
        "outputs": [make_ref(out) for out in outputs],
    }


def graph_from_json(
    graph_dict: dict, inputs: Sequence[Variable] | None = None
) -> tuple[list[Variable], list[Variable]]:
    """Rebuild a graph from :func:`graph_to_json` output, onto ``inputs`` if given, else fresh leaves."""
    if inputs is None:
        input_leaves = [type_from_json(type_dict)() for type_dict in graph_dict["inputs"]]
    else:
        input_leaves = list(inputs)
    built: dict[tuple[int, int], Variable] = {}

    def resolve_ref(reference: dict):
        if "input" in reference:
            return input_leaves[reference["input"]]
        if "node" in reference:
            return built[(reference["node"], reference["out"])]
        if "const" in reference:
            return const_from_json(reference["const"])
        raise ValueError(f"Bad reference: {reference!r}")

    for node_index, node in enumerate(graph_dict["nodes"]):
        op = op_from_json(node["op"])
        node_inputs = [resolve_ref(reference) for reference in node["inputs"]]
        # A RandomVariable's __call__ reorders the distribution params, so rebuild it through make_node.
        # Everything else needs __call__, which for OpFromGraph also builds the inner fgraph.
        if isinstance(op, RandomVariable):
            node_outputs = op.make_node(*node_inputs).outputs
        else:
            result = op(*node_inputs)
            node_outputs = list(result) if isinstance(result, list | tuple) else [result]
        for output_index, out in enumerate(node_outputs):
            built[(node_index, output_index)] = out

    return input_leaves, [resolve_ref(reference) for reference in graph_dict["outputs"]]
