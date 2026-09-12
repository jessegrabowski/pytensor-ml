from collections.abc import Sequence

from pytensor.graph.basic import Variable

# Importing the package registers every op-family handler into the codec's dispatch tables. Those
# handlers depend on serialize.base, never on this module, so the import runs one way.
import pytensor_ml.serialize  # noqa: F401

from pytensor_ml.serialize.base import (
    const_from_json,
    const_to_json,
    graph_from_json,
    graph_to_json,
    op_from_json,
    op_to_json,
    prop_from_json,
    prop_to_json,
    props_from_json,
    props_to_json,
    qualname,
    register_from_json,
    register_type,
    resolve_class,
    type_from_json,
    type_to_json,
)


def serialize_graph(inputs: Sequence[Variable], outputs: Sequence[Variable]) -> dict:
    """
    Serialize a pytensor graph to a JSON-native dict.

    Parameters
    ----------
    inputs : sequence of Variable
        The graph's inputs, in order. Include every shared variable the graph reads, since the serialized
        form addresses inputs positionally and carries no values.
    outputs : sequence of Variable
        The graph's outputs.

    Returns
    -------
    serialized_graph : dict
        A JSON-native description of the graph's structure (no parameter values).
    """
    return graph_to_json(list(inputs), list(outputs))


def deserialize_graph(
    graph_dict: dict, inputs: Sequence[Variable] | None = None
) -> tuple[list[Variable], list[Variable]]:
    """
    Rebuild a pytensor graph from :func:`serialize_graph` output.

    Parameters
    ----------
    graph_dict : dict
        A graph description produced by :func:`serialize_graph`.
    inputs : sequence of Variable, optional
        Input leaves to build the graph on, in the original order. Supply these to control input identity —
        for example to reattach named shared variables. Fresh, plain symbolic variables are created when
        omitted.

    Returns
    -------
    inputs : list of Variable
        The input variables, in the original order.
    outputs : list of Variable
        The rebuilt graph outputs.
    """
    return graph_from_json(graph_dict, inputs)


__all__ = [
    "const_from_json",
    "const_to_json",
    "deserialize_graph",
    "graph_from_json",
    "graph_to_json",
    "op_from_json",
    "op_to_json",
    "prop_from_json",
    "prop_to_json",
    "props_from_json",
    "props_to_json",
    "qualname",
    "register_from_json",
    "register_type",
    "resolve_class",
    "serialize_graph",
    "type_from_json",
    "type_to_json",
]
