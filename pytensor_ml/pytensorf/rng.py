import warnings

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from pytensor.compile import SharedVariable
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import FunctionGraph, graph_inputs
from pytensor.graph.basic import Apply, equal_computations
from pytensor.graph.fg import Output
from pytensor.scan.op import Scan
from pytensor.tensor.random.op import RandomVariable, RNGConsumerOp
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.random.variable import RandomGeneratorSharedVariable
from pytensor.tensor.variable import Variable

SeedSequenceSeed = None | int | Sequence[int] | np.ndarray | np.random.SeedSequence
RandomSeed = None | int | Sequence[int] | np.ndarray


def atleast_list(x):
    if not isinstance(x, list | tuple):
        return [x]
    return x


# RNG utilities vendored from pymc.pytensorf to keep pytensor as the only runtime dependency.
# collect_default_updates is the load-bearing one: it threads the next-RNG update for every RandomVariable,
# Scan, and OpFromGraph between inputs and outputs, so a compiled function advances its generators instead
# of repeating draws.
def find_rng_nodes(variables: Iterable[Variable]) -> list[RandomGeneratorSharedVariable]:
    """
    Return the shared RNG variables in a graph.

    Examples
    --------
    Locate the random generators a graph draws from, which is what a dropout layer or any other sampling op
    leaves behind:

    .. code-block:: python

        from pytensor_ml.layers import Dropout, Input
        from pytensor_ml.pytensorf import find_rng_nodes

        X = Input("X", shape=(None, 64))
        activations = Dropout(p=0.5, random_state=0)(X)

        generators = find_rng_nodes([activations])
    """
    return [
        node for node in graph_inputs(variables) if isinstance(node, RandomGeneratorSharedVariable)
    ]


def reseed_rngs(rngs: Sequence[SharedVariable], seed: SeedSequenceSeed) -> None:
    """Replace each shared RNG with a fresh generator seeded from ``seed``."""
    bit_generators = [
        np.random.PCG64(sub_seed)
        for sub_seed in np.random.SeedSequence(seed).spawn(len(rngs))  # type: ignore[arg-type]
    ]
    for rng, bit_generator in zip(rngs, bit_generators):
        rng.set_value(np.random.Generator(bit_generator), borrow=True)


def find_generators_drawn_from(
    outputs: Sequence[Variable],
) -> list[RandomGeneratorSharedVariable]:
    """
    Return the shared generators a draw op in this graph consumes.

    Being read is not being consumed: a generator handed back as an output needs no update. One passed
    into an op carrying an inner graph may draw from it or merely accept it, and only that inner graph
    knows which, so it is followed. A draw inside one counts.

    Parameters
    ----------
    outputs : sequence of Variable
        Graph outputs to trace back from.

    Returns
    -------
    generators : list of RandomGeneratorSharedVariable
        The generators a draw op consumes, in graph-input order.
    """
    fgraph = FunctionGraph(outputs=list(outputs), clone=False)
    return [
        generator
        for generator in fgraph.inputs
        if isinstance(generator, RandomGeneratorSharedVariable)
        and _is_drawn_from(fgraph.clients, generator)
    ]


def _inner_counterparts(node: Apply, input_index: int) -> list[Variable]:
    """Return the inner-graph inputs that a node's outer input at ``input_index`` is passed to."""
    op = node.op
    if isinstance(op, Scan):
        mapping = op.get_oinp_iinp_iout_oout_mappings()["inner_inp_from_outer_inp"]
        return [op.inner_inputs[index] for index in mapping.get(input_index, [])]
    if isinstance(op, OpFromGraph):
        return [op.inner_inputs[input_index]]
    return []


def _is_drawn_from(clients: Mapping[Variable, list[tuple[Apply, int]]], variable: Variable) -> bool:
    """
    Report whether a draw op consumes ``variable``, following it into the inner graphs it enters.

    An op carrying an inner graph is not itself a :class:`RNGConsumerOp`, so a generator drawn from only
    inside one is invisible to a check that reads the outer clients alone.
    """
    for client, input_index in clients.get(variable, ()):
        if isinstance(client.op, RNGConsumerOp):
            return True
        inner_inputs = _inner_counterparts(client, input_index)
        if not inner_inputs:
            continue
        inner_clients = client.op.fgraph.clients
        if any(_is_drawn_from(inner_clients, inner_input) for inner_input in inner_inputs):
            return True
    return False


def collect_default_updates_inner_fgraph(node: Apply) -> dict[Variable, Variable]:
    """Collect default RNG updates from a node carrying an inner function graph, mapped to outer variables."""
    op = node.op
    inner_updates = collect_default_updates(
        op.inner_outputs, inputs=op.inner_inputs, must_be_shared=False
    )
    updates = {}
    for rng, update in inner_updates.items():
        input_index = op.inner_inputs.index(rng)
        output_index = op.inner_outputs.index(update)
        updates[node.inputs[input_index]] = node.outputs[output_index]
    return updates


def collect_default_updates(
    outputs: Variable | Sequence[Variable],
    *,
    inputs: Sequence[Variable] | None = None,
    must_be_shared: bool = True,
) -> dict[Variable, Variable]:
    """
    Collect the default next-RNG update for every shared RNG used between ``inputs`` and ``outputs``.

    Parameters
    ----------
    outputs : Variable or sequence of Variable
        Graph outputs whose RNG updates to collect.
    inputs : sequence of Variable, optional
        Inputs above which updates are not collected. Defaults to the graph roots.
    must_be_shared : bool
        Whether to collect updates only for shared-variable RNGs. False is used when recursing into the
        inner graph of an op, whose RNG inputs are not shared. Default True.

    Returns
    -------
    updates : dict mapping Variable to Variable
        Each RNG variable to the expression for its next state.
    """

    def find_default_update(clients, rng: Variable) -> None | Variable:
        rng_clients = clients.get(rng, None)

        # Root case, RNG is not used elsewhere
        if not rng_clients:
            return None

        if len(rng_clients) > 1:
            # Multiple clients are fine if they are identical operations with the same default update.
            all_updates = [
                find_default_update(clients | {rng: [rng_client]}, rng)
                for rng_client in rng_clients
            ]
            updates = [update for update in all_updates if update is not None]
            if not updates:
                return None
            if len(updates) == 1:
                return updates[0]
            update, *other_updates = updates
            if all(equal_computations([update], [other_update]) for other_update in other_updates):
                return update
            warnings.warn(
                f"RNG Variable {rng} has multiple distinct clients {rng_clients}, "
                f"likely due to an inconsistent random graph. No default update will be returned.",
                UserWarning,
            )
            return None

        [client, _] = rng_clients[0]
        client_op = client.op

        match client_op:
            case Output():
                return None
            case RandomVariable():
                # A RandomVariable's first output is always the update of its input RNG.
                next_rng = client.outputs[0]
            case RNGConsumerOp():
                # RandomVariable is a subclass of RNGConsumerOp, specialized above for speed.
                next_rng = client_op.update(client).get(rng)
                if next_rng is None:
                    raise ValueError(f"No update found for at least one RNG used in {client_op}")
            case Scan():
                rng_index = client.inputs.index(rng)
                io_map = client_op.get_oinp_iinp_iout_oout_mappings()["outer_out_from_outer_inp"]
                output_index = io_map.get(rng_index, -1)
                if output_index != -1:
                    next_rng = client.outputs[output_index]
                else:
                    raise ValueError(
                        f"No update found for at least one RNG used in Scan Op {client_op}. Call "
                        "`collect_default_updates` inside the scan function and return what it gives you "
                        "as that step's updates."
                    )
            case OpFromGraph():
                try:
                    next_rng = collect_default_updates_inner_fgraph(client).get(rng)
                    if next_rng is None:
                        return None
                except ValueError as exc:
                    raise ValueError(
                        f"No update found for at least one RNG used in OpFromGraph Op {client_op}. Add "
                        "the advanced generator to the op's outputs, which "
                        "`pt.random.normal(rng=rng, return_next_rng=True)` gives you alongside the draw."
                    ) from exc
            case _:
                # Unknown consumer; the caller must provide an update manually.
                return None

        nested_next_rng = find_default_update(clients, next_rng)
        return next_rng if nested_next_rng is None else nested_next_rng

    if inputs is None:
        inputs = []

    outs = atleast_list(outputs)
    clients = FunctionGraph(outputs=outs, clone=False).clients

    rng_updates = {}
    for input_rng in (
        inp
        for inp in graph_inputs(outs, blockers=inputs)
        if (not must_be_shared or isinstance(inp, SharedVariable))
        and isinstance(inp.type, RandomType)
    ):
        default_update = find_default_update(clients, input_rng)
        if default_update is not None:
            rng_updates[input_rng] = default_update

    return rng_updates
