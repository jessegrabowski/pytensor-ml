from itertools import chain

import pytensor.tensor as pt

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.features import Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB
from pytensor.graph.traversal import ancestors, applys_between
from pytensor.scan.op import Scan
from pytensor.scan.utils import ScanArgs, expand_empty, safe_new
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import StatefulOp, update_chain_root
from pytensor_ml.params import NonTrainableParameter

optimize_db = EquilibriumDB()


def _one_step_of_each_inner_input(args: ScanArgs) -> dict[Variable, Variable]:
    """
    Map every inner input to the outer expression holding one step of it.

    A sequence, a recurrent state and an untraced state all arrive as an outer tensor whose leading axis
    is time, so one step of any of them is its first slice; a non-sequence is already one step's worth.
    Rebuilding an inner expression against this map moves it to the outer graph.

    A multi-tap state reaches the inner graph once per tap and the outer graph once in total, and every
    tap holds the same shape, so they all map to that one buffer's first slice.
    """

    def first_step(outer: Variable) -> Variable:
        assert isinstance(outer, TensorVariable), "a scan's sequences and states are tensors"
        return outer[0]

    one_step = {
        inner: first_step(outer)
        for inners, outers in (
            (args.inner_in_seqs, args.outer_in_seqs),
            (args.inner_in_sit_sot, args.outer_in_sit_sot),
            (args.inner_in_shared, args.outer_in_shared),
        )
        for inner, outer in zip(inners, outers)
    }
    for taps, outer in zip(args.inner_in_mit_sot, args.outer_in_mit_sot):
        one_step.update((tap, first_step(outer)) for tap in taps)
    one_step.update(zip(args.inner_in_non_seqs, args.outer_in_non_seqs))
    return one_step


def _varying_inner_inputs(args: ScanArgs) -> set[Variable]:
    """The inner inputs holding a different value at every step, as against the loop's non-sequences."""
    return {
        *args.inner_in_seqs,
        *args.inner_in_sit_sot,
        *chain.from_iterable(args.inner_in_mit_sot),
        *args.inner_in_shared,
    }


def _draws_from_a_frozen_generator(args: ScanArgs) -> list[Apply]:
    """
    Return the draws whose generator the loop never advances, in topological order.

    Such a draw yields one value that every step reuses, so the loop is not a function of a sequence of
    draws at all, and the graph does not compile because nothing can derive the generator's next state. A
    generator carried as a recurrent state is threaded deliberately and is left alone, as is one shared by
    two draws, which would need a generator each to come apart.

    A draw whose parameters vary from step to step is left alone as well. Its values depend on where the
    loop has got to, so a whole sequence of them cannot be drawn before the loop runs; only the shape may
    depend on the state, since every inner variable holds the same shape at every step.
    """
    inner_nodes = list(applys_between(args.inner_inputs, args.inner_outputs))

    # Every reader counts, not just the draws: taking a generator out of the loop has to leave nothing
    # behind that still reads it.
    read_by: dict[Variable, int] = {}
    for node in inner_nodes:
        for variable in node.inputs:
            read_by[variable] = read_by.get(variable, 0) + 1

    consumed = {variable for node in inner_nodes for variable in node.inputs}
    consumed.update(args.inner_outputs)
    varying = _varying_inner_inputs(args)

    hoistable = []
    for draw in inner_nodes:
        if not isinstance(draw.op, RandomVariable):
            continue
        generator, _size, *parameters = draw.inputs
        if (
            generator in args.inner_in_non_seqs
            and read_by[generator] == 1
            # A next state that something reads is a generator the caller threads themselves.
            and draw.outputs[0] not in consumed
            and varying.isdisjoint(ancestors(parameters))
        ):
            hoistable.append(draw)
    return hoistable


@node_rewriter([Scan])
def hoist_draws_out_of_scan(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """
    Draw a whole sequence's worth of randomness outside the loop and feed it back in as a sequence.

    A draw written inside a recurrence reads a generator the loop has no way to advance, so it yields one
    value reused at every step, and the graph does not compile because that generator has no update.
    Lifting the draw out replaces it with ``n_steps`` independent values and gives the generator its
    ordinary update. It is also what makes the loop differentiable: a draw inside the differentiated
    region leaves no fixed sample to take a gradient against, which :func:`pytensor.grad` reports as an
    undefined gradient rather than as a number.

    Parameters
    ----------
    fgraph : FunctionGraph
        Graph being rewritten.
    node : Apply
        The ``Scan`` node being rewritten.

    Returns
    -------
    outputs : list of Variable or None
        The rebuilt scan's outputs, or None when the loop has no such draw.
    """
    args = ScanArgs.from_node(node, clone=True)
    # Two shapes this does not rewrite, because it cannot reach every place a draw might be used: a
    # mit-mot's outputs are a nested list, and a while loop's condition is an output of its own.
    if args.inner_out_mit_mot or args.as_while:
        return None

    draws = _draws_from_a_frozen_generator(args)
    if not draws:
        return None

    one_step = _one_step_of_each_inner_input(args)
    per_step_draws = {}

    for draw in draws:
        generator, size, *parameters = draw.inputs
        # Rebuilt against one step of every inner input, so a size taken from an intermediate -- the
        # shape of the state at this step, say -- becomes an expression the outer graph can evaluate.
        outer_size, *outer_parameters = clone_replace([size, *parameters], replace=one_step)
        # The op's own inference rather than its `size` input, which is None whenever the shape follows
        # from the parameters, and which says nothing about a multivariate draw's core dimensions. Private
        # to pytensor, and the only place here that reaches past its public surface.
        per_step_shape = draw.op._infer_shape(outer_size, outer_parameters)
        _next_generator, drawn = draw.op(
            *outer_parameters,
            rng=one_step[generator],
            size=pt.stack([args.n_steps, *per_step_shape]),
            return_next_rng=True,
        )

        per_step = safe_new(draw.outputs[1], "_per_step")
        per_step_draws[draw.outputs[1]] = per_step
        args.inner_in_seqs.append(per_step)
        args.outer_in_seqs.append(drawn.astype(draw.outputs[1].type.dtype))

        # Read outside now, so the generator leaves the loop's inputs along with the draw.
        position = args.inner_in_non_seqs.index(generator)
        del args.inner_in_non_seqs[position], args.outer_in_non_seqs[position]

    # Every output field, not just the one an RNN happens to use: a draw left behind in any of them would
    # still reference the generator that just left the loop's inputs.
    for field in (
        "inner_out_mit_sot",
        "inner_out_sit_sot",
        "inner_out_nit_sot",
        "inner_out_shared",
    ):
        setattr(args, field, clone_replace(getattr(args, field), replace=per_step_draws))

    rebuilt = Scan(
        args.inner_inputs,
        args.inner_outputs,
        args.info,
        mode=node.op.mode,
        truncate_gradient=node.op.truncate_gradient,
        name=node.op.name,
        allow_gc=node.op.allow_gc,
    )
    outputs = rebuilt(*args.outer_inputs, return_list=True)
    assert isinstance(outputs, list), "return_list=True yields a list"
    return outputs


optimize_db.register("hoist_draws_out_of_scan", hoist_draws_out_of_scan, "hoist_draws", "scan")


class CarriedStatistics(Feature):
    """Holds the write-backs :func:`carry_statistics_through_scan` produced while rewriting a graph."""

    def __init__(self) -> None:
        self.written: dict[NonTrainableParameter, TensorVariable] = {}


def carried_statistics_of(fgraph: FunctionGraph) -> dict[NonTrainableParameter, TensorVariable]:
    """Return the write-backs recorded on ``fgraph``, attaching an empty record if it has none."""
    for feature in fgraph._features:
        if isinstance(feature, CarriedStatistics):
            return feature.written
    record = CarriedStatistics()
    fgraph.attach_feature(record)
    return record.written


def _statistics_to_carry(args: ScanArgs) -> dict[Variable, Variable]:
    """
    Map each non-sequence a stateful op writes to the value the last application leaves it holding.

    A statistic arrives as a non-sequence, so every step reads the value the loop started with and writes
    a result nothing keeps. Applying one layer object twice inside the loop chains the applications, and
    only the deepest holds both, so that is the one whose value has to survive the step.
    """
    carried: dict[Variable, Variable] = {}
    depths: dict[Variable, int] = {}
    for node in applys_between(args.inner_inputs, args.inner_outputs):
        if not isinstance(node.op, StatefulOp):
            continue
        for output_index, input_index in node.op.update_map().items():
            chain = update_chain_root(node.inputs[input_index])
            if chain is None:
                continue
            root, depth = chain
            if root not in args.inner_in_non_seqs or depth <= depths.get(root, -1):
                continue
            depths[root] = depth
            carried[root] = node.outputs[output_index]
    return carried


@node_rewriter([Scan])
def carry_statistics_through_scan(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """
    Turn a statistic a loop writes into a recurrent state, so every step accumulates it.

    A batch norm applied inside a recurrence reads its running statistics as non-sequences, which hold the
    value the loop started with at every step, and writes results the loop does not keep. Carrying the
    statistic as a recurrent state instead feeds each step what the step before it wrote, and leaves the
    accumulated value as an output the caller can write back.

    Parameters
    ----------
    fgraph : FunctionGraph
        Graph being rewritten.
    node : Apply
        The ``Scan`` node being rewritten.

    Returns
    -------
    outputs : list of Variable or None
        The rebuilt scan's outputs, in the order the original node reports them, or None when the loop
        writes no statistic. Each accumulated statistic is recorded on the graph, which
        :func:`~pytensor_ml.pytensorf.rewrite.carry_scan_statistics` returns to its caller.
    """
    args = ScanArgs.from_node(node, clone=True)
    # Two shapes this does not rewrite: a mit-mot's outputs are a nested list, and a while loop runs a
    # number of steps the accumulated value would depend on.
    if args.inner_out_mit_mot or args.as_while:
        return None

    carried = _statistics_to_carry(args)
    if not carried:
        return None

    seeded = []
    for inner_in, new_value in carried.items():
        position = args.inner_in_non_seqs.index(inner_in)
        outer_in = args.outer_in_non_seqs[position]
        if not isinstance(outer_in, NonTrainableParameter):
            continue
        del args.inner_in_non_seqs[position], args.outer_in_non_seqs[position]
        args.inner_in_sit_sot.append(inner_in)
        args.outer_in_sit_sot.append(expand_empty(pt.expand_dims(outer_in, 0), args.n_steps))
        args.inner_out_sit_sot.append(new_value)
        seeded.append(outer_in)

    if not seeded:
        return None

    rebuilt = Scan(
        args.inner_inputs,
        args.inner_outputs,
        args.info,
        mode=node.op.mode,
        truncate_gradient=node.op.truncate_gradient,
        name=node.op.name,
        allow_gc=node.op.allow_gc,
    )
    outputs = rebuilt(*args.outer_inputs, return_list=True)
    assert isinstance(outputs, list), "return_list=True yields a list"

    # Positional replacement, so each original output is matched to the rebuilt one in the same place of
    # the same group rather than by index into the node's outputs: adding a state reorders the groups, but
    # every group except the recurrent states keeps the length it had.
    original = ScanArgs.from_node(node, clone=False)
    rebuilt_args = ScanArgs.from_node(outputs[0].owner, clone=False)
    replacement = {
        was: now
        for group in (
            "outer_out_mit_mot",
            "outer_out_mit_sot",
            "outer_out_sit_sot",
            "outer_out_shared",
            "outer_out_nit_sot",
        )
        for was, now in zip(getattr(original, group), getattr(rebuilt_args, group))
    }

    # Recorded here rather than recovered later: differentiating this loop builds a second one that
    # replays the same recurrence, and nothing about the finished graph says which of the two the caller
    # should write back. The loop that was rewritten is the one that knows.
    written = carried_statistics_of(fgraph)
    for parameter, state in zip(seeded, rebuilt_args.outer_out_sit_sot[-len(seeded) :]):
        assert isinstance(state, TensorVariable), "a scan's recurrent states are tensors"
        written[parameter] = state[-1]

    return [replacement[output] for output in node.outputs]


optimize_db.register(
    "carry_statistics_through_scan", carry_statistics_through_scan, "carry_statistics", "scan"
)


def uncarried_statistics(outputs: list[Variable]) -> list[NonTrainableParameter]:
    """
    Find the statistics a loop still writes to a non-sequence, at any depth of nesting.

    :func:`carry_statistics_through_scan` reaches the loops whose own inner graph holds the stateful op,
    which leaves a layer applied inside a loop within a loop reading a value no step advances. Reporting
    those is what keeps that shape from training against statistics that never move.

    Parameters
    ----------
    outputs : list of Variable
        Graphs to search, after the carry has run.

    Returns
    -------
    parameters : list of NonTrainableParameter
        The parameters still read as non-sequences by a stateful op inside some loop.
    """
    uncarried = []
    # Each loop is visited with a map from the variables of the graph enclosing it to the top-level ones,
    # because a nested loop takes the enclosing loop's inner variables as its own outer inputs, and the
    # parameter behind one is only visible after resolving back up through every level.
    pending: list[tuple[Apply, dict[Variable, Variable]]] = [
        (node, {}) for node in applys_between([], outputs) if isinstance(node.op, Scan)
    ]
    while pending:
        node, enclosing = pending.pop()
        args = ScanArgs.from_node(node, clone=False)
        to_top_level = {
            inner: enclosing.get(outer, outer)
            for inner, outer in zip(args.inner_in_non_seqs, args.outer_in_non_seqs)
        }
        for inner_node in applys_between(args.inner_inputs, args.inner_outputs):
            if isinstance(inner_node.op, Scan):
                pending.append((inner_node, to_top_level))
                continue
            if not isinstance(inner_node.op, StatefulOp):
                continue
            for input_index in inner_node.op.update_map().values():
                chain = update_chain_root(inner_node.inputs[input_index])
                if chain is None:
                    continue
                parameter = to_top_level.get(chain[0])
                if isinstance(parameter, NonTrainableParameter):
                    uncarried.append(parameter)
    return uncarried
