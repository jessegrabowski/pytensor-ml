import pytensor.tensor as pt

from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB, RewriteDatabaseQuery
from pytensor.scan.op import Scan
from pytensor.scan.utils import ScanArgs
from pytensor.tensor.rewriting.basic import register_specialize
from pytensor.tensor.variable import Variable

from pytensor_ml.layers.dropout import DropoutLayer
from pytensor_ml.layers.norm import BatchNormLayer, PredictionBatchNormLayer


@register_specialize
@node_rewriter([DropoutLayer])
def drop_degenerate_dropout(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """
    Replace a dropout that keeps everything, or nothing, with the value it computes.

    Parameters
    ----------
    fgraph : FunctionGraph
        Graph being rewritten.
    node : Apply
        Node being rewritten.

    Returns
    -------
    replacement : Variable or None
        The layer's input when it keeps everything, zeros when it keeps nothing, and None for the
        probabilities in between, which have a mask to apply.
    """
    X, _mask = node.inputs
    if node.op.p == 0.0:
        return [X]
    if node.op.p == 1.0:
        # Keeping nothing scales the survivors by 1/0, which pytensor evaluates while canonicalizing
        # even though the branch it sits on is never selected.
        return [pt.zeros_like(X)]
    return None


predict_db = EquilibriumDB()


@node_rewriter([DropoutLayer])
def remove_dropout_for_prediction(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """
    Replace a dropout layer with its input, dropping it from the graph.

    Parameters
    ----------
    fgraph : FunctionGraph
        Graph being rewritten.
    node : Apply
        Node being rewritten.

    Returns
    -------
    X : Variable
        The dropout layer's input, which becomes the layer's replacement.
    """
    X, _mask = node.inputs
    return [X]


predict_db.register(
    "remove_dropout_for_prediction",
    remove_dropout_for_prediction,
    "basic",
)


@node_rewriter([BatchNormLayer])
def rewrite_batch_stats_to_running_average_stats(
    fgraph: FunctionGraph, node: Apply
) -> list[Variable] | None:
    """
    Replace usage of batch mean and variance with running mean and variance.

    Parameters
    ----------
    fgraph : FunctionGraph
        Graph being rewritten.
    node : Apply
        Node being rewritten.

    Returns
    -------
    X_normalized : Variable
        The input normalized by the accumulated running statistics instead of the batch statistics.
    running_mean : Variable
        The accumulated running mean, unchanged, since prediction writes no statistics.
    running_variance : Variable
        The accumulated running variance, unchanged, on the same terms.
    """
    # The affine parameters are absent when the layer was built with affine=False, so bind them as a
    # variable-length group rather than positionally.
    X, *affine_params, running_mean, running_var = node.inputs

    batch_norm_op = PredictionBatchNormLayer(
        name=f"{node.op.name}",
        n_in=node.op.n_in,
        epsilon=node.op.epsilon,
        affine=node.op.affine,
    )

    X_normalized = batch_norm_op(X, *affine_params, running_mean, running_var)

    return [X_normalized, running_mean, running_var]


predict_db.register(
    "rewrite_batch_stats_to_running_average_stats",
    rewrite_batch_stats_to_running_average_stats,
    "basic",
)


def _holds_a_training_layer(node: Apply) -> bool:
    """Report whether a loop's own graph, or one nested inside it, still holds a layer to specialize."""
    for inner in node.op.fgraph.apply_nodes:
        if isinstance(inner.op, DropoutLayer | BatchNormLayer):
            return True
        if isinstance(inner.op, Scan) and _holds_a_training_layer(inner):
            return True
    return False


def _drop_unread_non_sequences(node: Apply) -> list[Variable]:
    """
    Rebuild ``node`` without the non-sequences its graph no longer reads.

    Removing a dropout takes its draw with it and leaves the generator it read as an input the loop never
    touches, which the RNG collection would then demand an update for.

    Parameters
    ----------
    node : Apply
        The specialized ``Scan`` node, whose inner graph is already rewritten.

    Returns
    -------
    outputs : list of Variable
        The node's outputs, rebuilt without the dead non-sequences, or unchanged when the loop reads
        every non-sequence it takes.
    """
    # Keyed on the node's own inner variables, which is what its client map is built on; the clone below
    # carries different objects and would match nothing. A nested loop is specialized before this runs,
    # so a generator it has stopped reading is already unread here too.
    original = ScanArgs.from_node(node, clone=False)
    unread = [
        position
        for position, inner_input in enumerate(original.inner_in_non_seqs)
        if not node.op.fgraph.clients.get(inner_input)
    ]
    if not unread:
        return list(node.outputs)

    args = ScanArgs.from_node(node, clone=True)
    for position in reversed(unread):
        del args.inner_in_non_seqs[position], args.outer_in_non_seqs[position]

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


@node_rewriter([Scan])
def specialize_scan_for_prediction(fgraph: FunctionGraph, node: Apply) -> list[Variable] | None:
    """
    Apply the prediction rewrites to the graph a loop runs at every step.

    A node rewriter matches only the graph it is handed, and a loop keeps its own, so a layer inside a
    recurrence is otherwise left exactly as it was built.

    Parameters
    ----------
    fgraph : FunctionGraph
        Graph being rewritten.
    node : Apply
        The ``Scan`` node being rewritten.

    Returns
    -------
    outputs : list of Variable or None
        The rebuilt scan's outputs, or None when the loop holds nothing to specialize.
    """
    # The search sees through nesting because a nested loop is only reached by rewriting this one's
    # graph, and it has to be exact: reporting a loop that holds nothing would rebuild it forever.
    if not _holds_a_training_layer(node):
        return None

    inner = node.op.fgraph.unfreeze()
    predict_db.query(RewriteDatabaseQuery(include=["basic"])).rewrite(inner)
    specialized = node.op.clone_with_inner_graph(inner)(*node.inputs, return_list=True)
    assert isinstance(specialized, list), "return_list=True yields a list"
    return _drop_unread_non_sequences(specialized[0].owner)


predict_db.register(
    "specialize_scan_for_prediction",
    specialize_scan_for_prediction,
    "basic",
)
