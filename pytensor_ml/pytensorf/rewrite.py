from pytensor.graph import FunctionGraph, RewriteDatabaseQuery, rewrite_graph
from pytensor.tensor.variable import Variable

from pytensor_ml.rewriting.scan import carried_statistics_of, optimize_db, uncarried_statistics


def hoist_scan_draws(outputs):
    """
    Lift any draw written inside a scan out of the loop, across a whole set of graphs at once.

    Rewriting the outputs and the updates together keeps a subgraph they share shared, so a loop read by
    both is lifted once. See :func:`~pytensor_ml.rewriting.scan.hoist_draws_out_of_scan` for what the
    lift does and why a draw cannot stay in the loop.

    Parameters
    ----------
    outputs : sequence of Variable
        Graphs to rewrite, which are left untouched; the rewrite runs on a clone.

    Returns
    -------
    rewritten : list of Variable
        The rewritten graphs, in the order given.
    """
    fgraph = FunctionGraph(outputs=list(outputs), clone=True, copy_inputs=False)
    optimize_db.query(RewriteDatabaseQuery(include=["hoist_draws"])).rewrite(fgraph)
    return list(fgraph.outputs)


def carry_scan_statistics(outputs):
    """
    Turn every statistic a loop writes into a recurrent state, across a whole set of graphs at once.

    Rewriting the outputs and the updates together keeps a subgraph they share shared, so a loop read by
    both is carried once. See
    :func:`~pytensor_ml.rewriting.scan.carry_statistics_through_scan` for what the carry does and why a
    statistic cannot stay a non-sequence.

    Parameters
    ----------
    outputs : sequence of Variable
        Graphs to rewrite, which are left untouched; the rewrite runs on a clone.

    Returns
    -------
    rewritten : list of Variable
        The rewritten graphs, in the order given.
    carried : dict
        Mapping from each statistic's parameter to the value its loop leaves it holding, recorded by the
        rewrite as it ran.
    """
    fgraph = FunctionGraph(outputs=list(outputs), clone=True, copy_inputs=False)
    optimize_db.query(RewriteDatabaseQuery(include=["carry_statistics"])).rewrite(fgraph)

    stranded = uncarried_statistics(list(fgraph.outputs))
    if stranded:
        names = sorted({str(parameter.name or parameter) for parameter in stranded})
        raise NotImplementedError(
            f"{names} are written inside a loop nested in another loop, where the statistics cannot be "
            "carried, so every step would read the value the run started with. Apply the layer in the "
            "outer loop, or build it with track_running_stats=False."
        )

    return list(fgraph.outputs), carried_statistics_of(fgraph)


def rewrite_pregrad(graph):
    """
    Apply simplifying or stabilizing rewrites to graph that are safe to use pre-grad.

    Holds back the canonicalization that splices out pytensor's gradient markers, so a stop-gradient in
    ``graph`` still reaches ``grad``. Lifts a draw written inside a scan out of the loop, which
    :func:`grad` needs rather than merely tolerates: a draw inside the differentiated region leaves no
    fixed sample to take a gradient against, and scan reports it as an undefined gradient.

    Examples
    --------
    Apply the rewrites that have to run before differentiation, which is what every rule does to a loss
    before taking its gradient:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.pytensorf import rewrite_pregrad

        X = Input("X", shape=(None, 64))
        loss = Linear("fc", n_in=64, n_out=32)(X).sum()

        prepared = rewrite_pregrad(loss)
    """
    simplified = rewrite_graph(
        graph, include=("canonicalize", "stabilize"), exclude=("local_view_op",)
    )
    [hoisted] = hoist_scan_draws([simplified])
    # Carried before the gradient is taken, not after: differentiating a loop builds a second one holding
    # a copy of the same recurrence, and a statistic left uncarried here would be carried in both.
    [carried], _ = carry_scan_statistics([hoisted])
    return carried


def rewrite_for_prediction(graph):
    """
    Apply rewrites to specialize a graph for forward passes (e.g. removing Dropout layers).

    Parameters
    ----------
    graph : FunctionGraph, Variable, or sequence of Variable
        The graph to specialize.

    Returns
    -------
    specialized_graph : FunctionGraph, Variable, or list of Variable
        The specialized graph, matching the form of ``graph``. A FunctionGraph is rewritten in place and
        returned; a Variable or sequence is rewritten on a clone, leaving the original untouched.

    Examples
    --------
    Specialize a training graph for inference without compiling it, which is what :func:`compile_predict`
    does first. A Variable or a sequence is rewritten on a clone, leaving the original untouched:

    .. code-block:: python

        from pytensor_ml.layers import Dropout, Input, Linear, Sequential
        from pytensor_ml.pytensorf import rewrite_for_prediction

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            Dropout(p=0.5),
        )
        activations = network(X)

        inference_graph = rewrite_for_prediction(activations)
    """
    # Local by design, matching pytensor's own op/rewrite pattern: the rewrites import the layer ops they
    # match on, so a module-scope import would tie this module to the whole layer surface.
    from pytensor_ml.rewriting.layers import predict_db

    rewriter = predict_db.query(RewriteDatabaseQuery(include=["basic"]))

    if isinstance(graph, FunctionGraph):
        rewriter.rewrite(graph)
        return graph

    has_single_output = isinstance(graph, Variable)
    fgraph = FunctionGraph(
        outputs=[graph] if has_single_output else list(graph), clone=True, copy_inputs=False
    )
    rewriter.rewrite(fgraph)

    return fgraph.outputs[0] if has_single_output else fgraph.outputs
