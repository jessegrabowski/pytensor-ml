import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph.basic import equal_computations
from pytensor.graph.traversal import applys_between
from pytensor.scan import scan
from pytensor.scan.op import Scan
from pytensor.scan.utils import ScanArgs
from pytensor.tensor.random.type import RandomGeneratorType

from pytensor_ml.layers import BatchNorm, Dropout, Linear, PredictionBatchNormLayer, Sequential
from pytensor_ml.layers.dropout import DropoutLayer
from pytensor_ml.layers.norm import BatchNormLayer
from pytensor_ml.pytensorf import compile_predict, function, rewrite_for_prediction
from pytensor_ml.rewriting.layers import specialize_scan_for_prediction

floatX = pytensor.config.floatX


def dropout_layers_in(compiled):
    return [node for node in compiled.maker.fgraph.apply_nodes if isinstance(node.op, DropoutLayer)]


def training_layers_inside(graph):
    """Every dropout and batch-norm node still living in a loop reachable from ``graph``, at any depth."""
    found = []

    def walk(nodes):
        for node in nodes:
            if isinstance(node.op, Scan):
                args = ScanArgs.from_node(node, clone=False)
                inner = list(applys_between(args.inner_inputs, args.inner_outputs))
                found.extend(
                    one for one in inner if isinstance(one.op, DropoutLayer | BatchNormLayer)
                )
                walk(inner)

    walk(pytensor.graph.FunctionGraph(outputs=[graph], clone=False).apply_nodes)
    return found


def test_remove_dropout():
    first_layer = Linear("Layer_1", n_in=6, n_out=3)
    second_layer = Linear("Layer_2", n_in=3, n_out=1)
    X = pt.tensor("X", shape=(None, 6))

    with_dropout = Sequential(
        first_layer, Dropout("Dropout_1", p=0.5), second_layer, Dropout("Dropout_2", p=0.5)
    )(X)
    without_dropout = Sequential(first_layer, second_layer)(X)

    assert equal_computations([rewrite_for_prediction(with_dropout)], [without_dropout])


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no_affine"])
@pytest.mark.parametrize(
    "consumed_downstream", [False, True], ids=["as_graph_output", "consumed_by_a_layer"]
)
def test_rewrite_batch_stats_to_running_average_stats(consumed_downstream, affine):
    linear = Linear("Layer_1", n_in=6, n_out=3)
    batch_norm = BatchNorm("bn", n_in=3, affine=affine)
    # A downstream SymbolicOp type-checks its inputs more strictly than a graph output does.
    head = Linear("Layer_2", n_in=3, n_out=1) if consumed_downstream else None
    X = pt.tensor("X", shape=(None, 6))

    normalized = batch_norm(linear(X))
    trained = head(normalized) if head else normalized

    affine_params = [batch_norm.loc, batch_norm.scale] if affine else []
    prediction_normalized = PredictionBatchNormLayer(
        name="bn", n_in=3, epsilon=batch_norm.epsilon, affine=affine
    )(linear(X), *affine_params, batch_norm.running_mean, batch_norm.running_var)
    expected = head(prediction_normalized) if head else prediction_normalized

    assert equal_computations([rewrite_for_prediction(trained)], [expected])


def test_rewrite_leaves_batch_norm_without_running_stats_alone():
    X = pt.tensor("X", shape=(None, 6))
    normalized = Sequential(
        Linear("Layer_1", n_in=6, n_out=3), BatchNorm("bn", track_running_stats=False)
    )(X)

    assert equal_computations([rewrite_for_prediction(normalized)], [normalized])


def test_a_dropout_that_keeps_everything_is_dropped_at_compile_time():
    """Nothing is sampled at ``p=0``, but the graph still records the layer the user asked for, so the
    saving is the compiler's to make."""
    X = pt.tensor("X", shape=(None, 6))
    built = Dropout("d", p=0.0, random_state=0)(X)

    assert any(
        isinstance(node.op, DropoutLayer)
        for node in pytensor.graph.FunctionGraph(outputs=[built], clone=False).apply_nodes
    )

    compiled = pytensor.function([X], built)
    X_np = np.random.default_rng(0).normal(size=(4, 6)).astype(floatX)

    assert dropout_layers_in(compiled) == []
    np.testing.assert_allclose(compiled(X_np), X_np)


def test_a_dropout_that_keeps_nothing_is_dropped_at_compile_time():
    """Keeping nothing scales the survivors by 1/0, which pytensor evaluates while canonicalizing even
    though the branch it sits on is never selected."""
    X = pt.tensor("X", shape=(None, 6))
    built = Dropout("d", p=1.0, random_state=0)(X)

    assert any(
        isinstance(node.op, DropoutLayer)
        for node in pytensor.graph.FunctionGraph(outputs=[built], clone=False).apply_nodes
    )

    compiled = pytensor.function([X], built)
    X_np = np.random.default_rng(0).normal(size=(4, 6)).astype(floatX)

    assert dropout_layers_in(compiled) == []
    np.testing.assert_allclose(compiled(X_np), np.zeros_like(X_np))


def test_an_ordinary_dropout_survives_compilation():
    """The rewrite reads the probability off the op, so being too eager here would silently switch
    dropout off in training."""
    X = pt.tensor("X", shape=(None, 6))
    compiled = pytensor.function([X], Dropout("d", p=0.5, random_state=0)(X))

    assert len(dropout_layers_in(compiled)) == 1


@pytest.mark.parametrize("p", [0.0, 1.0], ids=["keeps_everything", "keeps_nothing"])
def test_a_degenerate_dropout_is_dropped_through_the_project_compile_path(p):
    """:func:`~pytensor_ml.pytensorf.function` compiles with a mode of its own, which has to keep
    carrying pytensor's specializations or a rewrite registered into them stops reaching real models."""
    X = pt.tensor("X", shape=(None, 6))

    compiled = function([X], Dropout("d", p=p, random_state=0)(X))

    assert dropout_layers_in(compiled) == []


@pytest.mark.parametrize("affine", [True, False], ids=["affine", "no_affine"])
def test_a_graph_reading_a_batch_norm_statistic_still_specializes(affine):
    """The rewrite substitutes the running statistics for the outputs that would have written them, so a
    graph reading one specializes rather than raising."""
    X = pt.tensor("X", shape=(None, 6))
    batch_norm = BatchNorm("bn", n_in=6, affine=affine)
    normalized = batch_norm(X)

    running_mean = np.arange(6.0, dtype=floatX)
    running_var = np.full(6, 2.0, dtype=floatX)
    batch_norm.running_mean.set_value(running_mean)
    batch_norm.running_var.set_value(running_var)

    _, mean, variance = rewrite_for_prediction(
        [normalized, batch_norm.new_running_mean, batch_norm.new_running_var]
    )

    np.testing.assert_allclose(mean.eval(), running_mean)
    np.testing.assert_allclose(variance.eval(), running_var)


def test_a_prediction_function_returning_a_batch_norm_statistic_compiles():
    """The substituted statistics are the layer's own inputs, so the compiled function hands back
    variables it was also given."""
    rng = np.random.default_rng(0)
    X = pt.tensor("X", shape=(None, 6))
    batch_norm = BatchNorm("bn", n_in=6)
    normalized = batch_norm(X)

    running_mean = np.arange(6.0, dtype=floatX)
    running_var = np.full(6, 2.0, dtype=floatX)
    batch_norm.running_mean.set_value(running_mean)
    batch_norm.running_var.set_value(running_var)

    predict = compile_predict(
        [normalized, batch_norm.new_running_mean, batch_norm.new_running_var], inputs=[X]
    )
    X_np = rng.normal(size=(4, 6)).astype(floatX)
    predicted, mean, variance = predict(X_np)

    expected = (X_np - running_mean) / np.sqrt(running_var + batch_norm.epsilon)
    expected = expected * batch_norm.scale.get_value() + batch_norm.loc.get_value()

    np.testing.assert_allclose(predicted, expected, rtol=1e-5)
    np.testing.assert_allclose(mean, running_mean)
    np.testing.assert_allclose(variance, running_var)


def test_a_loop_predicts_without_its_training_layers():
    """A node rewriter matches the nodes of the graph it is handed, and a loop keeps its own, so
    inference inside a recurrence used to keep sampling dropout and reading batch statistics."""
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    linear, batch_norm = Linear("l", n_in=4, n_out=4), BatchNorm("bn", n_in=4)
    dropout = Dropout("d", p=0.5, random_state=0)
    recurrence = scan(
        lambda x_t: dropout(batch_norm(linear(x_t))), sequences=[xseq], return_updates=False
    )

    assert len(training_layers_inside(recurrence)) == 2
    assert training_layers_inside(rewrite_for_prediction(recurrence)) == []


def test_a_loop_predicts_from_the_running_statistics_it_kept():
    rng = np.random.default_rng(0)
    """Determinism alone would also follow from dropping the batch norm, so this pins the values to the
    running statistics rather than to the batch."""
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    linear, batch_norm = Linear("l", n_in=4, n_out=4), BatchNorm("bn", n_in=4)
    recurrence = scan(lambda x_t: batch_norm(linear(x_t)), sequences=[xseq], return_updates=False)

    running_mean = rng.normal(size=4).astype(floatX)
    running_var = rng.uniform(0.5, 2.0, size=4).astype(floatX)
    batch_norm.running_mean.set_value(running_mean)
    batch_norm.running_var.set_value(running_var)

    X = rng.normal(size=(5, 8, 4)).astype(floatX)
    predicted = compile_predict(recurrence, inputs=[xseq])(X)

    weights, bias = linear.W.get_value(), linear.b.get_value()
    loc, scale = batch_norm.loc.get_value(), batch_norm.scale.get_value()
    expected = np.stack(
        [
            ((X[step] @ weights + bias) - running_mean)
            / np.sqrt(running_var + batch_norm.epsilon)
            * scale
            + loc
            for step in range(X.shape[0])
        ]
    )

    np.testing.assert_allclose(predicted, expected, rtol=1e-5)


def test_a_loop_inside_a_loop_predicts_without_its_training_layers():
    rng = np.random.default_rng(1)
    """The carry reaches a nested loop only by rewriting the outer one's graph, which runs the rewriter
    again, so a search that stopped at the first level would leave the inner dropout sampling."""
    outer = pt.tensor("outer", shape=(None, None, None, 4))
    linear, dropout = Linear("l", n_in=4, n_out=4), Dropout("d", p=0.5, random_state=0)
    nested = scan(
        lambda block: scan(
            lambda x_t: dropout(linear(x_t)), sequences=[block], return_updates=False
        ),
        sequences=[outer],
        return_updates=False,
    )

    assert len(training_layers_inside(nested)) == 1
    assert training_layers_inside(rewrite_for_prediction(nested)) == []

    predict = compile_predict(nested, inputs=[outer])
    X = rng.normal(size=(2, 3, 8, 4)).astype(floatX)

    np.testing.assert_allclose(predict(X), predict(X))


def test_a_loop_with_nothing_to_specialize_is_left_alone():
    """The rewrite has to decline cleanly, or an equilibrium database would rebuild every scan forever."""
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    plain = scan(
        lambda x_t: Linear("l", n_in=4, n_out=4)(x_t), sequences=[xseq], return_updates=False
    )
    fgraph = pytensor.graph.FunctionGraph(outputs=[plain], clone=False)
    [node] = [n for n in fgraph.apply_nodes if isinstance(n.op, Scan)]

    assert specialize_scan_for_prediction.transform(fgraph, node) is None


def test_specializing_a_loop_drops_the_generator_its_dropout_read():
    """Removing the dropout leaves the generator it drew from as an input the loop never reads, and a
    loop that takes a generator it cannot advance is one the RNG collection refuses to compile."""
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    linear, dropout = Linear("l", n_in=4, n_out=4), Dropout("d", p=0.5, random_state=0)
    recurrence = scan(lambda x_t: dropout(linear(x_t)), sequences=[xseq], return_updates=False)

    def generators_taken_by(graph):
        [node] = [
            n
            for n in pytensor.graph.FunctionGraph(outputs=[graph], clone=False).apply_nodes
            if isinstance(n.op, Scan)
        ]
        return [inp for inp in node.inputs if isinstance(inp.type, RandomGeneratorType)]

    assert len(generators_taken_by(recurrence)) == 1
    assert generators_taken_by(rewrite_for_prediction(recurrence)) == []


def test_specializing_a_loop_leaves_its_recurrent_state_in_place():
    """Rebuilding the loop to drop the dropout's generator has to leave the outputs corresponding one to
    one, and a carried state is the case where losing that would reorder them."""
    rng = np.random.default_rng(2)
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    linear, dropout = Linear("l", n_in=4, n_out=4), Dropout("d", p=0.5, random_state=0)
    initial_state = pt.zeros((8, 4), dtype=floatX)

    carried = scan(
        lambda x_t, hidden: pt.tanh(dropout(linear(x_t)) + hidden),
        sequences=[xseq],
        outputs_info=[initial_state],
        return_updates=False,
    )
    without_dropout = scan(
        lambda x_t, hidden: pt.tanh(linear(x_t) + hidden),
        sequences=[xseq],
        outputs_info=[initial_state],
        return_updates=False,
    )

    X = rng.normal(size=(5, 8, 4)).astype(floatX)

    # Dropping dropout at inference is the whole contract, so the specialized loop has to agree with the
    # loop that never had one, state and all.
    np.testing.assert_allclose(
        compile_predict(carried, inputs=[xseq])(X),
        pytensor.function([xseq], without_dropout)(X),
    )


def test_a_loop_reading_a_batch_norm_statistic_still_loses_its_dropout():
    """Pytensor swallows a node rewriter's exception and leaves the graph alone, so a raise while
    specializing the loop's own graph would abandon the whole recurrence and leave inference sampling."""
    xseq = pt.tensor("xseq", shape=(None, None, 4))
    linear, batch_norm = Linear("l", n_in=4, n_out=4), BatchNorm("bn", n_in=4)
    dropout = Dropout("d", p=0.5, random_state=0)

    recurrence, _statistics = scan(
        lambda x_t: (dropout(batch_norm(linear(x_t))), batch_norm.new_running_mean),
        sequences=[xseq],
        return_updates=False,
    )

    assert len(training_layers_inside(recurrence)) == 2
    assert training_layers_inside(rewrite_for_prediction(recurrence)) == []
