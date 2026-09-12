import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import config, shared
from pytensor.gradient import DisconnectedInputError, disconnected_grad, grad, zero_grad
from sklearn.datasets import load_digits, make_regression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from pytensor_ml.activations import LeakyReLU, Tanh
from pytensor_ml.layers import BatchNorm, Linear, Sequential
from pytensor_ml.loss import CrossEntropy, SquaredError, supervised_loss
from pytensor_ml.optim import adam, adamw, compile_train, cosine_schedule, sgd
from pytensor_ml.optim.base import state_for
from pytensor_ml.params import step_counter, trainable
from pytensor_ml.pytensorf import collect_non_trainable_params, collect_trainable_params
from pytensor_ml.state import initialize_params
from pytensor_ml.util import DataLoader


@pytest.fixture
def classification_data():
    features, labels = load_digits(return_X_y=True)
    onehot_labels = OneHotEncoder().fit_transform(labels[:, None]).toarray()
    return MinMaxScaler().fit_transform(features), onehot_labels


@pytest.fixture
def regression_data():
    features, target = make_regression(n_samples=1000, n_features=64, noise=10, random_state=0)
    return StandardScaler().fit_transform(features), StandardScaler().fit_transform(target[:, None])


def build_network(n_out: int) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("hidden1", n_in=64, n_out=256),
        LeakyReLU(),
        Linear("hidden2", n_in=256, n_out=128),
        LeakyReLU(),
        Linear("output", n_in=128, n_out=n_out),
    )
    return X, network(X)


def initialize(parameters, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for parameter, value in zip(parameters, initialize_params(parameters, rng=rng)):
        parameter.set_value(value)


def train(step, data, n_steps: int = 50, batch_size: int = 512) -> list[float]:
    dataloader = DataLoader(*data, batch_size=batch_size)
    return [float(step(*dataloader())) for _ in range(n_steps)]


@pytest.mark.parametrize(
    "rule", [sgd(learning_rate=1e-2), adam(learning_rate=1e-2)], ids=["sgd", "adam"]
)
def test_trains_classifier(classification_data, rule):
    X, prediction = build_network(n_out=10)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    loss, target = supervised_loss(
        prediction, CrossEntropy(expect_onehot_labels=True, expect_logits=True)
    )
    step = compile_train(loss, rule, parameters=parameters, inputs=[X, target])

    history = train(step, classification_data)
    assert history[-1] < history[0]


@pytest.mark.parametrize(
    "rule", [sgd(learning_rate=1e-2), adam(learning_rate=1e-2)], ids=["sgd", "adam"]
)
def test_trains_regressor(regression_data, rule):
    X, prediction = build_network(n_out=1)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    loss, target = supervised_loss(prediction, SquaredError())
    step = compile_train(loss, rule, parameters=parameters, inputs=[X, target])

    history = train(step, regression_data)
    assert history[-1] < history[0]


def test_supervised_loss_builds_target_and_scalar_loss():
    X = pt.tensor("X", shape=(None, 64))
    prediction = Linear("output", n_in=64, n_out=10)(X)
    loss, target = supervised_loss(
        prediction, CrossEntropy(expect_onehot_labels=True, expect_logits=True)
    )
    assert target.type.shape == (None, 10)
    assert loss.type.ndim == 0


def test_optimizer_state_is_reachable_from_the_updates_dict():
    # The updates dict is the checkpoint handle: optimizer state is exactly the keys that are not parameters,
    # held by object identity. No wrapper is needed to retain them.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Linear("output", n_in=4, n_out=2)(X)
    parameters = collect_trainable_params(prediction)
    loss, _ = supervised_loss(prediction, SquaredError())

    updates = adam(learning_rate=1e-2)(loss, parameters)
    state = [variable for variable in updates if variable not in set(parameters)]

    # One shared step counter plus a first and second moment per parameter.
    assert len(state) == 1 + 2 * len(parameters)
    assert "adam/step_count" in {variable.name for variable in state}


def test_compile_train_infers_parameters_and_inputs():
    X = pt.tensor("X", shape=(None, 4))
    prediction = Linear("output", n_in=4, n_out=2)(X)
    initialize(collect_trainable_params(prediction))
    loss, _ = supervised_loss(prediction, SquaredError())

    step = compile_train(loss, sgd(1e-2))  # parameters and inputs collected from the graph

    assert callable(step)


def test_compile_train_returns_extra_outputs_in_order():
    X = pt.tensor("X", shape=(None, 4))
    prediction = Linear("output", n_in=4, n_out=2)(X)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    loss, target = supervised_loss(prediction, SquaredError())
    weight_norm = pt.sqrt(sum((parameter**2).sum() for parameter in parameters))

    step = compile_train(
        loss,
        sgd(1e-2),
        parameters=parameters,
        inputs=[X, target],
        extra_outputs=[weight_norm, prediction],
    )

    norm_before_step = np.sqrt(sum((parameter.get_value() ** 2).sum() for parameter in parameters))

    rng = np.random.default_rng(0)
    step_loss, step_norm, step_prediction = step(
        rng.normal(size=(8, 4)).astype(config.floatX), np.zeros((8, 2), dtype=config.floatX)
    )

    assert step_loss.ndim == 0
    assert step_prediction.shape == (8, 2)

    # Extras are evaluated in the gradient pass, so they see the weights the step started from, not the
    # updated ones.
    np.testing.assert_allclose(step_norm, norm_before_step, rtol=1e-5)


def test_compile_train_collects_inputs_of_extra_outputs():
    # An extra output may read data the loss never touches; unless those inputs are collected too,
    # compilation raises MissingInputError. Called by name so the test says nothing about collection order.
    X = pt.tensor("X", shape=(None, 4))
    example_weights = pt.tensor("example_weights", shape=(None,))
    prediction = Linear("output", n_in=4, n_out=2)(X)
    initialize(collect_trainable_params(prediction))
    loss, target = supervised_loss(prediction, SquaredError())

    step = compile_train(loss, sgd(1e-2), extra_outputs=[(example_weights * loss).sum()])

    rng = np.random.default_rng(0)
    step_loss, weighted_loss = step(
        X=rng.normal(size=(8, 4)).astype(config.floatX),
        target=np.zeros((8, 2), dtype=config.floatX),
        example_weights=np.full(8, 2.0, dtype=config.floatX),
    )

    np.testing.assert_allclose(weighted_loss, step_loss * 16.0, rtol=1e-5)


def test_compile_train_ignores_updates_of_extra_outputs():
    # Extras are read-only observers: a stateful op reached only through an extra output must not have its
    # write-back folded into the step, or monitoring would silently mutate training state.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Linear("fc", n_in=4, n_out=4)(X)
    monitor = BatchNorm("bn", n_in=4)(prediction)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    loss, target = supervised_loss(prediction, SquaredError())

    step = compile_train(
        loss,
        sgd(1e-2),
        parameters=parameters,
        inputs=[X, target],
        extra_outputs=[monitor.mean()],
    )

    running_mean = next(
        p for p in collect_non_trainable_params(monitor) if "running_mean" in p.name
    )
    before = running_mean.get_value().copy()

    rng = np.random.default_rng(0)
    step(rng.normal(size=(16, 4)).astype(config.floatX), np.zeros((16, 4), dtype=config.floatX))

    np.testing.assert_allclose(running_mean.get_value(), before)


def test_state_for_requires_named_parameter():
    # An unnamed parameter would leave every state buffer named by its bare slot, so distinct parameters
    # would silently share state at serialization boundaries.
    anonymous = trainable(np.zeros(2))
    with pytest.raises(ValueError, match="unnamed parameter"):
        state_for(anonymous, "adam/first_moment")


def test_compile_train_rejects_duplicate_parameter_names():
    # Two parameters sharing a name give their optimizer state colliding names; compile_train refuses to
    # build a training step whose checkpointed state cannot be told apart.
    first = trainable(np.zeros(1), name="dup")
    second = trainable(np.zeros(1), name="dup")
    loss = ((first + second) ** 2).sum()
    with pytest.raises(ValueError, match="share the name"):
        compile_train(loss, adam(1e-3), parameters=[first, second], inputs=[])


def test_two_rules_over_different_parameter_groups_train_both():
    # Decaying the weights and leaving the biases alone means two rules in one training step. Their
    # rule-wide state is not parameter-scoped, so a name shared between two rules stops the step compiling.
    weight = trainable(np.array([2.0]), name="weight")
    bias = trainable(np.array([3.0]), name="bias")
    loss = 0.5 * ((weight**2).sum() + (bias**2).sum())

    def decay_the_weights_only(loss_or_gradients, parameters):
        return {
            **adamw(learning_rate=0.1)(loss_or_gradients, [weight]),
            **adam(learning_rate=0.1)(loss_or_gradients, [bias]),
        }

    step = compile_train(loss, decay_the_weights_only, parameters=[weight, bias], inputs=[])
    step()

    # Both rules take a first step of -lr * sign(gradient) once bias correction is applied, and adamw adds
    # its decoupled decay of weight_decay * parameter on top.
    np.testing.assert_allclose(weight.get_value(), [2.0 - 0.1 * (1 + 0.01 * 2.0)], rtol=1e-3)
    np.testing.assert_allclose(bias.get_value(), [3.0 - 0.1], rtol=1e-3)


def test_compile_train_includes_non_trainable_updates():
    # compile_train merges batch-norm running-stat updates that a bare gradient rule would omit.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), BatchNorm("bn", n_in=4))(X)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    loss, target = supervised_loss(prediction, SquaredError())
    step = compile_train(loss, sgd(1e-2), parameters=parameters, inputs=[X, target])

    running_mean = next(
        p for p in collect_non_trainable_params(prediction) if "running_mean" in p.name
    )
    before = running_mean.get_value().copy()

    rng = np.random.default_rng(0)
    step(rng.normal(size=(16, 4)).astype(config.floatX), np.zeros((16, 4), dtype=config.floatX))

    assert not np.allclose(running_mean.get_value(), before)


def test_compile_train_leaves_a_stop_gradient_target_network_untouched():
    # The DQN shape: a bootstrap target detached from the graph. Its weights must not move, and the online
    # weights must.
    X = pt.tensor("X", shape=(None, 4))
    online = Linear("online", n_in=4, n_out=2)(X)
    target = Linear("target", n_in=4, n_out=2)(X)
    parameters = collect_trainable_params([online, target])
    initialize(parameters)
    loss = ((online - disconnected_grad(target)) ** 2).mean()

    before = {parameter: parameter.get_value().copy() for parameter in parameters}
    step = compile_train(loss, adam(learning_rate=1e-2), inputs=[X])
    batch = np.random.default_rng(0).normal(size=(32, 4)).astype(config.floatX)
    for _ in range(20):
        step(batch)

    moved = {
        parameter.name: not np.allclose(parameter.get_value(), value)
        for parameter, value in before.items()
    }
    assert moved == {"online_W": True, "online_b": True, "target_W": False, "target_b": False}


def test_compile_train_leaves_a_zero_grad_parameter_untouched_under_weight_decay():
    # A zero gradient alone does not protect a parameter: adamw's decoupled decay shrinks it every step
    # whatever the gradient is, so it has to be left out of the optimizer's set rather than merely handed a
    # zero. adam would hide this, since a zero gradient gives it a zero step.
    X = pt.tensor("X", shape=(None, 4))
    live = Linear("live", n_in=4, n_out=2)(X)
    frozen = Linear("frozen", n_in=4, n_out=2)(X)
    parameters = collect_trainable_params([live, frozen])
    initialize(parameters)
    loss = ((live + zero_grad(frozen)) ** 2).mean()

    before = {parameter: parameter.get_value().copy() for parameter in parameters}
    step = compile_train(loss, adamw(learning_rate=1e-2, weight_decay=0.1), inputs=[X])
    batch = np.random.default_rng(0).normal(size=(32, 4)).astype(config.floatX)
    for _ in range(20):
        step(batch)

    moved = {
        parameter.name: not np.allclose(parameter.get_value(), value)
        for parameter, value in before.items()
    }
    assert moved == {"live_W": True, "live_b": True, "frozen_W": False, "frozen_b": False}


def test_a_parameter_that_differentiates_away_is_skipped_unless_you_name_it():
    """Which policy you get is decided by who chose the parameter set. Collected for you, a parameter the
    loss cannot reach is left out, so a physics-informed loss trains without hand-enumerating the rest.
    Named explicitly, it raises -- you asserted it should train, and it cannot."""
    x = pt.tensor("x", shape=(None, 1))
    u = Sequential(Linear("hidden", n_in=1, n_out=4), Tanh(), Linear("out", n_in=4, n_out=1))(x)
    loss = (grad(grad(u.sum(), x).sum(), x) ** 2).mean()
    every_parameter = collect_trainable_params(loss)
    initialize(every_parameter)
    before = {p.name: p.get_value().copy() for p in every_parameter}

    # Decoupled decay is what distinguishes skipped from handed-a-zero: it moves a parameter whatever the
    # gradient is, so out_b staying put means it is out of the optimizer's set rather than in it with a zero.
    step = compile_train(loss, adamw(learning_rate=1e-2, weight_decay=0.1), inputs=[x])
    batch = np.linspace(-1.0, 1.0, 32).reshape(32, 1).astype(config.floatX)
    history = [float(step(batch)) for _ in range(30)]

    assert history[-1] < history[0]  # training happened, so the assertions below are not vacuous
    after = {p.name: p.get_value() for p in every_parameter}

    # Exact equality, not a tolerance: skipped means no update expression touched it at all.
    np.testing.assert_array_equal(after["out_b"], before["out_b"])
    assert not np.array_equal(
        after["out_W"], before["out_W"]
    )  # its sibling in the same layer moved

    with pytest.raises(DisconnectedInputError, match=r"\['out_b'\]"):
        compile_train(loss, adamw(learning_rate=1e-2), parameters=every_parameter, inputs=[x])


def test_extra_updates_write_state_no_gradient_produces():
    # The DQN shape from the other side: a target network kept as a Polyak average of the online weights.
    # No gradient produces that write, so without extra_updates it cannot ride along in the training step.
    X = pt.tensor("X", shape=(None, 4))
    online_layer = Linear("online", n_in=4, n_out=2)
    prediction = online_layer(X)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    target_weight = shared(online_layer.W.get_value().copy(), name="target_W")
    loss, target = supervised_loss(prediction, SquaredError())

    step = compile_train(
        loss,
        sgd(1e-1),
        parameters=parameters,
        inputs=[X, target],
        extra_updates={target_weight: 0.5 * target_weight + 0.5 * online_layer.W},
    )

    online_before = online_layer.W.get_value().copy()
    target_before = target_weight.get_value().copy()

    rng = np.random.default_rng(0)
    features = rng.normal(size=(16, 4)).astype(config.floatX)
    targets = np.ones((16, 2), dtype=config.floatX)
    step(features, targets)

    # Updates are computed from the pre-update values, so the target lands halfway between where the two
    # started -- neither staying put nor jumping to where the online weights ended up.
    assert not np.allclose(online_layer.W.get_value(), online_before)
    np.testing.assert_allclose(
        target_weight.get_value(), 0.5 * target_before + 0.5 * online_before, rtol=1e-5
    )


def test_extra_updates_accept_a_write_that_agrees_with_the_rule():
    """Two components writing one variable is only a problem when they disagree. A rule builds a fresh
    expression on every invocation, so an extra update carrying a structurally identical one is the same
    write said twice -- which is how components share a quantity rather than fight over it."""
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()
    rule = adam(1e-1)
    first_invocation = rule(loss, [p])
    moment = next(key for key in first_invocation if key.name == "w/adam/first_moment")

    step = compile_train(loss, rule, extra_updates={moment: first_invocation[moment]}, inputs=[])
    step()

    assert not np.allclose(
        moment.get_value(), 0.0
    )  # the write took effect rather than being dropped


def test_a_rule_writing_a_statistic_the_model_owns_is_rejected():
    """A batch-norm statistic is the model's to write. A rule writing it too has no way to win: the model's
    write is folded in second, so the rule's would be replaced rather than merged."""
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), BatchNorm("bn", n_in=4))(X)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    loss, target = supervised_loss(prediction, SquaredError())
    running_mean = next(
        p for p in collect_non_trainable_params(prediction) if "running_mean" in p.name
    )

    def meddling_rule(loss_or_gradients, rule_parameters):
        updates = dict(sgd(1e-2)(loss_or_gradients, rule_parameters))
        updates[running_mean] = running_mean * 0.0
        return updates

    with pytest.raises(ValueError, match="the model's to update"):
        compile_train(loss, meddling_rule, parameters=parameters, inputs=[X, target])


def test_extra_updates_reject_a_write_the_rule_already_makes():
    # Silently overwriting an optimizer buffer would leave the rule configured but not working, for the whole
    # run, so the collision has to be loud.
    p = trainable(np.array([2.0]), name="w")
    loss = 0.5 * (p**2).sum()
    rule = adam(1e-1)
    first_moment = next(key for key in rule(loss, [p]) if key.name == "w/adam/first_moment")

    with pytest.raises(ValueError, match="already writes"):
        compile_train(loss, rule, extra_updates={first_moment: first_moment * 0.0}, inputs=[])


def test_extra_updates_reject_a_write_the_model_already_makes():
    # Batch-norm statistics are written by the model rather than the rule, and collide just the same.
    X = pt.tensor("X", shape=(None, 4))
    prediction = Sequential(Linear("fc", n_in=4, n_out=4), BatchNorm("bn", n_in=4))(X)
    parameters = collect_trainable_params(prediction)
    initialize(parameters)
    loss, target = supervised_loss(prediction, SquaredError())
    running_mean = next(
        p for p in collect_non_trainable_params(prediction) if "running_mean" in p.name
    )

    with pytest.raises(ValueError, match="already writes"):
        compile_train(
            loss,
            sgd(1e-2),
            parameters=parameters,
            inputs=[X, target],
            extra_updates={running_mean: running_mean * 0.0},
        )


def test_extra_updates_contribute_their_data_inputs():
    # An extra update may read data the loss never touches -- replay priorities, an importance weight. Unless
    # those inputs are collected too, compiling raises MissingInputError.
    p = trainable(np.array([2.0]), name="w")
    priorities = shared(np.zeros(3, dtype=config.floatX), name="priorities")
    fresh_priorities = pt.vector("fresh_priorities", shape=(3,))
    loss = 0.5 * (p**2).sum()

    step = compile_train(loss, sgd(1e-1), extra_updates={priorities: fresh_priorities})

    step(fresh_priorities=np.array([1.0, 2.0, 3.0], dtype=config.floatX))
    np.testing.assert_allclose(priorities.get_value(), [1.0, 2.0, 3.0])


def test_an_extra_update_reads_the_clock_and_advances_it_once():
    # An extra update is part of the step, so a clock it reads is a clock the step reads: it advances once
    # per step, and the expression sees the count from before the advance, like the rule's updates do.
    p = trainable(np.array([2.0]), name="w")
    decayed = shared(np.array(1.0, dtype=config.floatX), name="decayed")
    clock = step_counter(name="training_step")
    schedule = cosine_schedule(1.0, 10)
    loss = 0.5 * (p**2).sum()

    step = compile_train(loss, sgd(1e-1), extra_updates={decayed: schedule(clock)}, inputs=[])
    for _ in range(3):
        step()

    assert int(clock.get_value()) == 3
    expected = float(schedule(pt.as_tensor(2, dtype="int64")).eval())
    np.testing.assert_allclose(decayed.get_value(), expected, rtol=1e-6)


def test_an_extra_update_that_draws_noise_advances_its_generator():
    # A perturbed step -- SGLD, exploration noise -- draws inside the update rather than the loss. Nothing
    # else reads that generator, so unless the step advances it every call adds the identical perturbation.
    p = trainable(np.array([2.0]), name="w")
    perturbed = shared(np.zeros(3, dtype=config.floatX), name="perturbed")
    noise_rng = shared(np.random.default_rng(0), name="noise_rng")
    _, noise = pt.random.normal(size=(3,), rng=noise_rng, return_next_rng=True)
    loss = 0.5 * (p**2).sum()

    step = compile_train(
        loss, sgd(1e-1), extra_updates={perturbed: noise.astype(config.floatX)}, inputs=[]
    )
    step()
    first = perturbed.get_value().copy()
    step()

    assert not np.allclose(first, perturbed.get_value())


def test_updates_in_compile_kwargs_are_taken_as_extra_updates():
    # `updates` is what pytensor calls this, so it is the first place a caller looks. Forwarding it would
    # collide with the compiler's own updates argument and raise a TypeError naming an internal function.
    p = trainable(np.array([2.0]), name="w")
    call_count = shared(np.array(0.0, dtype=config.floatX), name="call_count")
    loss = 0.5 * (p**2).sum()

    step = compile_train(
        loss, sgd(1e-1), inputs=[], compile_kwargs={"updates": {call_count: call_count + 1.0}}
    )
    step()
    step()

    assert float(call_count.get_value()) == 2.0


def test_an_update_given_in_both_places_is_rejected():
    p = trainable(np.array([2.0]), name="w")
    call_count = shared(np.array(0.0, dtype=config.floatX), name="call_count")
    loss = 0.5 * (p**2).sum()

    with pytest.raises(ValueError, match="given twice"):
        compile_train(
            loss,
            sgd(1e-1),
            inputs=[],
            extra_updates={call_count: call_count + 1.0},
            compile_kwargs={"updates": {call_count: call_count + 2.0}},
        )


def test_compile_kwargs_is_not_mutated():
    # Taking `updates` out of the caller's dict would empty it, so compiling twice from one settings dict
    # would silently drop the update the second time.
    p = trainable(np.array([2.0]), name="w")
    call_count = shared(np.array(0.0, dtype=config.floatX), name="call_count")
    loss = 0.5 * (p**2).sum()
    compile_kwargs = {"updates": {call_count: call_count + 1.0}}

    compile_train(loss, sgd(1e-1), inputs=[], compile_kwargs=compile_kwargs)
    second = compile_train(loss, sgd(1e-1), inputs=[], compile_kwargs=compile_kwargs)
    second()

    assert "updates" in compile_kwargs
    assert float(call_count.get_value()) == 1.0
