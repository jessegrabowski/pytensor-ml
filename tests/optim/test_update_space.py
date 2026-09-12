import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor import config

from pytensor_ml.layers import Input, Linear
from pytensor_ml.loss import SquaredError, supervised_loss
from pytensor_ml.optim import (
    Gradients,
    Steps,
    Updates,
    adadelta,
    adagrad,
    adam,
    adamax,
    adamw,
    add_weight_decay,
    chain,
    clip_by_global_norm,
    clip_by_value,
    compile_train,
    cosine_schedule,
    nadam,
    reduce_on_plateau,
    rmsprop,
    rprop,
    scalar_state,
    scale,
    scale_by_schedule,
    sgd,
    state_for,
    to_updates,
    trace,
)
from pytensor_ml.params import trainable
from pytensor_ml.pytensorf import function


def spiking_problem():
    """A parameter whose gradient is read straight off a shared variable, so a spike can be dialled in."""
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    gradient = pytensor.shared(np.ones(3, dtype=config.floatX), name="g")
    return parameter, gradient


STEPS_AFTER_SPIKE = 19


def run_after_spike(rule, n_steps=STEPS_AFTER_SPIKE + 1, spike=1e6):
    """Take one exploding step, then ``n_steps - 1`` ordinary ones, and report where the parameter got to."""
    parameter, gradient = spiking_problem()
    step = pytensor.function([], parameter, updates=rule([gradient], [parameter]))

    gradient.set_value(np.full(3, spike, dtype=config.floatX))
    step()
    gradient.set_value(np.ones(3, dtype=config.floatX))
    for _ in range(n_steps - 1):
        value = step()
    return value


def stateful_transform(decay, slot="smooth/velocity"):
    """A hand-written stateful transform, written the way the ``Transform`` docstring teaches: with
    ``state_for`` and without ``reuses_state``, since nothing at the call site suggests the decorator."""

    def transform(loss_gradients_or_updates, parameters):
        updates = to_updates(loss_gradients_or_updates, parameters)
        smoothed = updates.copy()
        for parameter in parameters:
            velocity = state_for(parameter, slot)
            new_velocity = decay * velocity + (updates[parameter] - parameter)
            smoothed[velocity] = new_velocity
            smoothed[parameter] = parameter + new_velocity
        return smoothed

    return transform


def test_clipping_before_adam_survives_a_gradient_spike():
    """The reason the two spaces are distinguished at all.

    Adam normalizes its step to roughly the learning rate whatever the gradient was, so a clip behind it
    never fires while the spike still lands in the moment estimates and stalls every step after it. The
    same clip ahead of it bounds the gradient, and training proceeds at full speed.
    """
    clipped_gradient = run_after_spike(chain(clip_by_global_norm(1.0), adam(1e-3)))
    clipped_step = run_after_spike(chain(adam(1e-3), clip_by_global_norm(1.0)))

    np.testing.assert_allclose(clipped_gradient, np.full(3, -0.019), rtol=1e-3)
    assert abs(clipped_gradient[0]) > 3 * abs(clipped_step[0])


def test_clipping_before_the_rule_matches_clipping_the_gradients_by_hand():
    """Handing a rule pre-clipped gradients is the workaround the chain replaces, so the two must agree."""
    parameter, gradient = spiking_problem()
    by_hand = pytensor.function(
        [],
        parameter,
        updates=adam(1e-3)([pt.clip(gradient, -1.0, 1.0)], [parameter]),
    )

    gradient.set_value(np.full(3, 1e6, dtype=config.floatX))
    by_hand()
    gradient.set_value(np.ones(3, dtype=config.floatX))
    for _ in range(STEPS_AFTER_SPIKE):
        expected = by_hand()

    np.testing.assert_allclose(
        run_after_spike(chain(clip_by_value(-1.0, 1.0), adam(1e-3))), expected, rtol=1e-6
    )


@pytest.mark.parametrize(
    "build_rule",
    [
        lambda: adam(1e-3),
        lambda: chain(adam(1e-3)),
        lambda: chain(clip_by_global_norm(1.0), adam(1e-3)),
        lambda: chain(adam(1e-3), clip_by_global_norm(1.0)),
        lambda: chain(clip_by_global_norm(1.0), adam(1e-3), scale(0.5)),
        lambda: chain(clip_by_value(-1.0, 1.0), trace(0.9), adam(1e-3)),
    ],
    ids=["bare", "wrapped", "before", "after", "both-sides", "gradient-momentum"],
)
def test_every_ordering_composes_and_trains(build_rule):
    """Rules and transforms share one type, so any arrangement of them has to build, run, and descend."""
    parameter, gradient = spiking_problem()
    step = pytensor.function([], parameter, updates=build_rule()([gradient], [parameter]))

    trajectory = np.array([step() for _ in range(5)])

    assert np.all(np.isfinite(trajectory))
    # A constant positive gradient moves a parameter down by some amount on every step; a final sign
    # alone would also be satisfied by a rule that moves once and then stalls.
    assert np.all(np.diff(trajectory, axis=0) < 0.0)


def test_a_chain_wrapping_a_rule_matches_the_bare_rule():
    """chain() of one transform is that transform, so the round trip through an updates dict is inert."""
    np.testing.assert_allclose(
        run_after_spike(chain(adam(1e-3))), run_after_spike(adam(1e-3)), rtol=1e-12
    )


def test_to_updates_seeds_gradients_recoverable_by_subtraction():
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    gradient = pt.constant(np.array([1.0, 2.0, 3.0], dtype=config.floatX))

    updates = to_updates([gradient], [parameter])

    assert isinstance(updates, Gradients)
    np.testing.assert_allclose(function([], updates[parameter] - parameter)(), [1.0, 2.0, 3.0])


def test_to_updates_passes_an_updates_dict_through_unchanged():
    parameter = trainable(np.zeros(2, dtype=config.floatX), name="w")
    already = Steps({parameter: parameter + 1.0})

    assert to_updates(already, [parameter]) is already


def test_to_updates_leaves_a_plain_dict_unplaced():
    """A hand-written transform may build a bare dict, which says nothing about where it sits. Guessing a
    space for it would trip whichever check disagreed with the guess, so it stays unplaced and passes
    both."""
    parameter = trainable(np.zeros(2, dtype=config.floatX), name="w")

    unplaced = to_updates({parameter: parameter + 1.0}, [parameter])

    assert isinstance(unplaced, Updates)
    assert not isinstance(unplaced, Gradients | Steps)


def test_a_rule_turns_gradients_into_steps():
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    gradient = pt.constant(np.ones(3, dtype=config.floatX))

    assert isinstance(sgd(0.1)([gradient], [parameter]), Steps)


def test_a_clip_keeps_whichever_space_it_was_given():
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    gradient = pt.constant(np.ones(3, dtype=config.floatX))

    clip = clip_by_global_norm(1.0)
    assert isinstance(clip(Gradients({parameter: parameter + gradient}), [parameter]), Gradients)
    assert isinstance(clip(Steps({parameter: parameter + gradient}), [parameter]), Steps)


def test_writing_a_result_keeps_the_space():
    """Every transform writes its result over what it was given, so a write that dropped the subclass would
    silently widen the output back to an unplaced mapping and disarm every check downstream."""
    parameter = trainable(np.zeros(2, dtype=config.floatX), name="w")
    gradients = Gradients({parameter: parameter + 1.0})

    assert isinstance(gradients.replacing({}), Gradients)
    assert isinstance(gradients.copy(), Gradients)
    # `|` is what would be reached for by habit, and is exactly what `replacing` exists to displace.
    assert not isinstance(gradients | {}, Gradients)


def test_replacing_overrides_only_the_named_entries():
    parameter = trainable(np.zeros(2, dtype=config.floatX), name="w")
    clock = pytensor.shared(np.array(0), name="clock")
    updates = Steps({parameter: parameter + 1.0, clock: clock + 1})

    rewritten = updates.replacing({parameter: parameter + 2.0})

    assert rewritten[clock] is updates[clock]
    np.testing.assert_allclose(function([], rewritten[parameter] - parameter)(), [2.0, 2.0])


@pytest.mark.parametrize(
    "transform",
    [scale(0.5), scale_by_schedule(cosine_schedule(3e-4, 10))],
    ids=["scale", "scale_by_schedule"],
)
def test_scaling_gradients_ahead_of_a_rule_is_refused(transform):
    """Adam is scale-invariant, so a scale placed ahead of it trains exactly as though it were absent. That
    is worse than an error, so it is one."""
    parameter = trainable(np.zeros(2, dtype=config.floatX), name="w")
    gradients = Gradients({parameter: parameter + 1.0})

    with pytest.raises(ValueError, match="has no effect on gradients"):
        transform(gradients, [parameter])


def test_scaling_is_allowed_behind_a_rule():
    np.testing.assert_allclose(
        run_after_spike(chain(adam(1e-3), scale(0.5))),
        0.5 * np.asarray(run_after_spike(chain(adam(1e-3)))),
        rtol=1e-6,
    )


def test_reduce_on_plateau_names_the_fix_when_it_cannot_see_the_loss():
    """It decides from the loss, so it has to come first; the error has to say so rather than KeyError."""
    parameter = trainable(np.zeros(2, dtype=config.floatX), name="w")
    rate = scalar_state("rate", fill_value=1e-3)
    policy = reduce_on_plateau(adam(rate), rate, patience=1)

    with pytest.raises(ValueError, match="wrap the whole chain in it"):
        policy(Gradients({parameter: parameter + 1.0}), [parameter])


def test_chain_of_nothing_is_refused():
    with pytest.raises(ValueError, match="at least one transform"):
        chain()


@pytest.mark.parametrize(
    "transform",
    [
        clip_by_global_norm(1.0),
        clip_by_value(-0.5, 0.5),
        trace(0.9),
        add_weight_decay(0.01),
        adam(1e-3),
        sgd(0.1),
    ],
    ids=["clip_norm", "clip_value", "trace", "weight_decay", "adam", "sgd"],
)
def test_a_transform_does_not_mutate_what_it_was_given(transform):
    """`to_updates` hands back the caller's own dict rather than a copy, so a transform that assigned into
    it would reach back into the stage before it and rewrite a step already decided on."""
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    clock = pytensor.shared(np.array(0), name="clock")
    incoming = Gradients({parameter: parameter + 1.0, clock: clock + 1})
    before = dict(incoming)

    transform(incoming, [parameter])

    assert list(incoming) == list(before)
    assert all(incoming[variable] is written for variable, written in before.items())


def test_a_clip_ahead_of_a_rule_differentiates_the_loss_itself():
    """Every other case here hands the chain precomputed gradients. Here the clip is the stage that meets
    the loss, so it is the one that has to differentiate it rather than assume a dict."""
    X = Input("X", shape=(None, 4))
    loss, target = supervised_loss(Linear("fc", n_in=4, n_out=1)(X), SquaredError())

    step = compile_train(loss, chain(clip_by_global_norm(1.0), adam(1e-3)))

    rng = np.random.default_rng(0)
    batch = (
        rng.normal(size=(16, 4)).astype(config.floatX),
        rng.normal(size=(16, 1)).astype(config.floatX),
    )
    losses = [float(step(*batch)) for _ in range(20)]

    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]


@pytest.mark.parametrize(
    "build_rule",
    [
        lambda: chain(sgd(1.0), add_weight_decay(0.1)),
        lambda: chain(add_weight_decay(0.1), sgd(1.0)),
    ],
    ids=["behind-the-rule", "ahead-of-the-rule"],
)
def test_weight_decay_pulls_towards_zero_from_either_side_of_the_rule(build_rule):
    """The decay term enters a gradient and a step with opposite signs, because a rule negates what it
    reads. Under a zero gradient the decay is the only thing moving the parameter, so its direction is
    unambiguous -- and getting the sign wrong in one position turns the penalty into unbounded growth."""
    parameter = trainable(np.array([2.0], dtype=config.floatX), name="w")
    zero_gradient = pytensor.shared(np.array([0.0], dtype=config.floatX), name="g")

    pytensor.function([], parameter, updates=build_rule()([zero_gradient], [parameter]))()

    np.testing.assert_allclose(parameter.get_value(), [1.8], rtol=1e-6)


def test_a_chain_with_no_rule_in_it_is_refused():
    """Relabelling gradients as steps would compile `p <- p + g`, which walks uphill on every parameter."""
    parameter = trainable(np.array([2.0, 3.0], dtype=config.floatX), name="w")
    loss = (parameter**2).sum()

    with pytest.raises(ValueError, match="uphill"):
        compile_train(loss, clip_by_value(-10.0, 10.0), parameters=[parameter])


@pytest.mark.parametrize(
    "build_rule, named",
    [
        (lambda: chain(sgd(0.1), sgd(0.1)), "sgd"),
        (lambda: chain(adam(1e-3), adam(1e-3)), "adam"),
        (lambda: chain(adam(1e-3), adamw(1e-3)), "adamw"),
    ],
    ids=["sgd", "adam", "adamw"],
)
def test_a_rule_behind_another_rule_is_refused(build_rule, named):
    """The second rule would read the first's step as a gradient and negate it, undoing the descent. The
    error has to name the rule that was misplaced -- adam and adamw share an implementation, so reporting
    the shared one would send a reader to the optimizer they did not write."""
    parameter = trainable(np.array([2.0, 3.0], dtype=config.floatX), name="w")
    loss = (parameter**2).sum()

    with pytest.raises(
        ValueError, match=f"^{named} was given the step another rule already produced"
    ):
        compile_train(loss, build_rule(), parameters=[parameter])


@pytest.mark.parametrize(
    "build_rule",
    [lambda: adam(1e-3), lambda: trace(0.9), lambda: sgd(0.1, momentum=0.9)],
    ids=["adam", "trace", "momentum-sgd"],
)
def test_one_configured_transform_keeps_one_set_of_buffers(build_rule):
    """Compiling two training functions from one configured transform is natural, and both have to drive
    the same state: separate buffers under one derived name are wrong at runtime and collide only later,
    when both are checkpointed together."""
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    gradient = pt.constant(np.ones(3, dtype=config.floatX))
    rule = build_rule()

    first = {key for key in rule([gradient], [parameter]) if key is not parameter}
    second = {key for key in rule([gradient], [parameter]) if key is not parameter}

    assert first and first == second


def test_a_hand_written_stateful_transform_owns_its_buffers_in_a_chain():
    """A chain gives each member a frame of its own. Without one, a transform that allocates state falls
    through to the chain's frame, where the next such transform takes the slot over."""
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    loss = (parameter**2).sum()

    updates = chain(
        stateful_transform(0.9, "fast/velocity"),
        stateful_transform(0.5, "slow/velocity"),
        sgd(0.1),
    )(loss, [parameter])

    assert sorted(key.name for key in updates if key.name and "velocity" in key.name) == [
        "w/fast/velocity",
        "w/slow/velocity",
    ]


def test_a_hand_written_stateful_transform_keeps_its_buffers_across_invocations():
    """Two training functions compiled from one chain must drive the same state, which is the whole reason
    the buffers are held rather than reallocated."""
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    loss = (parameter**2).sum()
    rule = chain(stateful_transform(0.9))

    first = {id(key) for key in rule(loss, [parameter]) if key.name and "velocity" in key.name}
    again = {id(key) for key in rule(loss, [parameter]) if key.name and "velocity" in key.name}

    assert first and first == again


def test_two_claims_on_one_slot_in_a_single_step_are_refused():
    """Reuse across invocations is the point of the buffers; two claims within one is two components
    allocating over each other, and only the second one's writes would survive."""
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")

    def claims_twice(loss_gradients_or_updates, parameters):
        updates = to_updates(loss_gradients_or_updates, parameters)
        for each in parameters:
            state_for(each, "shared/velocity")
            state_for(each, "shared/velocity")
        return updates

    with pytest.raises(ValueError, match="in one step"):
        chain(claims_twice, sgd(0.1))((parameter**2).sum(), [parameter])


def test_two_transforms_of_one_kind_need_distinct_namespaces():
    """Same-kind transforms allocate under the same derived name, so they collide at the serialization
    boundary until one is given a namespace -- and the error has to say that rather than blame parameters."""
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    loss = (parameter**2).sum()

    with pytest.raises(ValueError, match="give one of them a `namespace`"):
        compile_train(loss, chain(trace(0.9), trace(0.5), sgd(0.1)), parameters=[parameter])

    namespaced = chain(trace(0.9, namespace="fast"), trace(0.5, namespace="slow"), sgd(0.1))
    updates = namespaced(loss, [parameter])
    assert sorted(key.name for key in updates if key.name and "velocity" in key.name) == [
        "w/fast/velocity",
        "w/slow/velocity",
    ]


def test_two_plateau_policies_coexist_under_distinct_namespaces():
    """One policy per rate is the reason to want two, so the history each keeps has to be separable."""
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    head_rate, backbone_rate = scalar_state("head_rate", 1e-3), scalar_state("backbone_rate", 1.0)

    rule = reduce_on_plateau(
        reduce_on_plateau(adam(head_rate * backbone_rate), head_rate, namespace="head"),
        backbone_rate,
        namespace="backbone",
    )

    histories = {key.name for key in rule((parameter**2).sum(), [parameter]) if key.name}
    assert {"head/best_loss", "backbone/best_loss"} <= histories


@pytest.mark.parametrize(
    "build_rule",
    [
        lambda: sgd(0.1),
        lambda: adam(1e-3),
        lambda: adamw(1e-3),
        lambda: nadam(1e-3),
        lambda: adamax(1e-3),
        lambda: adagrad(0.01),
        lambda: rmsprop(1e-3),
        lambda: adadelta(1.0),
        lambda: rprop(1e-3),
    ],
    ids=["sgd", "adam", "adamw", "nadam", "adamax", "adagrad", "rmsprop", "adadelta", "rprop"],
)
def test_every_rule_returns_steps(build_rule):
    """A rule that returned an unplaced mapping would disarm both checks that read the space -- the one
    stopping a second rule behind it, and the one stopping a rule-less chain from compiling ascent."""
    parameter = trainable(np.zeros(3, dtype=config.floatX), name="w")
    gradient = pt.constant(np.ones(3, dtype=config.floatX))

    assert isinstance(build_rule()([gradient], [parameter]), Steps)


def test_a_plateau_policy_wrapping_a_rule_less_chain_is_refused():
    """The policy relabels what it wraps, the same way compile_train does, so it needs the same check."""
    parameter = trainable(np.array([2.0, 3.0], dtype=config.floatX), name="w")
    rate = scalar_state("rate", fill_value=1e-3)

    policy = reduce_on_plateau(chain(clip_by_value(-10.0, 10.0)), rate)

    with pytest.raises(ValueError, match="uphill"):
        policy((parameter**2).sum(), [parameter])
