Optimization
============

.. currentmodule:: pytensor_ml.optim

Training
--------

.. autosummary::
    :toctree: generated/

    compile_train

Update rules
------------

.. autosummary::
    :toctree: generated/

    sgd
    adam
    adamw
    nadam
    adamax
    rmsprop
    rprop
    adagrad
    adadelta

Transforms
----------

.. autosummary::
    :toctree: generated/

    chain
    scale
    scale_by_schedule
    add_weight_decay
    trace
    clip_by_global_norm
    clip_by_value

Guards and policies
-------------------

.. autosummary::
    :toctree: generated/

    skip_if
    apply_if_finite
    nonfinite
    large_step
    reduce_on_plateau
    SkipCondition

Schedules
---------

Every step count a schedule takes -- ``total_steps``, ``transition_begin``, ``decay_every``, and the
``boundaries`` of :func:`join_schedules` -- accepts a symbolic value as readily as a literal one, so a
horizon can be written in terms of the data it will run over. With ``X`` the minibatch and
``n_samples`` the size of the dataset, ``total_steps=10 * (n_samples // X.shape[0])`` is ten epochs.
The same holds for the ``patience``, ``cooldown``, and ``accumulation_size`` of
:func:`reduce_on_plateau` and the ``max_norm`` of :func:`large_step` and
:func:`clip_by_global_norm`, and the bounds of :func:`clip_by_value`. A count whose value is not known
when the graph is built carries its check into the graph instead. That check rides on the value, so it
runs wherever the value does: not under jax, which drops assertions, and not where a rewrite has
eliminated the value's last consumer.

.. autosummary::
    :toctree: generated/

    constant_schedule
    linear_schedule
    linear_onecycle_schedule
    cosine_schedule
    exponential_schedule
    polynomial_schedule
    step_decay
    join_schedules

Building blocks
---------------

.. autosummary::
    :toctree: generated/

    counter
    get_gradients
    reuses_state
    scalar_state
    state_for
    steps_of
    to_floatx
    to_updates

.. currentmodule:: pytensor_ml.optim.base

.. autosummary::
    :toctree: generated/

    Transform
    Updates
    Gradients
    Steps
    Schedule
    Rate
    LearningRate

.. currentmodule:: pytensor_ml.optim.guards

.. autosummary::
    :toctree: generated/

    Decision

.. currentmodule:: pytensor_ml.optim

Low-level update functions
--------------------------

.. autosummary::
    :toctree: generated/

    sgd_updates
    adam_updates
    adamw_updates
    nadam_updates
    adamax_updates
    rmsprop_updates
    rprop_updates
    adagrad_updates
    adadelta_updates
