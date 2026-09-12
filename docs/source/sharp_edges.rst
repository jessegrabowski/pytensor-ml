🔪 Sharp Edges 🔪
=================

pytensor-ml is a package for defining, training, and deploying deep learning models. It is
not the only deep learning framework, and shares many similarities with Torch, Keras,
Optax, Equinox, and Flax. Users coming from these packages might be surprised by some of
the design choices we have made in layer and optimizer implementations. The purpose of this
page is to highlight our choices and keep users from stepping on too many rakes.

pytensor-ml is built on top of PyTensor, a static graph library derived from Theano. As a
result, it has idiosyncrasies that users coming from other frameworks will not be used to.
This page serves to highlight these as well.

This page is not exhaustive. We suggest new users start with our
:doc:`example gallery <examples/gallery>`.

Compile with this library's ``function``
----------------------------------------

When making a model in pytensor-ml, you use PyTensor to construct a static computational
graph that can be analyzed, manipulated, and optimized. PyTensor graphs represent pure
functions with no side-effects. The interested reader is referred to the `PyTensor
documentation <https://pytensor.readthedocs.io/en/latest/>`_ for more details.

This design is awkward for deep learning, where we want to track internal state like
running statistics, random generator entropy, and optimizer step count. PyTensor's escape
hatch for this case is shared variables and updates. The good news is that these
automatically track state for you inside the training function. The bad news is that
PyTensor requires they be manually threaded into :func:`pytensor.function` when compiling a
graph into an executable program.

To help users, pytensor-ml ships with its own separate
:func:`~pytensor_ml.pytensorf.function` implementation. This wraps
:func:`pytensor.function` and automatically handles the threading of
these updates. If you use the plain PyTensor version, internal state will not advance and
your training function will be silently broken.

.. code-block:: python

    import pytensor

    from pytensor_ml.optim import adam, compile_train
    from pytensor_ml.pytensorf import function

    step = pytensor.function([X, y], loss, updates=adam(1e-3)(loss, parameters))  # threads none
    step = function([X, y], loss, updates=adam(1e-3)(loss, parameters))           # threads all three
    step = compile_train(loss, adam(1e-3), inputs=[X, y])                         # and collects the parameters

With the plain call:

- Dropout draws one mask and reuses it for the life of the run.
- Every schedule stays pinned at step zero. Under a warmup that is a rate of exactly zero,
  and the parameters never move at all.
- Batch norm's running statistics never leave their initial values. The network trains
  against batch statistics, then predicts against ``mean=0, var=1``.

Position in a chain decides what a transform means
--------------------------------------------------

A :func:`~pytensor_ml.optim.chain` has no head. Every stage reads
``updates[parameter] - parameter``, and what that difference *is* depends on where the
stage sits: ahead of a rule it is a gradient, behind one it is the step the rule chose.
The same call means two different things in the two positions.

.. code-block:: python

    chain(clip_by_global_norm(1.0), adam(1e-3))   # bounds the gradient
    chain(adam(1e-3), clip_by_global_norm(1.0))   # bounds the step

Only the first stops an exploding gradient. Adam normalizes its step to roughly the
learning rate whatever the gradient was. A clip behind it almost never fires, and the
spike still lands in the moment estimates.

Both placements of :func:`~pytensor_ml.optim.add_weight_decay` are real optimizers: ahead
of the rule it adds the penalty to the gradient, which is coupled L2, and behind it
subtracts from the step, which is AdamW. Placed behind a terminal
:func:`~pytensor_ml.optim.scale` it decays by the learning rate; placed after
that scale it decays by the full coefficient, roughly a thousand times more at a rate of
1e-3.

Clipping cannot rescue a step that has already gone non-finite
--------------------------------------------------------------

:func:`~pytensor_ml.optim.clip_by_global_norm` bounds a step's size. It is not a
guard. One infinite coordinate makes the global norm infinite. The scale factor becomes
``max_norm / inf``, which is zero, and every healthy parameter is multiplied by it. The
poisoned coordinate becomes ``inf * 0``, which is NaN. You lose the whole step and one
parameter, and nothing raises.

.. code-block:: python

    chain(clip_by_global_norm(1.0), adam(1e-3))                   # bounds a large gradient
    apply_if_finite(chain(clip_by_global_norm(1.0), adam(1e-3)))  # survives an infinite one

:func:`~pytensor_ml.optim.apply_if_finite` is the guard for a batch that overflows.
Clipping bounds a batch that is merely large.

Optimizer state is matched by name
-----------------------------------

Serializing a model here means freezing the PyTensor graph into a JSON description of its
structure, and writing the values held by its shared variables into a separate
``.safetensors`` archive. Nothing in either file records object identity, so the two halves
are rejoined on load by name: each variable's name selects the archive entry to load into
it.

Names are therefore load-bearing at the boundary and inert everywhere else. During training
the updates dict is keyed by the variable itself, so two parameters called ``W`` are two
different objects and both train correctly. The collision only appears at the first
:func:`~pytensor_ml.save_state`, possibly hours in. Two transforms of the same kind in one
chain derive their state names the same way. Give one of them a namespace:

.. code-block:: python

    chain(trace(0.9), trace(0.5), sgd(0.1))                              # collides
    chain(trace(0.9, namespace="fast"), trace(0.5, namespace="slow"), sgd(0.1))

``reduce_on_plateau`` decides per step, not per epoch
------------------------------------------------------

A compiled step is not only the forward pass. The loss, the gradients taken from it, the
moment estimates an optimizer keeps, the clock a schedule reads, and any policy that
adjusts the rate are all nodes in one graph. On compile, this entire graph is fused into
a single function. Nothing is left over to run between calls. In pytensor-ml, schedulers and
their internal states (called "clocks"), are symbolic.

Contrast this to Torch's scheduler. These are Python objects that run when called and
read whatever they are passed. They can be called at whatever cadence you wish. pytensor-ml
schedulers, on the other hand, have to be expressible as graph nodes that are evaluated
once per call. The graph has no notion of an epoch, which lives in the
:class:`~pytensor_ml.util.DataLoader` outside it, so ``patience`` counts steps.

Torch's ``ReduceLROnPlateau(patience=10)`` waits ten epochs; this waits ten steps. A
configuration transcribed across cuts the rate roughly ``steps_per_epoch`` times too
aggressively, on per-batch noise. Two defaults compound it: ``min_scale=0.0`` lets a noisy
loss cut the rate into the ground, and ``cooldown=0`` lets the counter start toward the next
cut immediately after one.

``accumulation_size`` recovers torch's cadence by deciding on the mean of a window rather
than on one batch. Set it to the number of steps in an epoch, and note that it multiplies
the wait: ``patience=2`` with a window of 4 cuts on step 12, not step 3.

An epoch is never a first-class concept here, but it is arithmetic on steps, and these
counts take a symbolic value as readily as a Python one. A count that depends on the batch
can therefore be written in terms of it, which is the reason there is no ``epoch_size``
argument to keep in sync with your loader:

.. code-block:: python

    steps_per_epoch = n_samples // X.shape[0]
    rule = reduce_on_plateau(
        adam(rate), rate, patience=10 * steps_per_epoch, accumulation_size=steps_per_epoch
    )

A count whose value is known when the graph is built is checked there. One that is not
carries its check into the graph instead, and that check is best effort: jax drops
assertions, and a rewrite that eliminates the count eliminates its check along with it. A
nonsensical count can reach a running model without complaint.

A schedule's third argument is an endpoint, not a factor
---------------------------------------------------------

Optax's ``cosine_decay_schedule(init, decay_steps, alpha)`` takes ``alpha`` as a fraction
of the initial rate. Here the third positional argument is the rate the schedule arrives
at. The optax spelling therefore builds a schedule that ramps *up*, and nothing warns,
because an endpoint above the starting rate is a legitimate warmup.

.. code-block:: python

    cosine_schedule(3e-4, 10_000, 0.1)     # ends at 0.1, ramping up
    cosine_schedule(3e-4, 10_000, 3e-5)    # ends at 3e-5, what optax's alpha=0.1 means

There is no ``warmup`` helper for the same reason: an endpoint above the start already is
one, and :func:`~pytensor_ml.optim.join_schedules` composes the phases.

Hyperparameters are keyword-only
--------------------------------

A layer takes its name first and every hyperparameter by keyword. There are no positional
hyperparameters anywhere in the library, so the torch and keras spelling raises rather than
binding a hyperparameter to ``name`` and leaving the real one at its default:

.. code-block:: python

    Dropout(0.1)              # TypeError: Dropout's `name` must be a string ... Dropout(p=0.1)
    Conv2D("conv", 1, 16, 3)  # TypeError: takes 2 positional arguments but 5 were given

    Dropout("drop", p=0.1)    # torch's nn.Dropout(0.1)
    Conv2D("conv", in_channels=1, out_channels=16, kernel_size=3)

Only the first can tell you which parameter you meant. Once the name slot holds a string
the extra arguments are anonymous, and the error is the one Python writes.

Every layer's name is optional and falls back to its class, so ``BatchNorm(n_in=4)`` is named
``BatchNorm`` and a stack of unnamed layers offers several variables of one name. The
serialization boundary numbers those apart, keying them ``BatchNorm_1_scale``,
``BatchNorm_2_scale``, ... **in the order the variables are passed**. Collecting them from the
same graph gives the same order every time, so an unnamed stack round-trips across processes and
across rebuilds -- but a checkpoint saved from one ordering will not load under another:

.. code-block:: python

    shared = collect_shared_variables(network(X))
    save_state(shared, "weights.safetensors")
    load_state(collect_shared_variables(network(X)), "weights.safetensors")   # same order, loads

Name the layers you care about and the numbering never touches them; a name that appears once is
its own key. The ordinal numbers the layer, not the variable, so two unnamed
:class:`~pytensor_ml.layers.Linear` layers give ``Linear_1_W``, ``Linear_1_b``, ``Linear_2_W``,
``Linear_2_b``. Shared state no layer built -- a step counter, an optimizer's moments -- carries no
layer to number, so a repeat there is numbered at the end of its own name instead.

The only positional argument a layer takes is its name. The two that wrap other layers take those
instead, and their name by keyword: :class:`~pytensor_ml.layers.Recurrent` is
``Recurrent(cell, name=...)`` and :class:`~pytensor_ml.layers.Bidirectional` is
``Bidirectional(forward, backward, name=...)``.

Convolution inputs are channels-last
------------------------------------

Following jax, flax, and keras, convolutional inputs are ``(batch, *spatial, channels)``.
The reason for this is that convolution is lowered to an ``im2col`` gather followed by one
matmul, which wants the reduction axis last.

The rank is checked when the layer is called, and the error names the shape it wanted. The
channel count is not, and is discovered only when the matmul compares its operands. A
torch-shaped ``(batch, channels, height, width)`` batch with its channels declared
correctly fails as ``Incompatible shared dimension for dot product: (240, 288), (27, 16)``,
numbers that appear nowhere in your code. When the trailing spatial extent happens to equal
the declared channel count, nothing fails at all and the graph convolves over the wrong
axes.

Kernels follow the inputs: they are ``(*kernel_size, in_channels, out_channels)``, not
torch's ``(out, in, *kernel_size)``. ``im2col`` flattens the patch and its channels into
the reduction axis, which leaves the output channels last.

Normalization is channels-last too
----------------------------------

Batch norm takes statistics over every axis but the last, so a channels-first image gives
per-width-position statistics written into a per-channel slot. As with convolution it
builds whenever the trailing axis equals the declared ``n_in``, and otherwise fails as
``Incompatible Elemwise input shapes [(32,), (3,)]``, which names neither the layer nor the
axis at fault.

Linear weights are the transpose of torch's
-------------------------------------------

A batch enters a matmul as ``(batch, n_in)``, so the weight consuming it must be
``(n_in, n_out)`` and the product reads left to right: data in, features out.
:class:`~pytensor_ml.layers.Linear` stores exactly that and computes ``X @ W``, where torch
stores ``(out, in)`` because ``nn.Linear`` computes ``x @ W.T``.

A ``state_dict`` copied across without transposing raises only when ``n_in != n_out``. For
a square projection, which covers every attention q/k/v/out projection and every recurrent
``W_hh``, it loads clean and computes the transposed map.

Pooling strides by its kernel; convolution strides by one
-----------------------------------------------------------

``MaxPool2D(kernel_size=3)`` steps 3 at a time. ``Conv2D(kernel_size=3)`` steps 1. Reading
the two signatures side by side this looks like a mistake, but it matches torch: pooling
tiles, convolution slides.

A reversed recurrent layer stays aligned with its input
---------------------------------------------------------

Keras and flax return a backward pass last-step-first. Porting one, you would expect to
flip it back. Here output step ``t`` always corresponds to input step ``t``, in both
directions, and flipping it yourself is what misaligns a bidirectional concatenation.

Backends do not always compute the same answer
------------------------------------------------

MLX computes in float32 on Metal. A float64 graph is demoted throughout, the declared
output dtype no longer describes the result, and ``get_value()`` on a shared variable
returns an ``mlx.core.array`` rather than a numpy one after any update touches it.

Max pooling routes a tied window's whole gradient to the first tap on every backend except
mlx, which splits it evenly. Ties are routine: any window a rectifier has clamped entirely
to zero is one. The total gradient is conserved either way, so nothing downstream notices.
Two runs of the same script under the same seed, one on a Mac and one on Linux, diverge
from the first backward pass.

Random streams differ across backends. Between numpy and jax that is expected. Between the
python and numba linkers it is not: ``bernoulli``, the draw
:class:`~pytensor_ml.layers.Dropout` uses, disagrees
while ``uniform`` and ``normal`` match. On jax the shared generator does not advance at
all, and two functions compiled from the same graph replay one stream.
