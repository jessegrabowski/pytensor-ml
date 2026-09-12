pytensor_ml
===========

A(nother) deep learning library, built on top of PyTensor.

Networks are ordinary PyTensor graphs. You build one out of layers, and
everything PyTensor already does — symbolic differentiation, graph rewrites,
and compilation to Numba, C, JAX, PyTorch, or MLX — applies to it unchanged.
Training is a compiled function that takes a batch and returns a loss; there
is no separate runtime or tape.

That goes all the way down: layers are graph constructors, parameters are
shared variables, and a training step is a compiled function whose updates are
the optimizer. Because a model is only a graph, it composes with any other
PyTensor graph — a PyMC model included — as there is nothing else to
interoperate with.

pytensor_ml ships the usual layer library (dense, convolutional, recurrent,
attention, normalization), composable optimizers with learning-rate schedules
and step guards, and safetensors-backed serialization that round-trips both
weights and architecture.

.. note::

   pytensor_ml is pre-alpha. The API is still moving, and there is no
   release-to-release compatibility guarantee yet.

Quick install
-------------

.. code-block:: bash

    pip install pytensor-ml

See the :doc:`installation guide <install>` for backend extras.

Quick example
-------------

.. code-block:: python

    import numpy as np

    from pytensor import config
    from sklearn.datasets import load_digits

    from pytensor_ml.activations import ReLU
    from pytensor_ml.layers import Input, Linear, Sequential
    from pytensor_ml.loss import CrossEntropy
    from pytensor_ml.model import Model
    from pytensor_ml.optim import adam, chain, clip_by_global_norm, cosine_schedule
    from pytensor_ml.util import DataLoader

    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(config.floatX)
    y_onehot = np.eye(10, dtype=config.floatX)[y]

    X_in = Input("X_in", shape=(None, 64))
    network = Sequential(
        Linear("fc1", n_in=64, n_out=128),
        ReLU(),
        Linear("logits", n_in=128, n_out=10),
    )
    model = Model(network(X_in)).initialize(seed=0)

    rule = chain(clip_by_global_norm(1.0), adam(learning_rate=cosine_schedule(1e-3, total_steps=500)))
    loss_fn = CrossEntropy(expect_onehot_labels=True, expect_logits=True, reduction="mean")
    step = model.compile_train(rule, loss_fn)

    loader = DataLoader(X, y_onehot, batch_size=64, random_state=0)
    for _ in range(500):
        loss_value = step(*loader())

    predictions = model.predict(X).argmax(axis=-1)

See the :doc:`example gallery <examples/gallery>` for full end-to-end
walkthroughs.

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   install
   sharp_edges
   examples/gallery
   api
   dev/index
