# pytensor_ml

A(nother) deep learning library, built on top of [PyTensor](https://github.com/pymc-devs/pytensor).

Networks are ordinary PyTensor graphs. You build one out of layers, and everything PyTensor already does —
symbolic differentiation, graph rewrites, and compilation to Numba, C, JAX, PyTorch, or MLX — applies to it
unchanged. Training is a compiled function that takes a batch and returns a loss; there is no separate runtime
or tape.

That goes all the way down: layers are graph constructors, parameters are shared variables, and a training
step is a compiled function whose updates are the optimizer. Because a model is only a graph, it composes with
any other PyTensor graph — a PyMC model included — as there is nothing else to interoperate with.

> **Status: pre-alpha.** The API is still moving, and there is no release-to-release compatibility guarantee yet.

## Installation

```bash
pip install pytensor-ml
```

The only hard dependencies are `pytensor`, `numpy`, and `safetensors`. A backend beyond the default (`numba`,
`jax`, `torch`, `mlx`) is installed separately, and only loads when you actually compile against it.

## Quickstart

Train a classifier on scikit-learn's digits, then run inference:

```python
import numpy as np
import pytensor

pytensor.config.floatX = "float32"

from sklearn.datasets import load_digits

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import Input, Linear, Sequential
from pytensor_ml.loss import CrossEntropy
from pytensor_ml.model import Model
from pytensor_ml.optim import adam, chain, clip_by_global_norm, cosine_schedule
from pytensor_ml.util import DataLoader

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype("float32")
y_onehot = np.eye(10, dtype="float32")[y]

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

accuracy = (model.predict(X).argmax(axis=-1) == y).mean()
```

`compile_train` builds the loss against a target placeholder, differentiates it, folds in any stateful layer
updates (batch norm running statistics, RNG advances, the training clock a schedule reads), and compiles a
one-step function. `predict` compiles a separate inference pass, with dropout removed and batch norm reading
its running statistics.

## Documentation

The full API reference and user guide live at
[pytensor-ml.readthedocs.io](https://pytensor-ml.readthedocs.io).

For worked models end to end — training loops, convolutional and recurrent networks, transformers, saving and
reloading — see the [examples gallery](https://pytensor-ml.readthedocs.io/en/latest/examples/gallery.html).

## Contributing

Contributions are welcome. To get set up:

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

Formatting and linting run through `ruff` under pre-commit, and `mypy` checks `pytensor_ml/`; both also run in
CI. Bug reports and feature requests belong in the
[issue tracker](https://github.com/pymc-devs/pytensor-ml/issues).

## License

Apache 2.0. See [LICENSE](LICENSE).
