import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor import config

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import Linear, Sequential
from pytensor_ml.pytensorf import collect_trainable_params, function
from pytensor_ml.state import initialize_params


@pytest.fixture
def run_training():
    """
    Return a helper that trains a fresh 2-layer network on a fixed regression target with the given rule.

    Each call builds a new network (new shared variables), so optimizer state never leaks between tests. The
    helper returns the per-step loss history.
    """

    def _run(rule, n_steps: int = 50, seed: int = 0) -> list[float]:
        X = pt.tensor("X", shape=(None, 8))
        network = Sequential(
            Linear("fc1", n_in=8, n_out=16), ReLU(), Linear("fc2", n_in=16, n_out=4)
        )
        y = network(X)

        params = collect_trainable_params(y)
        rng = np.random.default_rng(seed)
        for param, value in zip(params, initialize_params(params, rng=rng)):
            param.set_value(value)

        target = pt.matrix("target")
        loss = ((y - target) ** 2).mean()

        fn = function([X, target], loss, updates=rule(loss, params))
        X_val = rng.normal(size=(64, 8)).astype(config.floatX)
        target_val = rng.normal(size=(64, 4)).astype(config.floatX)
        return [float(fn(X_val, target_val)) for _ in range(n_steps)]

    return _run
