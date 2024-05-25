"""
PyTorch's philosophy:

These algorithms are
`manually` validated by comparing to the paper and
`systematically` validated by assuring that the loss
goes the right direction when the optimizer has been applied.

So it means that they just eyeball the implementation and
only test if the optimizer indeed makes the loss go down.

For faster implementation such as foreach or fused, they
numerically compare the value with the original for-loop
implementation.

We can reorganize structure later.
"""

from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pytensor_ml.layers import Linear, Sequential
from pytensor_ml.activations import LeakyReLU
from pytensor_ml.loss import CrossEntropy
from pytensor_ml.model import Model
from pytensor_ml.optimizers import SGD, ADAGrad, Adam
import numpy as np
import pytensor


def generate_example_data():
    X, y = load_digits(return_X_y=True)
    y_onehot = OneHotEncoder().fit_transform(y[:, None]).toarray()
    X_normed = MinMaxScaler().fit_transform(X)
    return X_normed, y_onehot


def create_simple_model():
    X_in = pytensor.tensor.tensor("X_in", shape=(None, 64))
    prediction_network = Sequential(
        Linear("Linear_1", n_in=64, n_out=256),
        LeakyReLU(),
        Linear("Linear_2", n_in=256, n_out=128),
        LeakyReLU(),
        Linear("Logits", n_in=128, n_out=10),
    )

    y_hat = prediction_network(X_in)
    model = Model(X_in, y_hat)
    return model


X_normed, y_onehot = generate_example_data()
model = create_simple_model()
loss_fn = CrossEntropy(expect_onehot_labels=True, expect_logits=True, reduction="mean")
model.initalize_weights()


def test_adam():
    optim = Adam(model, loss_fn, ndim_out=2, learning_rate=1e-3)

    n_obs = X_normed.shape[0]
    cutpoints = np.arange(0, n_obs, 1000).tolist()
    cutpoints += [n_obs]
    batch_slices = list(zip(cutpoints[:-1], cutpoints[1:]))
    loss_history = []
    n_epochs = 10

    for _ in range(n_epochs):
        all_idx = np.arange(n_obs)
        np.random.shuffle(all_idx)
        y_epoch = y_onehot[all_idx, :]
        X_epoch = X_normed[all_idx, :]
        for start, stop in batch_slices:
            idx = slice(start, stop)
            loss = optim.step(X_epoch[idx], y_epoch[idx])
            loss_history.append(loss)

    assert loss_history[0] > loss_history[-1]
