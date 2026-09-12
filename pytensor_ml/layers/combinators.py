from collections.abc import Callable, Sequence

import pytensor.tensor as pt


def Input(name: str, shape: tuple[int | None, ...], dtype: str | None = None) -> pt.TensorVariable:
    """
    Create a named symbolic input tensor.

    Parameters
    ----------
    name : str
        Name of the input variable.
    shape : tuple of int or None
        Size of each dimension. Use None wherever the size varies between calls, such as a batch axis.
    dtype : str, optional
        Data type of the input. Default ``floatX``.

    Examples
    --------
    Start every network with one: it names the placeholder a batch is fed to, with ``None`` wherever the
    size varies from call to call:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear

        X = Input("X", shape=(None, 64))
        activations = Linear("fc", n_in=64, n_out=10)(X)
    """
    return pt.tensor(name=name, shape=shape, dtype=dtype)


def Sequential(*layers: Callable) -> Callable:
    """
    Compose layers left to right into a single callable that threads its input through each in turn.

    Examples
    --------
    Thread an input through several layers in order. The result is itself callable, so it nests inside
    another ``Sequential`` wherever a block repeats:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import BatchNorm, Input, Linear, Sequential

        block = Sequential(
            Linear("fc1", n_in=64, n_out=32),
            BatchNorm("bn1", n_in=32),
            ReLU(),
        )
        network = Sequential(
            block,
            Linear("logits", n_in=32, n_out=10),
        )

        logits = network(Input("X", shape=(None, 64)))
    """

    def forward(x: pt.TensorLike) -> pt.TensorLike:
        for layer in layers:
            x = layer(x)
        return x

    return forward


def Flatten(X: pt.TensorLike) -> pt.TensorVariable:
    """
    Collapse everything after the batch axis into one, so a convolution stack can reach a dense head.

    Parameters
    ----------
    X : TensorLike
        An activation of any rank, whose first axis is the batch.

    Returns
    -------
    flattened : TensorVariable
        Shape ``(batch, features)``, with ``features`` the product of every remaining axis.

    Examples
    --------
    Collapse everything after the batch axis, which is how a convolutional stack hands off to a dense head:

    .. code-block:: python

        from pytensor_ml.layers import Conv2D, Flatten, Input, Linear, MaxPool2D, Sequential

        X = Input("X", shape=(None, 28, 28, 1))
        network = Sequential(
            Conv2D("conv", in_channels=1, out_channels=8, kernel_size=3),
            MaxPool2D(),
        )
        features = network(X)

        logits = Linear("logits", n_in=8 * 13 * 13, n_out=10)(Flatten(features))
    """
    return pt.join_dims(X, start_axis=1)


def Squeeze(X: pt.TensorLike, axis: int | Sequence[int] | None = None) -> pt.TensorVariable:
    """
    Drop length-1 axes, so a layer that emits a singleton axis feeds one that does not expect it.

    Parameters
    ----------
    X : TensorLike
        Tensor to squeeze.
    axis : int or sequence of int, optional
        Axes to drop. An axis whose length is statically known to be anything but 1 is rejected as the
        graph is built; an axis of unknown length is accepted and checked when the function runs.
        Default None, which drops every axis already known to have length 1 and leaves the rest.

    Returns
    -------
    squeezed : TensorVariable
        ``X`` with the selected axes removed.

    Examples
    --------
    Drop a length-1 axis a layer left behind, such as the trailing feature axis of a single-output
    regression head:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear, Squeeze

        X = Input("X", shape=(None, 64))
        prediction = Linear("fc", n_in=64, n_out=1)(X)

        per_row = Squeeze(prediction, axis=-1)
    """
    return pt.squeeze(X, axis=axis)


def Concatenate(tensors: Sequence[pt.TensorLike], axis: int = 0) -> pt.TensorVariable:
    """
    Join tensors end to end along one axis.

    Parameters
    ----------
    tensors : sequence of TensorLike
        Tensors to join. Every one must agree in rank, and in size on every axis but ``axis``.
    axis : int, optional
        Axis to join along; negative values count from the right. Default 0, which for a batched
        activation is the batch axis -- merging two ``(batch, features)`` branches feature-wise wants
        ``axis=-1``.

    Returns
    -------
    joined : TensorVariable
        The inputs joined, with extent along ``axis`` equal to the sum of the inputs' extents.

    Examples
    --------
    Merge parallel branches back into one tensor. Pass ``axis=-1`` to join along features, since the
    default of 0 joins along the batch:

    .. code-block:: python

        from pytensor_ml.layers import Concatenate, Input, Linear

        X = Input("X", shape=(None, 64))
        wide = Linear("wide", n_in=64, n_out=8)(X)
        deep = Linear("deep", n_in=64, n_out=4)(X)

        merged = Concatenate([wide, deep], axis=-1)
    """
    return pt.concatenate(tensors, axis=axis)
