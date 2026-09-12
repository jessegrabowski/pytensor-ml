import pytensor.tensor as pt

from pytensor_ml.base import Layer, UnaryLayerOp, _resolve_layer_name
from pytensor_ml.params import trainable
from pytensor_ml.state import Initializer, XavierNormalInitializer, ZeroInitializer


def shape_to_str(shape):
    inner = ",".join([str(st_dim) if st_dim is not None else "?" for st_dim in shape])
    return f"({inner})"


class LinearLayer(UnaryLayerOp):
    __props__ = ("n_in", "n_out", "bias")

    def build_inner_graph(self, X, W, b=None):
        res = X @ W
        if self.bias:
            res = res + b
        return [res]


class Linear(Layer):
    r"""
    Affine map :math:`y = x W + b`.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "Linear" when None.
    n_in : int
        Size of the input feature axis.
    n_out : int
        Size of the output feature axis.
    bias : bool, optional
        Add the learned shift :math:`b`, which starts at zero and is redrawn there. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W` is drawn, at construction and on every redraw. Xavier normal when omitted.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted, following Keras and flax rather than torch, which draws
        the bias from :math:`\mathcal{U}(\pm 1/\sqrt{\text{fan\_in}})`.

    Notes
    -----
    Both parameters are drawn when the layer is built, so a network trains without any further call.
    :meth:`~pytensor_ml.model.Model.initialize` redraws them from a single seed, which is what makes a run
    reproducible.

    Examples
    --------
    The dense layer: multiply by a learned matrix and add a learned bias. Pass ``bias=False`` where a
    normalization layer follows and would subtract the bias away again:

    .. code-block:: python

        from pytensor_ml.layers import BatchNorm, Input, Linear, Sequential

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32, bias=False),
            BatchNorm("bn", n_in=32),
        )

        activations = network(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_in: int,
        n_out: int,
        bias: bool = True,
        weight_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "n_in")
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias

        # Drawn here rather than left at zero: a weight and its value are one thing, and a stack of zero
        # weights passes no gradient below its last layer. The initializer is declared as well as used, so
        # `initialize(seed=...)` redraws from the same law.
        W_initializer = (
            XavierNormalInitializer() if weight_initializer is None else weight_initializer
        )
        self.W = trainable(
            W_initializer.initial_value((n_in, n_out)),
            f"{self.name}_W",
            initializer=W_initializer,
            layer_name=self.name,
        )

        if self.bias:
            b_initializer = ZeroInitializer() if bias_initializer is None else bias_initializer
            self.b = trainable(
                b_initializer.initial_value((n_out,)),
                f"{self.name}_b",
                initializer=b_initializer,
                layer_name=self.name,
            )

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)

        inputs = [X, self.W]
        if self.bias:
            inputs.append(self.b)

        input_shape = shape_to_str(X.type.shape)
        output_shape = shape_to_str((X @ self.W).type.shape)

        ofg = LinearLayer(
            name=f"{self.name}[{input_shape} -> {output_shape}]",
            n_in=self.n_in,
            n_out=self.n_out,
            bias=self.bias,
        )
        out = ofg(*inputs)
        out.name = f"{self.name}_output"

        return out
