import numpy as np
import pytensor.tensor as pt

from pytensor import config

from pytensor_ml.base import Layer


def _constant_like(value: float, x: pt.TensorVariable) -> pt.TensorVariable:
    """
    Wrap a scalar so that combining it with ``x`` cannot widen ``x``'s dtype.

    PyTensor's autocaster types a bare Python float by value, so whether a literal widens its operand
    depends on that value: against a float32 input ``0.5 * x`` stays float32, while ``0.01 * x``
    promotes to float64. Pinning the constant to ``x``'s dtype removes the dependence.
    """
    dtype = np.dtype(x.dtype)
    # np.finfo maps complex64 -> float32, keeping complex inputs at their own precision.
    dtype = np.finfo(dtype).dtype if np.issubdtype(dtype, np.inexact) else np.dtype(config.floatX)
    return pt.constant(np.asarray(value, dtype=dtype))


class Activation(Layer):
    """
    Base class for the elementwise nonlinearities, each of which is called on an activation.

    Examples
    --------
    Subclass it to add a nonlinearity of your own; the body builds a graph from its input:

    .. code-block:: python

        import pytensor.tensor as pt

        from pytensor_ml.activations import Activation
        from pytensor_ml.layers import Input


        class HardTanh(Activation):
            def __call__(self, X):
                return pt.clip(X, -1.0, 1.0)


        activations = HardTanh()(Input("X", shape=(None, 4)))
    """


class ReLU(Activation):
    r"""
    Rectified Linear Unit.

    Compute the positive part of the input:

    .. math::

        \mathrm{ReLU}(x) = \max(0, x).

    Examples
    --------
    Drop it into a stack wherever a nonlinearity belongs, usually straight after a linear layer:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import Input, Linear, Sequential

        X = Input("X", shape=(None, 4))
        network = Sequential(
            Linear("fc", n_in=4, n_out=8),
            ReLU(),
        )
        activations = network(X)
    """

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.maximum(0, x)
        out.name = "ReLU"
        return out


class LeakyReLU(Activation):
    r"""
    Leaky Rectified Linear Unit.

    Replace ReLU's flat negative branch with a small negative slope, so negative inputs keep a
    nonzero gradient:

    .. math::

        \mathrm{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \le 0 \end{cases}

    Parameters
    ----------
    negative_slope : float, optional
        The slope :math:`\alpha` applied to negative inputs. Default is 0.01.

    Examples
    --------
    The slope below zero is what separates it from :class:`ReLU`, and a wider one keeps more gradient
    flowing through units that would otherwise be dead:

    .. code-block:: python

        from pytensor_ml.activations import LeakyReLU
        from pytensor_ml.layers import Input, Linear, Sequential

        X = Input("X", shape=(None, 4))
        network = Sequential(
            Linear("fc", n_in=4, n_out=8),
            LeakyReLU(negative_slope=0.2),
        )
        activations = network(X)
    """

    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        x = pt.as_tensor(x)
        out = pt.switch(x > 0, x, _constant_like(self.negative_slope, x) * x)
        out.name = "LeakyReLU"
        return out


class Tanh(Activation):
    r"""
    Hyperbolic tangent.

    Squash the input to :math:`(-1, 1)`:

    .. math::

        \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}.

    Examples
    --------
    Squash an activation into ``(-1, 1)``, keeping it centred on zero:

    .. code-block:: python

        from pytensor_ml.activations import Tanh
        from pytensor_ml.layers import Input, Linear, Sequential

        X = Input("X", shape=(None, 4))
        network = Sequential(
            Linear("fc", n_in=4, n_out=8),
            Tanh(),
        )
        activations = network(X)
    """

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.tanh(x)
        out.name = "Tanh"
        return out


class Sigmoid(Activation):
    r"""
    Logistic sigmoid.

    Squash the input to :math:`(0, 1)`:

    .. math::

        \sigma(x) = \frac{1}{1 + e^{-x}}.

    Examples
    --------
    Squash an activation into ``(0, 1)``, which is what a binary output head wants:

    .. code-block:: python

        from pytensor_ml.activations import Sigmoid
        from pytensor_ml.layers import Input, Linear, Sequential

        X = Input("X", shape=(None, 4))
        network = Sequential(
            Linear("fc", n_in=4, n_out=8),
            Sigmoid(),
        )
        activations = network(X)
    """

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.sigmoid(x)
        out.name = "Sigmoid"
        return out


class SoftPlus(Activation):
    r"""
    Softplus activation.

    Compute a smooth approximation to ReLU:

    .. math::

        \mathrm{softplus}(x) = \log(1 + e^x).

    Examples
    --------
    Reach for it where an output has to stay strictly positive, such as a predicted scale:

    .. code-block:: python

        from pytensor_ml.activations import SoftPlus
        from pytensor_ml.layers import Input, Linear, Sequential

        X = Input("X", shape=(None, 4))
        network = Sequential(
            Linear("fc", n_in=4, n_out=8),
            SoftPlus(),
        )
        activations = network(X)
    """

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.softplus(x)
        out.name = "SoftPlus"
        return out


class GELU(Activation):
    r"""
    Gaussian Error Linear Unit.

    Compute :math:`\mathrm{GELU}(x) = x \, \Phi(x)`, where :math:`\Phi` is the standard normal
    cumulative distribution function:

    .. math::

        \mathrm{GELU}(x) = \frac{x}{2} \left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right).

    Parameters
    ----------
    approximate : bool, optional
        Use the tanh approximation

        .. math::

            \mathrm{GELU}(x) \approx \frac{x}{2}
            \left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right]\right)

        This is the variant HuggingFace calls ``"gelu_new"`` / ``"gelu_pytorch_tanh"``, PyTorch exposes
        as ``nn.GELU(approximate="tanh")``, and Flax as ``gelu(approximate=True)``; GPT-2 uses it. It is
        cheaper to evaluate than the exact :math:`\operatorname{erf}` form. Default is True.

    Examples
    --------
    The default takes the tanh approximation. Pass ``approximate=False`` for the exact error-function
    form, which costs more to evaluate:

    .. code-block:: python

        from pytensor_ml.activations import GELU
        from pytensor_ml.layers import Input, Linear, Sequential

        X = Input("X", shape=(None, 4))
        network = Sequential(
            Linear("fc", n_in=4, n_out=8),
            GELU(approximate=False),
        )
        activations = network(X)
    """

    def __init__(self, approximate: bool = True):
        self.approximate = approximate

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        x = pt.as_tensor(x)
        half = _constant_like(0.5, x)
        one = _constant_like(1.0, x)
        if self.approximate:
            sqrt_2_over_pi = _constant_like(np.sqrt(2.0 / np.pi), x)
            cubic_coef = _constant_like(0.044715, x)
            out = half * x * (one + pt.tanh(sqrt_2_over_pi * (x + cubic_coef * x**3)))
        else:
            sqrt2 = _constant_like(np.sqrt(2.0), x)
            out = half * x * (one + pt.erf(x / sqrt2))
        out.name = "GELU"
        return out


class Swish(Activation):
    r"""
    Swish activation, also known as SiLU.

    Compute :math:`\mathrm{Swish}(x) = x \, \sigma(\beta x)`, where :math:`\sigma` is the logistic
    sigmoid. With :math:`\beta = 1` this is the Sigmoid Linear Unit (PyTorch ``nn.SiLU``, HuggingFace
    ``"silu"`` / ``"swish"``).

    Parameters
    ----------
    beta : float, optional
        Slope of the sigmoid gate. Larger :math:`\beta` sharpens the gate toward a ReLU; :math:`\beta
        \to 0` collapses it toward the linear map :math:`x/2`. Default is 1.0.

    Examples
    --------
    Raise ``beta`` to sharpen the gate towards ReLU's hinge, or lower it to soften towards a linear unit:

    .. code-block:: python

        from pytensor_ml.activations import Swish
        from pytensor_ml.layers import Input, Linear, Sequential

        X = Input("X", shape=(None, 4))
        network = Sequential(
            Linear("fc", n_in=4, n_out=8),
            Swish(beta=1.5),
        )
        activations = network(X)
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        x = pt.as_tensor(x)
        out = x * pt.sigmoid(_constant_like(self.beta, x) * x)
        out.name = type(self).__name__
        return out


class QuickGELU(Swish):
    r"""
    Sigmoid approximation to GELU.

    Compute :math:`\mathrm{QuickGELU}(x) = x \, \sigma(1.702 x)`, which is exactly :class:`Swish`
    at :math:`\beta = 1.702`. Every OpenAI CLIP checkpoint was trained with it, and loading those
    weights under :class:`GELU` gives quietly wrong activations rather than an error. Prefer
    :class:`GELU` for anything new.

    Examples
    --------
    Use it where a checkpoint was trained with it, such as a CLIP text encoder's MLP:

    .. code-block:: python

        from pytensor_ml.activations import QuickGELU
        from pytensor_ml.layers import FeedForward, Input

        X = Input("X", shape=(None, 77, 768))
        activations = FeedForward("mlp", d_model=768, activation=QuickGELU())(X)
    """

    def __init__(self):
        super().__init__(beta=1.702)


class Softmax(Activation):
    r"""
    Softmax activation.

    Normalize the input along one axis into a probability distribution:

    .. math::

        \mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}.

    Parameters
    ----------
    axis : int, optional
        The axis along which the values sum to one. Default is -1.

    Examples
    --------
    Normalizes over the last axis by default, which is the class axis of a ``(batch, classes)`` logit
    matrix. A loss built with ``expect_logits=True`` wants the logits themselves, so reach for this only
    when you need the probabilities:

    .. code-block:: python

        from pytensor_ml.activations import Softmax
        from pytensor_ml.layers import Input, Linear, Sequential

        X = Input("X", shape=(None, 4))
        network = Sequential(
            Linear("logits", n_in=4, n_out=3),
            Softmax(axis=-1),
        )
        probabilities = network(X)
    """

    def __init__(self, axis: int = -1):
        self.axis = axis

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        out = pt.special.softmax(x, axis=self.axis)
        out.name = "Softmax"
        return out


__all__ = [
    "GELU",
    "Activation",
    "LeakyReLU",
    "ReLU",
    "Sigmoid",
    "SoftPlus",
    "Softmax",
    "Swish",
    "Tanh",
]
