import inspect

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

import numpy as np

from pytensor import config
from pytensor.compile.sharedvalue import SharedVariable

from pytensor_ml.params import TrainableParameter
from pytensor_ml.pytensorf import RandomSeed

RandomState = RandomSeed | np.random.RandomState | np.random.Generator

InitializationScheme = Literal[
    "zeros", "ones", "xavier_uniform", "xavier_normal", "unit_uniform", "normal", "orthogonal"
]


class Initializer(ABC):
    """
    Base class for parameter initializers.

    Subclasses implement :meth:`sample`. Calling an instance assigns a freshly sampled value to a
    parameter in place, while :func:`initialize_params` calls :meth:`sample` directly and leaves the
    assignment to its caller.

    ``__props__`` names the constructor arguments that define the draw, borrowing pytensor's Op convention
    so the same encoder serves both. A subclass taking arguments must list them, or a saved network rebuilds
    it with the defaults rather than the values it was built with.

    Examples
    --------
    Subclass it to describe a draw of your own, or reach for the :func:`initializer` decorator, which builds
    the class from a sampling function. ``__props__`` is what a saved config records, so a parameter that is
    reloaded is drawn the same way:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.state import Initializer


        class ConstantInitializer(Initializer):
            __props__ = ("value",)

            def __init__(self, value=0.0):
                self.value = value

            def sample(self, shape, dtype, rng):
                return np.full(shape, self.value, dtype=dtype)


        layer = Linear("fc", n_in=64, n_out=32, weight_initializer=ConstantInitializer(0.5))
        activations = layer(Input("X", shape=(None, 64)))
    """

    __props__: tuple[str, ...] = ()

    def __call__(self, param: SharedVariable, rng: RandomState | None = None) -> SharedVariable:
        param.set_value(self._sample_like(param, rng))
        return param

    @abstractmethod
    def sample(
        self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator
    ) -> np.ndarray: ...

    def initial_value(self, shape: tuple[int, ...]) -> np.ndarray:
        """
        Draw the value a parameter of ``shape`` is born holding, at the current ``floatX``.

        Uses fresh entropy: reproducibility comes from :meth:`~pytensor_ml.model.Model.initialize`, which
        redraws every parameter from one seed and discards whatever this produced.

        Parameters
        ----------
        shape : tuple of int
            Shape of the parameter to draw.
        """
        allocator = _ALLOCATOR.get()
        drawn_by = self if allocator is None else allocator
        return drawn_by.sample(shape, config.floatX, np.random.default_rng())

    def _sample_like(self, param: SharedVariable, rng: RandomState | None = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        value = param.get_value()
        return self.sample(value.shape, str(value.dtype), rng)


_ALLOCATOR: ContextVar[Initializer | None] = ContextVar("allocator", default=None)


@contextmanager
def initial_values_from(initializer: Initializer) -> Iterator[None]:
    """
    Draw the values parameters are born holding from ``initializer`` rather than from what they declare.

    Only the birth value changes. Each parameter still declares the law it came with, so
    :meth:`~pytensor_ml.model.Model.initialize` redraws it the way it always would.

    Parameters
    ----------
    initializer : Initializer
        Draws every parameter built inside the block, whatever its layer asked for.

    Examples
    --------
    Skip the draw entirely for a graph whose weights all arrive from a checkpoint, which is what
    :func:`~pytensor_ml.pretrained.from_pretrained` does:

    .. code-block:: python

        from pytensor_ml.layers import Linear
        from pytensor_ml.state import EmptyInitializer, initial_values_from

        with initial_values_from(EmptyInitializer()):
            layer = Linear("fc", n_in=4096, n_out=4096)
    """
    token = _ALLOCATOR.set(initializer)
    try:
        yield
    finally:
        _ALLOCATOR.reset(token)


class ZeroInitializer(Initializer):
    """
    Fill every element with zero, the right draw for a bias or any additive offset.

    Examples
    --------
    Layers already draw their biases this way; pass it explicitly to zero a weight matrix that would
    otherwise be drawn at random:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.model import Model
        from pytensor_ml.state import ZeroInitializer

        X = Input("X", shape=(None, 64))
        layer = Linear("fc", n_in=64, n_out=32, weight_initializer=ZeroInitializer())

        model = Model(layer(X)).initialize(seed=0)
    """

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)


class EmptyInitializer(Initializer):
    """
    Allocate without filling, for a parameter whose value arrives from somewhere else.

    The memory comes back holding whatever was in it, so a parameter left this way is not a model. Use it
    only where every parameter is then written: both loaders fill every one of them or raise.

    Examples
    --------
    Pair it with :func:`initial_values_from` to build a graph at allocation speed, then fill it:

    .. code-block:: python

        from pytensor_ml.layers import Linear
        from pytensor_ml.state import EmptyInitializer, initial_values_from

        with initial_values_from(EmptyInitializer()):
            layer = Linear("fc", n_in=4096, n_out=4096)
    """

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return np.empty(shape, dtype=dtype)


class OneInitializer(Initializer):
    """
    Fill every element with one, the right draw for a multiplicative scale.

    Examples
    --------
    The draw a normalization scale wants, so the layer starts as the identity transform:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.model import Model
        from pytensor_ml.state import OneInitializer

        X = Input("X", shape=(None, 64))
        layer = Linear("fc", n_in=64, n_out=32, weight_initializer=OneInitializer())

        model = Model(layer(X)).initialize(seed=0)
    """

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return np.ones(shape, dtype=dtype)


class UnitUniformInitializer(Initializer):
    r"""
    Draw from :math:`\mathcal{U}(0, 1)`.

    Examples
    --------
    Draws on the unit interval, which suits a parameter that is a probability or a fraction rather than
    a weight:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.model import Model
        from pytensor_ml.state import UnitUniformInitializer

        X = Input("X", shape=(None, 64))
        layer = Linear("fc", n_in=64, n_out=32, weight_initializer=UnitUniformInitializer())

        model = Model(layer(X)).initialize(seed=0)
    """

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=shape).astype(dtype)


class NormalInitializer(Initializer):
    r"""
    Draw every element from :math:`\mathcal{N}(\mu, \sigma^2)`, independent of the parameter's shape.

    The fan-scaled initializers derive their spread from the shape; this one is told it, which is what a
    reference implementation quoting a specific standard deviation needs -- GPT-2 initializes its embeddings
    and weights from ``NormalInitializer(0.0, 0.02)`` whatever their fans work out to.

    Parameters
    ----------
    mean : float
        Center of the distribution :math:`\mu`. Default 0.0.
    std : float
        Standard deviation :math:`\sigma`. Default 0.01.

    Examples
    --------
    A fixed-width Gaussian, ignoring the parameter's shape. GPT-2 initializes this way at ``std=0.02``,
    where a fan-scaled draw would widen as layers narrow:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.model import Model
        from pytensor_ml.state import NormalInitializer

        X = Input("X", shape=(None, 64))
        layer = Linear("fc", n_in=64, n_out=32, weight_initializer=NormalInitializer(std=0.02))

        model = Model(layer(X)).initialize(seed=0)
    """

    __props__ = ("mean", "std")

    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.std, size=shape).astype(dtype)


def fans(shape: tuple[int, ...]) -> tuple[int, int]:
    r"""
    Return the number of units feeding into and out of one position of a parameter of ``shape``.

    The two trailing dimensions are the input and output features, in that order, as
    :class:`~pytensor_ml.layers.Linear` builds ``(n_in, n_out)`` for ``X @ W``,
    :class:`~pytensor_ml.layers.Embedding` builds ``(vocabulary, features)`` and
    :class:`~pytensor_ml.layers.Conv1D` builds ``(*kernel_size, in_channels, out_channels)``. This is the
    layout flax and keras use and the transpose of torch's, where the output dimension leads. Anything
    ahead of those two is a receptive field: every input reaches an output at each of its offsets, so
    both fans carry a factor of :math:`\prod \text{kernel}`.

    Only the sum of the two matters to a Xavier draw, which is why the orientation is invisible there and
    load-bearing for anything scaling by fan-in alone.

    Parameters
    ----------
    shape : tuple of int
        Shape of the parameter, with at least two dimensions.

    Returns
    -------
    fan_in : int
        Units feeding one output position.
    fan_out : int
        Output positions one input feeds.

    Examples
    --------
    Report how many units feed into and out of one position of a parameter, which is what a fan-scaled draw
    needs. A convolution kernel folds its receptive field into both counts, so the answer is not simply the
    last two dimensions:

    .. code-block:: python

        from pytensor_ml.state import fans

        dense_fans = fans((64, 32))
        kernel_fans = fans((3, 5, 8, 16))
    """
    if len(shape) < 2:
        raise ValueError(
            f"A fan-scaled initializer needs a parameter of at least two dimensions to size its draws, but "
            f"got shape {shape}. A bias or a norm scale has no fans; give it an initializer of its own -- "
            "`trainable(value, name, initializer=ZeroInitializer())`."
        )
    receptive_field = int(np.prod(shape[:-2]))
    return shape[-2] * receptive_field, shape[-1] * receptive_field


class XavierUniformInitializer(Initializer):
    r"""
    Draw from :math:`\mathcal{U}(\pm\sqrt{6 / (\text{fan\_in} + \text{fan\_out})})`.

    The bound is chosen so the variance of the activations, and of the gradients flowing back, stays roughly
    constant through depth. Also called Glorot initialization.

    References
    ----------
    .. [1] Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward
           neural networks. Proceedings of AISTATS, 249-256.

    Examples
    --------
    The default for weight matrices: bounds chosen so activations neither grow nor shrink as they pass
    through the layer. Only the sum of the fans enters, so the layout of the parameter cannot skew it:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.model import Model
        from pytensor_ml.state import XavierUniformInitializer

        X = Input("X", shape=(None, 64))
        layer = Linear("fc", n_in=64, n_out=32, weight_initializer=XavierUniformInitializer())

        model = Model(layer(X)).initialize(seed=0)
    """

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        fan_in, fan_out = fans(shape)
        scale = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-scale, scale, size=shape).astype(dtype)


class XavierNormalInitializer(Initializer):
    r"""
    Draw from :math:`\mathcal{N}(0, 2 / (\text{fan\_in} + \text{fan\_out}))`.

    The normal counterpart of :class:`XavierUniformInitializer`, targeting the same variance.

    References
    ----------
    .. [1] Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward
           neural networks. Proceedings of AISTATS, 249-256.

    Examples
    --------
    Xavier's variance drawn from a Gaussian rather than a bounded interval, so a few weights start
    larger than any uniform draw would allow:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.model import Model
        from pytensor_ml.state import XavierNormalInitializer

        X = Input("X", shape=(None, 64))
        layer = Linear("fc", n_in=64, n_out=32, weight_initializer=XavierNormalInitializer())

        model = Model(layer(X)).initialize(seed=0)
    """

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        fan_in, fan_out = fans(shape)
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return rng.normal(0, scale, size=shape).astype(dtype)


class OrthogonalInitializer(Initializer):
    r"""
    Draw a matrix whose rows or columns are orthonormal, whichever the shape admits, scaled by ``gain``.

    An orthogonal map leaves the norm of whatever it multiplies unchanged, so applying one repeatedly
    neither grows nor shrinks a vector. The draw is a QR decomposition of a Gaussian matrix, which is
    uniform over the orthogonal group once the sign ambiguity is resolved.

    Parameters
    ----------
    gain : float
        Multiplier on the orthogonal matrix, setting the norm the map scales by. Default 1.0, which
        preserves it.

    References
    ----------
    .. [1] Saxe, A. M., McClelland, J. L., and Ganguli, S. (2014). Exact solutions to the nonlinear
           dynamics of learning in deep linear neural networks. Proceedings of ICLR.

    Examples
    --------
    Draws a matrix whose rows stay orthogonal, preserving the norm of whatever passes through it. That
    is what keeps a recurrent layer's repeated multiplications from exploding or vanishing:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.model import Model
        from pytensor_ml.state import OrthogonalInitializer

        X = Input("X", shape=(None, 64))
        layer = Linear("fc", n_in=64, n_out=32, weight_initializer=OrthogonalInitializer(gain=1.0))

        model = Model(layer(X)).initialize(seed=0)
    """

    __props__ = ("gain",)

    def __init__(self, gain: float = 1.0):
        self.gain = gain

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        if len(shape) != 2:
            raise ValueError(
                f"An orthogonal draw needs a matrix, but got a parameter of shape {shape}. "
                "Orthogonality is a property of a two-dimensional map; give a parameter of any other "
                "rank an initializer of its own."
            )

        # A reduced QR gives a Q of (rows, min(rows, cols)), so a wide matrix is drawn and decomposed
        # transposed, leaving orthonormal rows where orthonormal columns will not fit.
        n_rows, n_cols = shape
        transposed = n_rows < n_cols
        q, r = np.linalg.qr(rng.normal(size=(n_cols, n_rows) if transposed else shape))
        # QR pins Q only up to the signs of R's diagonal, and the ones LAPACK happens to return are not
        # uniform over the group. Folding them into Q is what makes the draw unbiased.
        q = q * np.sign(np.diag(r))
        return (self.gain * (q.T if transposed else q)).astype(dtype)


# Every sampler is handed these two, in this order, before its own parameters.
_DRAW_ARGUMENTS = ("rng", "shape")


def _split_signature(
    sample_fn: Callable[..., np.ndarray],
) -> tuple[list[str], dict[str, object]]:
    """
    Check that a sampler takes the draw's arguments, and return its own parameters after them.

    Returns
    -------
    parameters : list of str
        Names after the draw's arguments, which become the initializer's ``__props__``.
    defaults : dict
        Default value for each parameter that has one.
    """
    declared = list(inspect.signature(sample_fn).parameters.values())
    if tuple(parameter.name for parameter in declared[: len(_DRAW_ARGUMENTS)]) != _DRAW_ARGUMENTS:
        got = ", ".join(parameter.name for parameter in declared) or "nothing"
        raise ValueError(
            f"{sample_fn.__name__} must take {' and '.join(_DRAW_ARGUMENTS)} as its first two parameters, "
            f"before any of its own; got ({got}). Every draw is handed both, so a sampler that ignores one "
            "still declares it."
        )

    parameters: list[str] = []
    defaults: dict[str, object] = {}
    for parameter in declared[len(_DRAW_ARGUMENTS) :]:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            stars = "*" if parameter.kind is inspect.Parameter.VAR_POSITIONAL else "**"
            raise ValueError(
                f"{sample_fn.__name__} takes {stars}{parameter.name}, which has no name to record. Name "
                "each parameter explicitly."
            )
        parameters.append(parameter.name)
        if parameter.default is not inspect.Parameter.empty:
            defaults[parameter.name] = parameter.default

    return parameters, defaults


def initializer(sample_fn: Callable[..., np.ndarray]) -> type[Initializer]:
    """
    Build an :class:`Initializer` class from a sampling function, with its parameters as the draw's own.

    A sampler takes ``rng`` and ``shape``, in that order, and then whatever parameters it wants, called
    whatever it likes. Return an array of ``shape``; the dtype is applied for you.

    Those parameters are constants, frozen when you instantiate the initializer and written verbatim into a
    saved config -- which is the whole point, since a closure over them could not be written to a file. So
    anything that varies with the parameter being drawn belongs in the body, derived from ``shape``. A
    fan-scaled draw computes ``fans(shape)`` rather than taking ``fan_in``, which would otherwise be frozen
    at one layer's value and silently used for every other shape.

    Defining the function at module level is what lets a saved network find it again, because the config
    records the class by import path.

    Parameters
    ----------
    sample_fn : callable
        Draws one parameter's value, as ``(rng, shape, **parameters)``.

    Returns
    -------
    initializer_class : type of Initializer
        A class to instantiate with the parameters, e.g. ``scaled_normal(std=0.02)``.

    Examples
    --------
    Decorate a sampling function to turn it into an initializer class a layer can take:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import Linear
        from pytensor_ml.state import fans, initializer


        @initializer
        def he_normal(rng, shape):
            fan_in, _ = fans(shape)
            return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)


        layer = Linear("fc", n_in=8, n_out=8, weight_initializer=he_normal())
    """
    parameters, defaults = _split_signature(sample_fn)

    class FunctionInitializer(Initializer):
        __props__ = tuple(parameters)

        def __init__(self, **given: object):
            unexpected = set(given) - set(parameters)
            if unexpected:
                takes = f"takes {parameters}" if parameters else "takes no parameters"
                raise TypeError(
                    f"{sample_fn.__name__}() got unexpected parameters {sorted(unexpected)}; it {takes}."
                )
            bound = defaults | given
            missing = [name for name in parameters if name not in bound]
            if missing:
                raise TypeError(f"{sample_fn.__name__}() is missing parameters {missing}.")
            for name, value in bound.items():
                setattr(self, name, value)

        def sample(
            self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator
        ) -> np.ndarray:
            drawn = np.asarray(
                sample_fn(rng, shape, **{name: getattr(self, name) for name in parameters}),
                dtype=dtype,
            )
            # Not broadcast: a draw that forgot `size=shape` returns one number, and filling a parameter
            # with it would give every unit the same weight, silently, with nothing left to break the
            # symmetry.
            if drawn.shape != tuple(shape):
                raise ValueError(
                    f"{sample_fn.__name__} returned shape {drawn.shape} for a parameter of shape "
                    f"{tuple(shape)}. Pass the shape on, as in `rng.normal(0.0, 1.0, size=shape)`."
                )
            return drawn

    FunctionInitializer.__name__ = sample_fn.__name__
    FunctionInitializer.__qualname__ = sample_fn.__qualname__
    FunctionInitializer.__module__ = sample_fn.__module__
    FunctionInitializer.__doc__ = sample_fn.__doc__
    return FunctionInitializer


class UnrecordedInitializer(Initializer):
    """
    Stands in for an initializer a saved config could not record, so that a redraw says what was lost.

    Loading raises nothing: restoring values with :func:`~pytensor_ml.checkpoint.load_state` is the usual
    reason to rebuild a network, and it needs no initializer at all. Only a redraw needs the law back, and
    only then is there something to report.

    Parameters
    ----------
    original : str
        Name of the initializer class the parameter was built with.

    Examples
    --------
    What a parameter carries when its config named an initializer this build cannot resolve. It refuses to
    draw, so a reloaded network reports the missing initializer rather than silently substituting one:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.params import trainable
        from pytensor_ml.state import UnrecordedInitializer

        parameter = trainable(
            np.zeros((4, 4)), "W", initializer=UnrecordedInitializer("my_package.custom_draw")
        )
    """

    __props__ = ("original",)

    def __init__(self, original: str):
        self.original = original

    def sample(self, shape: tuple[int, ...], dtype: str, rng: np.random.Generator) -> np.ndarray:
        raise ValueError(
            f"This parameter was drawn from {self.original}, which the saved config could not record: its "
            "parameters were not JSON, or it was defined where an import cannot reach it. Redrawing needs "
            "it back. Build it with `@initializer` at module level, giving it JSON parameters, or name this "
            "parameter in `initialize(initializers={parameter: ...})`."
        )


# The name of each initializer, for the config a network is serialized to: an initializer crosses that
# boundary as a name, since a class cannot. Every entry must construct with no arguments to come back.
_INITIALIZERS: dict[str, type[Initializer]] = {
    "zeros": ZeroInitializer,
    "ones": OneInitializer,
    "xavier_uniform": XavierUniformInitializer,
    "xavier_normal": XavierNormalInitializer,
    "unit_uniform": UnitUniformInitializer,
    "normal": NormalInitializer,
    "orthogonal": OrthogonalInitializer,
}


def _initializer_for(
    param: SharedVariable, initializers: Mapping[SharedVariable, Initializer]
) -> Initializer:
    """The initializer to draw ``param`` with: one named for it, else the one it declares."""
    if param in initializers:
        return initializers[param]

    declared = param.initializer if isinstance(param, TrainableParameter) else None
    if declared is None:
        raise ValueError(
            f"{param.name!r} declares no initializer, so there is no law to redraw it from. Every layer in "
            "this library declares one for each parameter it builds; a parameter built by hand needs the "
            "same -- `trainable(value, name, initializer=XavierNormalInitializer())` -- or an entry naming "
            "it in `initializers={parameter: XavierNormalInitializer()}`."
        )
    return declared


def initialize_params(
    params: Sequence[SharedVariable],
    rng: RandomState | None = None,
    initializers: Mapping[SharedVariable, Initializer] | None = None,
) -> list[np.ndarray]:
    """
    Redraw each parameter from its own initializer, or from the one ``initializers`` names for it.

    Every parameter carries the law its value comes from, so there is nothing to decide here beyond the
    seed: a batch norm layer redraws to its unit scale, a bias to zero, and a weight matrix to a fresh
    Xavier draw. Pass ``initializers`` to draw a particular parameter some other way.

    Parameters
    ----------
    params : sequence of SharedVariable
        Parameters to draw values for. Each must declare an initializer or be named in ``initializers``.
    rng : int or numpy Generator, optional
        Random number generator. A fresh one is created when omitted.
    initializers : dict mapping parameter to Initializer, optional
        How to draw specific parameters, taking precedence over what they declare. Keyed by the parameter
        itself rather than its name, so nothing here depends on how a layer names what it builds.

    Returns
    -------
    values : list of ndarray
        Drawn values, matching the shapes and dtypes of ``params``.

    Examples
    --------
    Draw fresh values for a list of parameters without touching them, which is what
    :meth:`~pytensor_ml.model.Model.initialize` does before assigning. One seed reproduces the whole set:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.pytensorf import collect_trainable_params
        from pytensor_ml.state import initialize_params

        X = Input("X", shape=(None, 64))
        activations = Linear("fc", n_in=64, n_out=32)(X)

        parameters = collect_trainable_params(activations)
        values = initialize_params(parameters, rng=0)
    """
    # Resolve once and share: a seed handed to each _sample_like call would repeat draws across parameters.
    rng = np.random.default_rng(rng)
    if initializers is None:
        initializers = {}

    return [_initializer_for(param, initializers)._sample_like(param, rng) for param in params]
