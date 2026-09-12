from collections.abc import Sequence
from functools import reduce
from operator import add, mul

import numpy as np
import pytensor.tensor as pt

from numpy.lib.stride_tricks import sliding_window_view
from pytensor import config
from pytensor.gradient import disconnected_type
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.tensor.basic import get_scalar_constant_value
from pytensor.tensor.pad import PadMode
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import Layer, LayerOp, UnaryLayerOp, _check_input_rank, _resolve_layer_name
from pytensor_ml.layers.padding import _pad_spatial
from pytensor_ml.params import trainable_parameter
from pytensor_ml.state import Initializer, XavierNormalInitializer, ZeroInitializer


def _max_over_taps(patches: TensorVariable, axis: int) -> TensorVariable:
    """
    Take each window's largest tap, routing the whole gradient to the one tap that won it.

    Reducing with :func:`pytensor.tensor.max` would instead give the full cotangent to *every* tap tied
    for the maximum, which returns more gradient than the window received -- and a window of ties is
    routine rather than exotic, since max pooling usually follows a rectifier that clamps whole windows
    to zero. Selecting through an ``argmax`` picks one tap, and conserves the gradient the way every
    backend's own pooling does.
    """
    winner = pt.argmax(patches, axis=axis)
    pattern: list[int | str] = ["x"] * patches.ndim
    pattern[axis] = 0
    taps = pt.arange(patches.shape[axis]).dimshuffle(*pattern)
    # Selected with `switch` rather than multiplied by the mask: an unselected tap may be the -inf a
    # padded position carries, and -inf times zero is nan rather than nothing.
    chosen = pt.switch(pt.eq(pt.expand_dims(winner, axis), taps), patches, 0)
    return chosen.sum(axis=axis)


# Each reduction and the value that pads without changing it: -inf loses every comparison to a real
# element, and zero is what an average counts padded positions as, matching torch's count_include_pad.
_REDUCTIONS = {"max": _max_over_taps, "mean": pt.mean}
_PADDING_IDENTITY = {"max": -np.inf, "mean": 0.0}


def _check_reduction(reduction: str) -> None:
    """Reject a reduction neither pooling op knows, where a lookup would otherwise raise a KeyError."""
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {sorted(_REDUCTIONS)}, but got {reduction!r}.")


def _window_span(extent: int, spacing: int) -> int:
    """How far one window reaches along an axis, once dilation has spread its taps."""
    return spacing * (extent - 1) + 1


def _window_indices(
    X: TensorVariable, kernel_size: Sequence[int], stride: Sequence[int], dilation: Sequence[int]
) -> list[TensorVariable]:
    """One advanced index per spatial axis, carrying that axis's windows and its taps.

    Each broadcasts against the others, so indexing with all of them at once gives windows-then-taps in
    order. Shared by the gather's reference graph and by the scatter that reverses it.
    """
    n_spatial = len(kernel_size)
    indices = []
    for axis, (extent, step, spacing) in enumerate(zip(kernel_size, stride, dilation)):
        span = _window_span(extent, spacing)
        starts = pt.arange(0, X.shape[1 + axis] - span + 1, step)
        window = starts[:, None] + pt.arange(extent)[None, :] * spacing

        pattern: list[int | str] = ["x"] * (2 * n_spatial)
        pattern[axis] = 0
        pattern[n_spatial + axis] = 1
        indices.append(window.dimshuffle(*pattern))
    return indices


def _scatter_patches(
    cotangent: TensorVariable,
    X: TensorVariable,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> TensorVariable:
    """Add each window's cotangent back at the position it was gathered from.

    Windows overlap, so a position reached by several of them accumulates all of their contributions --
    which is why this is a scatter-add rather than an assignment.
    """
    indices = _window_indices(X, kernel_size, stride, dilation)
    zeros = pt.zeros(X.shape, dtype=cotangent.dtype)
    return pt.inc_subtensor(zeros[(slice(None), *indices, slice(None))], cotangent)


class Im2Col(Op):
    """
    Gather every window a kernel visits, as one node a backend can put a real copy behind.

    The equivalent advanced-indexing graph does scalar index arithmetic per element; the copy this
    describes is one contiguous channel-row per ``(batch, window, tap)``, which is the difference between
    a few GB/s and memory bandwidth. It sits inside :class:`ConvLayer`'s inner graph, so a backend that
    dispatches the convolution itself never reaches it.

    Parameters
    ----------
    kernel_size, stride, dilation : tuple of int
        One entry per spatial axis.
    """

    __props__ = ("kernel_size", "stride", "dilation")

    def __init__(
        self,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        dilation: Sequence[int],
    ):
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.dilation = tuple(dilation)

    def __call__(self, *inputs, **kwargs) -> TensorVariable:
        """Narrow the single output to a tensor, which ``Op.__call__`` cannot know it is."""
        out = super().__call__(*inputs, **kwargs)
        assert isinstance(out, TensorVariable), "Im2Col produces one output"
        return out

    def make_node(self, X):
        X = pt.as_tensor(X)
        n_spatial = len(self.kernel_size)
        spatial = [
            None
            if X.type.shape[1 + axis] is None
            else (
                X.type.shape[1 + axis] - _window_span(self.kernel_size[axis], self.dilation[axis])
            )
            // self.stride[axis]
            + 1
            for axis in range(n_spatial)
        ]
        out_type = pt.tensor(
            dtype=X.type.dtype,
            shape=(X.type.shape[0], *spatial, *self.kernel_size, X.type.shape[-1]),
        )
        return Apply(self, [X], [out_type])

    def perform(self, node, inputs, outputs):
        (X,) = inputs
        n_spatial = len(self.kernel_size)
        spans = tuple(
            _window_span(self.kernel_size[axis], self.dilation[axis]) for axis in range(n_spatial)
        )

        # (batch, *windows, channels, *spans), a view over X with nothing copied yet
        view = sliding_window_view(X, spans, axis=tuple(range(1, 1 + n_spatial)))
        view = view[
            (
                slice(None),
                *(slice(None, None, step) for step in self.stride),
                slice(None),
                *(slice(None, None, spacing) for spacing in self.dilation),
            )
        ]
        # channels trail the taps in our layout, so move that axis past them and make it contiguous
        order = (
            0,
            *range(1, 1 + n_spatial),
            *range(2 + n_spatial, 2 + 2 * n_spatial),
            1 + n_spatial,
        )
        outputs[0][0] = np.ascontiguousarray(view.transpose(order))

    def infer_shape(self, node, input_shapes):
        [(batch, *spatial, channels)] = input_shapes
        windows = [
            (spatial[axis] - _window_span(self.kernel_size[axis], self.dilation[axis]))
            // self.stride[axis]
            + 1
            for axis in range(len(self.kernel_size))
        ]
        return [[batch, *windows, *self.kernel_size, channels]]

    def pullback(self, inputs, outputs, cotangents):
        """Scatter each window's cotangent back where it was gathered from."""
        (X,) = inputs
        (cotangent,) = cotangents
        spatial = [X.shape[1 + axis] for axis in range(len(self.kernel_size))]
        return [Col2Im(self.kernel_size, self.stride, self.dilation)(cotangent, *spatial)]


class Col2Im(Op):
    """
    Add every window's contribution back at the position it was gathered from.

    The pullback of :class:`Im2Col`, and like it one node a backend can put a real kernel behind rather
    than an indexing graph. Windows overlap, so a position several of them reached accumulates all of
    their contributions, which is what makes this a scatter-add rather than an assignment.

    Parameters
    ----------
    kernel_size, stride, dilation : tuple of int
        One entry per spatial axis, describing the gather being reversed.
    """

    __props__ = ("kernel_size", "stride", "dilation")

    def __init__(
        self,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        dilation: Sequence[int],
    ):
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.dilation = tuple(dilation)

    def __call__(self, *inputs, **kwargs) -> TensorVariable:
        """Narrow the single output to a tensor, which ``Op.__call__`` cannot know it is."""
        out = super().__call__(*inputs, **kwargs)
        assert isinstance(out, TensorVariable), "Col2Im produces one output"
        return out

    def make_node(self, patches, *spatial):
        """Take the spatial extents one scalar at a time, so a known one stays known statically."""
        if len(spatial) != len(self.kernel_size):
            raise ValueError(
                f"Col2Im over {len(self.kernel_size)} spatial axes needs that many extents, got "
                f"{len(spatial)}"
            )
        patches = pt.as_tensor(patches)
        spatial = [pt.as_tensor(length, dtype="int64") for length in spatial]

        lengths = [
            get_scalar_constant_value(length, raise_not_constant=False) for length in spatial
        ]
        out_type = pt.tensor(
            dtype=patches.type.dtype,
            shape=(
                patches.type.shape[0],
                *(None if isinstance(length, Variable) else int(length) for length in lengths),
                patches.type.shape[-1],
            ),
        )
        return Apply(self, [patches, *spatial], [out_type])

    def perform(self, node, inputs, outputs):
        patches, *lengths = inputs
        spatial = tuple(int(length) for length in lengths)
        n_spatial = len(self.kernel_size)

        # One index array per spatial axis, broadcasting to windows-then-taps as the gather's do
        indices = []
        for axis in range(n_spatial):
            span = _window_span(self.kernel_size[axis], self.dilation[axis])
            starts = np.arange(0, spatial[axis] - span + 1, self.stride[axis])
            window = starts[:, None] + np.arange(self.kernel_size[axis]) * self.dilation[axis]
            pattern = [1] * (2 * n_spatial)
            pattern[axis], pattern[n_spatial + axis] = window.shape
            indices.append(window.reshape(pattern))

        out = np.zeros((patches.shape[0], *spatial, patches.shape[-1]), dtype=patches.dtype)
        np.add.at(out, (slice(None), *indices, slice(None)), patches)
        outputs[0][0] = out

    def infer_shape(self, node, input_shapes):
        # Only the patches carry a shape worth reading; the spatial extents are the scalars that
        # give the output its own, and they arrive as further entries in `input_shapes`.
        batch, *_, channels = input_shapes[0]
        return [[batch, *node.inputs[1:], channels]]

    def connection_pattern(self, node):
        """Only the patches reach the output; the spatial extents give it its shape."""
        return [[True], *([False] for _ in self.kernel_size)]

    def pullback(self, inputs, outputs, cotangents):
        """Gather each position's cotangent into every window that reached it."""
        (cotangent,) = cotangents
        gathered = Im2Col(self.kernel_size, self.stride, self.dilation)(cotangent)
        return [gathered, *(disconnected_type() for _ in self.kernel_size)]


def _extract_patches(
    X: TensorVariable,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> TensorVariable:
    """
    Gather every window a kernel visits, for an input with any number of spatial axes.

    Parameters
    ----------
    X : TensorVariable
        Input of shape ``(batch, *spatial, channels)``.
    kernel_size : sequence of int
        Window extent along each spatial axis. Its length is the number of spatial axes.
    stride : sequence of int
        Step between windows along each spatial axis.
    dilation : sequence of int
        Spacing between the taps of one window along each spatial axis.

    Returns
    -------
    patches : TensorVariable
        Shape ``(batch, *out_spatial, *kernel_size, channels)``, where ``out_spatial`` counts the
        windows that fit.
    """
    indices = _window_indices(X, kernel_size, stride, dilation)
    return X[(slice(None), *indices, slice(None))]


class ConvLayer(UnaryLayerOp):
    """
    Cross-correlate an input with a kernel, over any number of spatial axes.

    Marks the convolution as one node so a backend with a real kernel can be dispatched to it, the way
    :class:`~pytensor_ml.layers.attention.AttentionLayer` is. The graph inside gathers every window and
    contracts them against the kernel with a single ``Dot``, whose reduction runs over
    ``prod(kernel_size) * in_channels`` -- deep enough to reach BLAS, which is what a convolution over
    single-channel signal ops cannot offer.

    Correlation, not convolution: the kernel is not flipped, matching torch, keras and flax.
    """

    __props__ = ("kernel_size", "stride", "dilation")

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        **kwargs,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        super().__init__(**kwargs)

    def build_inner_graph(self, X, W, *bias):
        """
        Correlate ``X`` of shape ``(batch, *spatial, in_channels)`` with ``W`` of shape
        ``(*kernel_size, in_channels, out_channels)``, optionally adding ``bias``.
        """
        out = _correlate(X, W, self.kernel_size, self.stride, self.dilation)
        if bias:
            out = out + bias[0]
        return [out]

    def pullback(self, inputs, outputs, cotangents):
        """
        Differentiate through one :class:`ConvLayerGrad` rather than through the inner graph.

        The default would inline the pullback into an anonymous ``OpFromGraph``, which no backend can
        dispatch against, so the whole backward pass would run as the gather and its scatter whatever
        kernels were available. ``compute_dX`` starts True and a rewrite lowers it where the input
        gradient turns out to be unused; see :func:`~pytensor_ml.rewriting.conv.drop_unused_input_grad`.
        """
        X, W, *bias = inputs
        (cotangent,) = cotangents

        dX, dW = ConvLayerGrad(self.kernel_size, self.stride, self.dilation, compute_dX=True)(
            X, W, cotangent
        )
        if not bias:
            return [dX, dW]
        # One bias per output channel, so its gradient sums the cotangent over everything else.
        summed = cotangent.sum(axis=tuple(range(cotangent.ndim - 1)))
        return [dX, dW, summed]


class ConvLayerGrad(LayerOp):
    """
    The pullback of :class:`ConvLayer`, as a node a backend can dispatch against.

    Both gradients are convolutions in their own right -- the input's is a full convolution of the
    cotangent with the flipped kernel, the kernel's a correlation of the input with the cotangent -- so a
    backend with a convolution kernel has one for each. The dispatches reach them by differentiating
    that backend's own convolution rather than by spelling either out.

    Parameters
    ----------
    kernel_size, stride, dilation : tuple of int
        The forward convolution being differentiated.
    compute_dX, compute_dW : bool, optional
        Which gradients to return, in that order. Dropping one is right wherever nothing consumes it:
        the first convolution of a network needs no input gradient, and a transposed convolution needs
        no kernel gradient. At least one must be asked for. Both default to ``True``.
    """

    __props__ = ("kernel_size", "stride", "dilation", "compute_dX", "compute_dW")

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        compute_dX: bool = True,
        compute_dW: bool = True,
        **kwargs,
    ):
        if not (compute_dX or compute_dW):
            raise ValueError(
                "ConvLayerGrad must return at least one gradient, but both compute_dX and compute_dW "
                "are False."
            )
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.compute_dX = compute_dX
        self.compute_dW = compute_dW
        super().__init__(**kwargs)

    def build_inner_graph(self, X, W, cotangent):
        """Differentiate the forward correlation, which is what every dispatch also does."""
        out = _correlate(X, W, self.kernel_size, self.stride, self.dilation)
        wrt = [
            variable for variable, wanted in ((X, self.compute_dX), (W, self.compute_dW)) if wanted
        ]
        return list(pt.grad(cost=None, wrt=wrt, known_grads={out: cotangent}))

    def pullback(self, inputs, outputs, cotangents):
        r"""
        Differentiate through convolutions rather than through the inner graph.

        The default would inline an anonymous ``OpFromGraph`` around the gather, which no backend can
        dispatch against and which only numba can run at all. Each output is linear in two of the three
        inputs and the adjoint of a correlation is the other output, so writing :math:`C` for the
        forward correlation, :math:`T` for the input gradient and :math:`K` for the kernel gradient,

        .. math::

            \langle T(V, W), G \rangle = \langle V, C(G, W) \rangle = \langle W, K(G, V) \rangle

            \langle K(X, V), H \rangle = \langle V, C(X, H) \rangle = \langle X, T(V, H) \rangle

        Each gradient is therefore a :class:`ConvLayer` or a one-sided :class:`ConvLayerGrad`. The input
        gradient does not read ``X`` and the kernel gradient does not read ``W``, so each is
        disconnected from the input the other consumes when its own output is not computed.
        """
        X, W, cotangent = inputs
        geometry = (self.kernel_size, self.stride, self.dilation)
        # One cotangent arrives per output, in output order, so the flags that chose the outputs
        # also say which cotangent belongs to which gradient.
        arriving = iter(cotangents)
        dX_bar = next(arriving) if self.compute_dX else None
        dW_bar = next(arriving) if self.compute_dW else None

        # The cotangent is the one input both outputs read, so its gradient collects a term from each.
        cotangent_terms = []
        if dX_bar is not None:
            cotangent_terms.append(ConvLayer(*geometry)(dX_bar, W))
        if dW_bar is not None:
            cotangent_terms.append(ConvLayer(*geometry)(X, dW_bar))

        dX = (
            disconnected_type()
            if dW_bar is None
            else ConvLayerGrad(*geometry, compute_dW=False)(X, dW_bar, cotangent)
        )
        dW = (
            disconnected_type()
            if dX_bar is None
            else ConvLayerGrad(*geometry, compute_dX=False)(dX_bar, W, cotangent)
        )
        return [dX, dW, reduce(add, cotangent_terms)]


def _correlate(
    X: TensorVariable,
    W: TensorVariable,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> TensorVariable:
    """Cross-correlate ``(batch, *spatial, in_channels)`` with ``(*kernel_size, in_channels, out_channels)``."""
    patches = Im2Col(kernel_size, stride, dilation)(X)
    n_spatial = len(kernel_size)
    # Both reshape targets are taken an axis at a time rather than from a slice of the shape vector.
    # `Shape_i` folds to a constant wherever the extent is known statically, a sliced `Subtensor` does
    # not, and `Reshape` keeps a static output shape only for the entries it can fold.
    kept = [patches.shape[axis] for axis in range(1 + n_spatial)]
    contracted = reduce(mul, (patches.shape[axis] for axis in range(1 + n_spatial, patches.ndim)))

    # Collapsed to a 2-D matmul rather than contracting the 4-D patch tensor directly, because `pt.grad`
    # reads this graph as written: leaving the batch and window axes in place makes the kernel's gradient
    # a batched contraction, one matmul per window row over an intermediate larger than the patch buffer.
    flat = patches.reshape((-1, contracted))
    out = flat @ W.reshape((-1, W.shape[-1]))
    return out.reshape((*kept, W.shape[-1]))


class PoolLayer(UnaryLayerOp):
    """
    Reduce each window a kernel would visit, instead of contracting it against one.

    The same windows :class:`ConvLayer` correlates over, with ``max`` or ``mean`` in place of the matmul,
    so a backend that gathers windows well serves both.

    Parameters
    ----------
    kernel_size, stride, dilation : tuple of int
        One entry per spatial axis.
    reduction : {"max", "mean"}
        Which reduction each window collapses to.
    """

    __props__ = ("kernel_size", "stride", "dilation", "reduction")

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        reduction: str,
        **kwargs,
    ):
        _check_reduction(reduction)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.reduction = reduction
        super().__init__(**kwargs)

    def build_inner_graph(self, X):
        """Collapse the tap axes of ``(batch, *out_spatial, *kernel_size, channels)``."""
        return [_pool(X, self.kernel_size, self.stride, self.dilation, self.reduction)]

    def pullback(self, inputs, outputs, cotangents):
        """
        Differentiate through one :class:`PoolLayerGrad` rather than through the inner graph.

        The default would wrap the pullback in an anonymous ``OpFromGraph``, which registers against no
        type, so the backward pass would gather every window and reduce it again on every backend.
        """
        (X,) = inputs
        (cotangent,) = cotangents
        grad_op = PoolLayerGrad(self.kernel_size, self.stride, self.dilation, self.reduction)
        return [grad_op(X, cotangent)]


class PoolLayerGrad(UnaryLayerOp):
    """
    The pullback of :class:`PoolLayer`, as a node a backend can dispatch against.

    A max pool routes each window's cotangent to the position that won it and a mean pool splits it
    evenly, so a backend with its own pooling primitive has both by differentiating that primitive rather
    than by spelling either out.

    Parameters
    ----------
    kernel_size, stride, dilation : tuple of int
        The forward pooling being differentiated.
    reduction : {"max", "mean"}
        Which reduction the forward collapsed each window to.
    """

    __props__ = ("kernel_size", "stride", "dilation", "reduction")

    def __init__(
        self,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        dilation: tuple[int, ...],
        reduction: str,
        **kwargs,
    ):
        _check_reduction(reduction)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.reduction = reduction
        super().__init__(**kwargs)

    def build_inner_graph(self, X, cotangent):
        """Differentiate the forward reduction, which is what every dispatch also does."""
        out = _pool(X, self.kernel_size, self.stride, self.dilation, self.reduction)
        [gradient] = pt.grad(cost=None, wrt=[X], known_grads={out: cotangent})
        return [gradient]


def _pool(
    X: TensorVariable,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    reduction: str,
) -> TensorVariable:
    """Reduce every window of ``(batch, *spatial, channels)`` to one value per channel."""
    patches = Im2Col(kernel_size, stride, dilation)(X)
    n_spatial = len(kernel_size)
    kept = [patches.shape[axis] for axis in range(1 + n_spatial)]
    # One tap axis rather than one per spatial dimension, so a single reduction covers the window.
    merged = patches.reshape((*kept, -1, patches.shape[-1]))
    return _REDUCTIONS[reduction](merged, axis=1 + n_spatial)


def _resolve_padding(
    padding: str | int | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """
    Turn a padding argument into an explicit ``(before, after)`` pair per spatial axis.

    ``"same"`` pads by the kernel's span less one, which leaves the output spatial size at
    ``ceil(input / stride)`` -- the input's own size at unit stride. That total is odd whenever the span
    is even, and the extra element goes after rather than before, matching torch and keras.

    Parameters
    ----------
    padding : {"valid", "same"}, int, or sequence of int
        No padding, enough to leave the output at ``ceil(input / stride)``, or an explicit amount
        applied to both sides of every axis or of each axis in turn.
    kernel_size : sequence of int
        Window extent along each spatial axis.
    dilation : sequence of int
        Spacing between taps, which stretches the span the padding has to cover.
    """
    if padding == "valid":
        return tuple((0, 0) for _ in kernel_size)
    if padding == "same":
        pads = []
        for extent, spacing in zip(kernel_size, dilation):
            total = spacing * (extent - 1)
            pads.append((total // 2, total - total // 2))
        return tuple(pads)
    if isinstance(padding, str):
        raise ValueError(
            f"padding must be 'valid', 'same', or an explicit number of elements, but got {padding!r}."
        )

    amounts = [padding] * len(kernel_size) if isinstance(padding, int) else list(padding)
    if any(amount < 0 for amount in amounts):
        raise ValueError(
            f"Padding adds elements, so it cannot be negative; got {padding!r}. To use less of the "
            "input than it has, take a slice of it before convolving."
        )
    if len(amounts) != len(kernel_size):
        raise ValueError(
            f"A convolution over {len(kernel_size)} spatial axes needs one padding amount per axis, but "
            f"got {len(amounts)}."
        )
    return tuple((amount, amount) for amount in amounts)


def _as_spatial_tuple(value: int | Sequence[int], n_spatial: int, name: str) -> tuple[int, ...]:
    """Broadcast a scalar argument across the spatial axes, or check one already given per axis."""
    if isinstance(value, int):
        return (value,) * n_spatial
    given = tuple(value)
    if len(given) != n_spatial:
        raise ValueError(
            f"{name} must be an int or one value per spatial axis, but got {len(given)} values for "
            f"{n_spatial} spatial axes."
        )
    return given


def _check_output_survives_cropping(
    X: TensorVariable,
    name: str,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[tuple[int, int]],
    output_padding: Sequence[int],
) -> None:
    """
    Reject a crop that removes the whole output, where the input's length is known at build time.

    A transposed convolution's padding takes elements off its output rather than adding them to its
    input, so enough of it leaves nothing at all: every downstream shape has a zero axis and the graph
    computes an empty answer rather than failing. Whether that happens depends on the input's length as
    well as on the arguments -- a crop that empties a length-1 input can be right for a longer one --
    so it cannot be settled at construction. Only a statically known spatial size can be checked; a
    symbolic one still reaches the op and comes back empty.
    """
    extents = zip(kernel_size, stride, dilation, padding, output_padding)
    for axis, (extent, step, spacing, (before, after), extra) in enumerate(extents):
        length = X.type.shape[1 + axis]
        if length is None:
            continue
        uncropped = (length - 1) * step + spacing * (extent - 1) + 1 + extra
        if uncropped - before - after <= 0:
            raise ValueError(
                f"{name} would crop {before + after} elements from spatial axis {axis}, which the "
                f"transposed convolution grows to only {uncropped}, leaving nothing. Reduce padding, "
                f"or give the layer an input longer than {length} on that axis."
            )


def _check_input_covers_a_window(
    X: TensorVariable,
    name: str,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[tuple[int, int]],
) -> None:
    """
    Reject a padded input too short for one window, where its length is known at build time.

    Such an input yields no windows at all, so every downstream shape has a zero axis and the graph
    computes an empty answer rather than failing. Only a statically known spatial size can be checked; a
    symbolic one still reaches the loop and comes back empty.
    """
    for axis, (extent, spacing, (before, after)) in enumerate(zip(kernel_size, dilation, padding)):
        length = X.type.shape[1 + axis]
        if length is None:
            continue
        span = _window_span(extent, spacing)
        if length + before + after < span:
            raise ValueError(
                f"{name} needs at least {span} elements along spatial axis {axis} to place one window, "
                f"but the input has {length} there and the padding adds {before + after}. Shorten the "
                "kernel, lower the dilation, or pad more."
            )


class _ConvNd(Layer):
    """
    Everything a convolution does that does not depend on how many spatial axes it has.

    Subclasses set :attr:`n_spatial`; see :class:`Conv1D` for the arguments, which are shared.
    """

    n_spatial: int

    def __init__(
        self,
        name: str | None = None,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] = "valid",
        padding_mode: PadMode = "constant",
        bias: bool = True,
        weight_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "in_channels")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _as_spatial_tuple(kernel_size, self.n_spatial, "kernel_size")
        self.stride = _as_spatial_tuple(stride, self.n_spatial, "stride")
        self.dilation = _as_spatial_tuple(dilation, self.n_spatial, "dilation")
        self.padding = _resolve_padding(padding, self.kernel_size, self.dilation)
        self.padding_mode: PadMode = padding_mode
        self.bias = bias

        # Receptive field, then input channels, then output: the layout flax and keras use, and the one
        # `fans` reads, since it takes the two trailing dimensions as the features.
        self.W = trainable_parameter(
            f"{self.name}_W",
            (*self.kernel_size, in_channels, out_channels),
            weight_initializer,
            XavierNormalInitializer(),
            layer_name=self.name,
        )
        if bias:
            self.b = trainable_parameter(
                f"{self.name}_b",
                (out_channels,),
                bias_initializer,
                ZeroInitializer(),
                layer_name=self.name,
            )

    def __call__(self, X: pt.TensorLike) -> TensorVariable:
        """
        Correlate ``X`` of shape ``(batch, *spatial, in_channels)`` with the layer's kernel.

        Returns
        -------
        output : TensorVariable
            Shape ``(batch, *out_spatial, out_channels)``.
        """
        X = pt.as_tensor(X)
        _check_input_rank(X, self.name, self.n_spatial)
        _check_input_covers_a_window(X, self.name, self.kernel_size, self.dilation, self.padding)

        padded = _pad_spatial(X, self.padding, self.padding_mode)
        op = ConvLayer(self.kernel_size, self.stride, self.dilation)
        parameters = (self.W, self.b) if self.bias else (self.W,)

        out = op(padded, *parameters)
        out.name = f"{self.name}_output"
        return out


class Conv1D(_ConvNd):
    r"""
    Cross-correlate a sequence with a learned kernel, over one spatial axis.

    Takes ``(batch, time, in_channels)`` and returns ``(batch, out_time, out_channels)``, so it stacks
    with the recurrent layers without a transpose between them. Each output position is the kernel
    contracted against the window of the input beneath it:

    .. math::

        y_{t,o} = b_o + \sum_{c} \sum_{j} x_{t s + j d,\,c} \, W_{j,c,o},

    with :math:`s` the stride and :math:`d` the dilation. The kernel is not flipped, so this is
    correlation in the signal-processing sense and convolution in the sense every ML framework means.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to the class name when None.
    in_channels : int
        Size of the input's channel axis.
    out_channels : int
        Size of the output's channel axis, one per kernel.
    kernel_size : int
        Window extent along the time axis.
    stride : int, optional
        Step between windows. Default is 1.
    dilation : int, optional
        Spacing between the kernel's taps, which widens the receptive field without adding parameters.
        Default is 1.
    padding : {"valid", "same"} or int, optional
        No padding, enough to leave the output length equal to :math:`\lceil \text{time} / s \rceil`, or
        an explicit number of elements on each side. Default is "valid".
    padding_mode : str, optional
        How padded elements are filled, passed to :func:`pytensor.tensor.pad`. Default is "constant",
        which pads with zeros; "reflect", "symmetric" and "edge" are the other useful ones.
    bias : bool, optional
        Add the learned shift :math:`b`, one per output channel. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W` is drawn. Xavier normal when omitted, whose fans count the receptive field.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.

    Examples
    --------
    Slide a learned kernel along one spatial axis. The layout is channels-last, so a batch of sequences is
    ``(batch, time, channels)``:

    .. code-block:: python

        from pytensor_ml.layers import Conv1D, Input

        X = Input("X", shape=(None, 128, 16))
        features = Conv1D("conv", in_channels=16, out_channels=32, kernel_size=5, padding="same")(X)
    """

    n_spatial = 1


class Conv2D(_ConvNd):
    r"""
    Cross-correlate an image with a learned kernel, over two spatial axes.

    Takes ``(batch, height, width, in_channels)`` and returns ``(batch, out_height, out_width,
    out_channels)``, keeping the channels-last layout the rest of the library uses. Each output position
    is the kernel contracted against the window of the input beneath it:

    .. math::

        y_{h,w,o} = b_o + \sum_{c} \sum_{i,j}
            x_{h s_0 + i d_0,\; w s_1 + j d_1,\; c} \, W_{i,j,c,o},

    with :math:`s` the stride and :math:`d` the dilation, one of each per axis. The kernel is not
    flipped, so this is correlation in the signal-processing sense and convolution in the sense every
    ML framework means.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to the class name when None.
    in_channels : int
        Size of the input's channel axis.
    out_channels : int
        Size of the output's channel axis, one per kernel.
    kernel_size : int or tuple of int
        Window extent, shared by both axes or given as ``(height, width)``.
    stride : int or tuple of int, optional
        Step between windows, shared or per axis. Default is 1.
    dilation : int or tuple of int, optional
        Spacing between the kernel's taps, which widens the receptive field without adding parameters.
        Shared or per axis. Default is 1.
    padding : {"valid", "same"}, int, or tuple of int, optional
        No padding, enough to leave each output extent at :math:`\lceil \text{extent} / s \rceil`, or
        an explicit number of elements on each side, shared or per axis. Default is "valid".
    padding_mode : str, optional
        How padded elements are filled, passed to :func:`pytensor.tensor.pad`. Default is "constant",
        which pads with zeros; "reflect", "symmetric" and "edge" are the other useful ones.
    bias : bool, optional
        Add the learned shift :math:`b`, one per output channel. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W` is drawn. Xavier normal when omitted, whose fans count the receptive field.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.

    Examples
    --------
    The image workhorse, taking ``(batch, height, width, channels)``. Padding ``"valid"`` shrinks each
    spatial extent by ``kernel_size - 1``; ``"same"`` holds it at ``ceil(extent / stride)``:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import Conv2D, Input, Sequential

        X = Input("X", shape=(None, 32, 32, 3))
        network = Sequential(
            Conv2D("conv1", in_channels=3, out_channels=16, kernel_size=3, padding="same"),
            ReLU(),
            Conv2D("conv2", in_channels=16, out_channels=32, kernel_size=3, stride=2),
        )

        features = network(X)
    """

    n_spatial = 2


class _ConvTransposeNd(_ConvNd):
    """
    Everything a transposed convolution does that does not depend on how many spatial axes it has.

    Subclasses set :attr:`n_spatial`; see :class:`ConvTranspose1D` for the arguments, which are shared.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] = 0,
        output_padding: int | Sequence[int] = 0,
        bias: bool = True,
        weight_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        if isinstance(padding, str):
            raise ValueError(
                f"A transposed convolution's padding removes elements from its output rather than "
                f"adding them to its input, so it takes a number of elements rather than {padding!r}."
            )
        super().__init__(
            name,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )
        self.output_padding = _as_spatial_tuple(output_padding, self.n_spatial, "output_padding")
        for axis, (extra, step) in enumerate(zip(self.output_padding, self.stride)):
            if not 0 <= extra < step:
                raise ValueError(
                    f"output_padding picks between the input sizes a stride collapses together, so "
                    f"it must be at least 0 and less than that stride; on spatial axis {axis} it is "
                    f"{extra} against a stride of {step}."
                )

    def __call__(self, X: pt.TensorLike) -> TensorVariable:
        """
        Scatter ``X`` of shape ``(batch, *spatial, in_channels)`` back through the layer's kernel.

        Returns
        -------
        output : TensorVariable
            Shape ``(batch, *out_spatial, out_channels)``, where each output axis is
            ``(spatial - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1``.
        """
        X = pt.as_tensor(X)
        _check_input_rank(X, self.name, self.n_spatial)
        _check_output_survives_cropping(
            X,
            self.name,
            self.kernel_size,
            self.stride,
            self.dilation,
            self.padding,
            self.output_padding,
        )

        # The forward convolution this inverts consumed an input of this size. Several sizes collapse
        # to `X`'s under a stride greater than one, and `output_padding` says which was meant.
        spatial = [X.shape[1 + axis] for axis in range(self.n_spatial)]
        uncropped = [
            (length - 1) * step + spacing * (extent - 1) + 1 + extra
            for length, step, spacing, extent, extra in zip(
                spatial, self.stride, self.dilation, self.kernel_size, self.output_padding
            )
        ]
        placeholder = pt.zeros((X.shape[0], *uncropped, self.out_channels), dtype=X.dtype)

        # The op differentiates a forward correlation, whose kernel runs (taps, in, out) in that
        # convolution's terms -- our output channels then our input ones -- so the stored layout,
        # which `fans` reads as (fan_in, fan_out), is swapped on the way in.
        op = ConvLayerGrad(self.kernel_size, self.stride, self.dilation, compute_dW=False)
        out = op(placeholder, pt.moveaxis(self.W, -1, -2), X)

        if any(before or after for before, after in self.padding):
            cropped = tuple(
                slice(before, size - after)
                for (before, after), size in zip(self.padding, uncropped)
            )
            out = out[(slice(None), *cropped, slice(None))]
        if self.bias:
            out = out + self.b

        out.name = f"{self.name}_output"
        return out


class ConvTranspose1D(_ConvTransposeNd):
    r"""
    Scatter a sequence back through a learned kernel, over one spatial axis.

    Takes ``(batch, time, in_channels)`` and returns ``(batch, out_time, out_channels)``, where
    ``out_time`` is at least as long as ``time`` -- this is the operation that grows a sequence where
    :class:`Conv1D` shrinks it, so it is what a decoder upsamples with. It is the gradient of
    :class:`Conv1D` with respect to that layer's input, which is why it is sometimes called a
    fractionally-strided convolution rather than a deconvolution: it inverts the shape of a
    convolution, not its values.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to the class name when None.
    in_channels : int
        Size of the input's channel axis.
    out_channels : int
        Size of the output's channel axis.
    kernel_size : int
        Window extent along the time axis.
    stride : int, optional
        Step between the windows the output is scattered into, so the factor the input grows by.
        Default is 1.
    dilation : int, optional
        Spacing between the kernel's taps. Default is 1.
    padding : int, optional
        Elements removed from each end of the output, being the inverse of the padding a forward
        convolution would add to its input. Default is 0.
    output_padding : int, optional
        Extra elements added to the far end of the output. A stride greater than one maps several
        input lengths onto the same output length, and this picks between them, so it must be less
        than ``stride``. Default is 0.
    bias : bool, optional
        Add the learned shift, one per output channel. Default is True.
    weight_initializer : Initializer, optional
        How the kernel is drawn. Xavier normal when omitted.
    bias_initializer : Initializer, optional
        How the bias is drawn. Zeros when omitted.

    Examples
    --------
    Run a convolution's gradient forwards, so the sequence grows rather than shrinks. A stride of 2 roughly
    doubles the extent, which is what a decoder wants:

    .. code-block:: python

        from pytensor_ml.layers import ConvTranspose1D, Input

        X = Input("X", shape=(None, 32, 16))
        upsampled = ConvTranspose1D("up", in_channels=16, out_channels=8, kernel_size=4, stride=2)(X)
    """

    n_spatial = 1


class ConvTranspose2D(_ConvTransposeNd):
    r"""
    Scatter an image back through a learned kernel, over two spatial axes.

    Takes ``(batch, height, width, in_channels)`` and returns ``(batch, out_height, out_width,
    out_channels)``, both at least as large as the input's -- the operation that grows an image where
    :class:`Conv2D` shrinks it, and so the one a decoder upsamples with. It is the gradient of
    :class:`Conv2D` with respect to that layer's input.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to the class name when None.
    in_channels : int
        Size of the input's channel axis.
    out_channels : int
        Size of the output's channel axis.
    kernel_size : int or tuple of int
        Window extent, either shared by both axes or given per axis.
    stride : int or tuple of int, optional
        Step between the windows the output is scattered into, so the factor each axis grows by.
        Default is 1.
    dilation : int or tuple of int, optional
        Spacing between the kernel's taps. Default is 1.
    padding : int or tuple of int, optional
        Elements removed from each end of each output axis, being the inverse of the padding a forward
        convolution would add to its input. Default is 0.
    output_padding : int or tuple of int, optional
        Extra elements added to the far end of each output axis, picking between the input sizes a
        stride greater than one collapses together, so each must be less than its stride. Default is 0.
    bias : bool, optional
        Add the learned shift, one per output channel. Default is True.
    weight_initializer : Initializer, optional
        How the kernel is drawn. Xavier normal when omitted.
    bias_initializer : Initializer, optional
        How the bias is drawn. Zeros when omitted.

    Examples
    --------
    The upsampling half of an autoencoder or a generator: each input position paints a kernel-sized patch
    into the output, so a stride of 2 roughly doubles both extents:

    .. code-block:: python

        from pytensor_ml.layers import ConvTranspose2D, Input

        X = Input("X", shape=(None, 8, 8, 64))
        upsampled = ConvTranspose2D("up", in_channels=64, out_channels=32, kernel_size=4, stride=2)(X)
    """

    n_spatial = 2


class _PoolNd(Layer):
    """
    Everything pooling does that does not depend on how many spatial axes it has.

    Subclasses set :attr:`n_spatial` and :attr:`reduction`; see :class:`MaxPool2D` for the arguments,
    which are shared.
    """

    n_spatial: int
    reduction: str

    def __init__(
        self,
        name: str | None = None,
        *,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] | None = None,
        dilation: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] = "valid",
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "kernel_size")
        self.kernel_size = _as_spatial_tuple(kernel_size, self.n_spatial, "kernel_size")
        self.stride = (
            self.kernel_size
            if stride is None
            else _as_spatial_tuple(stride, self.n_spatial, "stride")
        )
        self.dilation = _as_spatial_tuple(dilation, self.n_spatial, "dilation")
        self.padding = _resolve_padding(padding, self.kernel_size, self.dilation)

    def __call__(self, X: pt.TensorLike) -> TensorVariable:
        """
        Reduce each window of ``X``, of shape ``(batch, *spatial, channels)``.

        Returns
        -------
        pooled : TensorVariable
            Shape ``(batch, *out_spatial, channels)``, with the channel axis untouched.
        """
        X = pt.as_tensor(X)
        _check_input_rank(X, self.name, self.n_spatial)
        _check_input_covers_a_window(X, self.name, self.kernel_size, self.dilation, self.padding)

        padded = _pad_spatial(X, self.padding, "constant", _PADDING_IDENTITY[self.reduction])
        out = PoolLayer(self.kernel_size, self.stride, self.dilation, self.reduction)(padded)
        out.name = f"{self.name}_output"
        return out


class MaxPool2D(_PoolNd):
    r"""
    Downsample by taking the largest activation in each window, over two spatial axes.

    Takes ``(batch, height, width, channels)`` and returns ``(batch, out_height, out_width, channels)``.
    Padding fills with :math:`-\infty` so a padded position never wins a window, which zero-filling
    would do wherever every real activation is negative. Where a window ties, the whole gradient goes
    to the earliest tap; a backend that pools with its own kernel may instead split the gradient
    between the tied taps, but every backend returns exactly the gradient the window received.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer. Defaults to the class name when None.
    kernel_size : int or tuple of int, optional
        Window extent, shared by both axes or given as ``(height, width)``. Default is 2.
    stride : int or tuple of int, optional
        Step between windows. Defaults to ``kernel_size``, so windows tile without overlapping --
        the opposite of the convolution layers, which step by one.
    dilation : int or tuple of int, optional
        Spacing between the positions a window covers. Default is 1.
    padding : {"valid", "same"}, int, or tuple of int, optional
        No padding, enough to leave each output extent at :math:`\lceil \text{extent} / s \rceil`, or
        an explicit number of elements on each side. Default is "valid".

    Examples
    --------
    Downsample by keeping the largest activation in each window. Stride defaults to ``kernel_size``, so
    windows tile without overlapping and a 2x2 pool halves both extents:

    .. code-block:: python

        from pytensor_ml.layers import Conv2D, Input, MaxPool2D, Sequential

        X = Input("X", shape=(None, 32, 32, 3))
        network = Sequential(
            Conv2D("conv", in_channels=3, out_channels=16, kernel_size=3, padding="same"),
            MaxPool2D(kernel_size=2),
        )

        features = network(X)
    """

    n_spatial = 2
    reduction = "max"


class MaxPool1D(_PoolNd):
    """
    Downsample a sequence by taking the largest activation in each window; see :class:`MaxPool2D`.

    Examples
    --------
    The one-dimensional pool, keeping the largest activation in each window along the time axis:

    .. code-block:: python

        from pytensor_ml.layers import Conv1D, Input, MaxPool1D, Sequential

        X = Input("X", shape=(None, 128, 16))
        network = Sequential(
            Conv1D("conv", in_channels=16, out_channels=32, kernel_size=5, padding="same"),
            MaxPool1D(kernel_size=2),
        )

        features = network(X)
    """

    n_spatial = 1
    reduction = "max"


class AvgPool2D(_PoolNd):
    """
    Downsample by averaging each window, over two spatial axes.

    Takes ``(batch, height, width, channels)`` and returns ``(batch, out_height, out_width, channels)``.
    Padded positions count toward the average as zeros, matching torch's ``count_include_pad``. See
    :class:`MaxPool2D` for the arguments, which are shared.

    Examples
    --------
    Average each window instead of taking its maximum, which keeps every activation in the window rather
    than routing the whole gradient to one of them:

    .. code-block:: python

        from pytensor_ml.layers import AvgPool2D, Input

        X = Input("X", shape=(None, 32, 32, 16))
        pooled = AvgPool2D(kernel_size=2)(X)
    """

    n_spatial = 2
    reduction = "mean"


class AvgPool1D(_PoolNd):
    """
    Downsample a sequence by averaging each window; see :class:`AvgPool2D`.

    Examples
    --------
    The one-dimensional average pool, smoothing along the time axis rather than picking a winner per window:

    .. code-block:: python

        from pytensor_ml.layers import AvgPool1D, Input

        X = Input("X", shape=(None, 128, 16))
        pooled = AvgPool1D(kernel_size=2)(X)
    """

    n_spatial = 1
    reduction = "mean"


def _nearest_source_indices(
    in_extent: int | TensorVariable, out_extent: int | TensorVariable
) -> TensorVariable:
    """Index of the input element each output position copies, as ``floor(position * in / out)``.
    Spelled as an integer division so the ratio is never rounded before the floor."""
    return pt.arange(out_extent) * in_extent // out_extent


def _linear_source_coordinates(
    in_extent: int | TensorVariable, out_extent: int | TensorVariable, align_corners: bool
) -> TensorVariable:
    """Fractional input position each output position samples, in input coordinates. Without
    ``align_corners`` the outermost samples fall outside the input, which is what the clamp handles."""
    # Cast the extents before dividing: they arrive as int64, and an int64 division is float64,
    # which would carry the whole interpolation off floatX and back in the layer's output.
    in_span = pt.cast(in_extent, config.floatX)
    out_span = pt.cast(out_extent, config.floatX)

    positions = pt.arange(out_extent, dtype=config.floatX)
    if align_corners:
        # The maximum keeps a single-element output spreading over nothing rather than dividing by it.
        source = positions * ((in_span - 1) / pt.maximum(out_span - 1, 1.0))
    else:
        source = (positions + 0.5) * (in_span / out_span) - 0.5
    return pt.clip(source, 0.0, in_span - 1)


def _resample_axis(
    X: TensorVariable,
    axis: int,
    out_extent: int | TensorVariable,
    mode: str,
    align_corners: bool,
) -> TensorVariable:
    """Resample one spatial axis of ``X`` to ``out_extent``, leaving every other axis alone."""
    in_extent = X.shape[axis]
    if mode == "nearest":
        return pt.take(X, _nearest_source_indices(in_extent, out_extent), axis=axis)

    source = _linear_source_coordinates(in_extent, out_extent, align_corners)
    lower_position = pt.floor(source)
    lower_index = pt.cast(lower_position, "int64")
    upper_index = pt.minimum(lower_index + 1, in_extent - 1)

    # The weight varies along `axis` alone, so it broadcasts against every other axis of the gather.
    along_axis = (np.newaxis,) * axis + (slice(None),) + (np.newaxis,) * (X.ndim - axis - 1)
    weight = (source - lower_position)[along_axis]

    lower = pt.take(X, lower_index, axis=axis)
    upper = pt.take(X, upper_index, axis=axis)
    return lower + (upper - lower) * weight


class _UpsampleNd(Layer):
    """
    Everything resampling does that does not depend on how many spatial axes it has.

    Subclasses set :attr:`n_spatial` and :attr:`linear_mode`, the name torch gives linear
    interpolation at that rank; see :class:`Upsample2D` for the arguments, which are shared.
    """

    n_spatial: int
    linear_mode: str

    def __init__(
        self,
        name: str | None = None,
        *,
        scale_factor: int | Sequence[int] | None = None,
        size: int | Sequence[int] | None = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "scale_factor")

        if (scale_factor is None) == (size is None):
            raise ValueError(
                f"{self.name} resizes either by a factor or to an extent, so pass exactly one of "
                f"scale_factor and size."
            )
        if mode not in ("nearest", self.linear_mode):
            raise ValueError(
                f"{self.name} interpolates either 'nearest' or '{self.linear_mode}', but got "
                f"{mode!r}."
            )
        if align_corners and mode == "nearest":
            raise ValueError(
                f"{self.name} has no corners to align when it copies the nearest element, so "
                f"align_corners means nothing with mode='nearest'."
            )

        self.mode = mode
        self.align_corners = align_corners
        self.scale_factor = (
            None
            if scale_factor is None
            else _as_spatial_tuple(scale_factor, self.n_spatial, "scale_factor")
        )
        self.size = None if size is None else _as_spatial_tuple(size, self.n_spatial, "size")

        argument, extents = (
            ("size", self.size)
            if self.scale_factor is None
            else ("scale_factor", self.scale_factor)
        )
        assert extents is not None
        if any(extent < 1 for extent in extents):
            raise ValueError(
                f"{self.name} needs a positive {argument} on every spatial axis, but got {extents}."
            )

    def _output_extents(self, X: TensorVariable) -> tuple:
        """Output extent per spatial axis, as a Python int wherever the input's extent is known so
        the resampled type keeps its static shape."""
        if self.size is not None:
            return self.size

        assert self.scale_factor is not None
        extents = []
        for axis, factor in enumerate(self.scale_factor, start=1):
            static_extent = X.type.shape[axis]
            extents.append(
                X.shape[axis] * factor if static_extent is None else static_extent * factor
            )
        return tuple(extents)

    def __call__(self, X: pt.TensorLike) -> TensorVariable:
        """
        Resample the spatial axes of ``X``, of shape ``(batch, *spatial, channels)``.

        Returns
        -------
        resampled : TensorVariable
            Shape ``(batch, *out_spatial, channels)``, with the channel axis untouched.
        """
        X = pt.as_tensor(X)
        _check_input_rank(X, self.name, self.n_spatial)

        out = X
        for axis, out_extent in enumerate(self._output_extents(X), start=1):
            out = _resample_axis(out, axis, out_extent, self.mode, self.align_corners)

        out.name = f"{self.name}_output"
        return out


class Upsample2D(_UpsampleNd):
    r"""
    Resample an image to a new spatial extent, over two spatial axes.

    Takes ``(batch, height, width, channels)`` and returns ``(batch, out_height, out_width,
    channels)``, leaving the channel axis untouched. This is a gather, not a learned operation: it
    has no parameters, and it is what modern decoders use to raise resolution before an ordinary
    convolution, in place of the transposed convolution :class:`ConvTranspose2D` provides.

    Nearest-neighbor copies the input element at :math:`\lfloor j \cdot H / H' \rfloor` to output
    position :math:`j`. Bilinear interpolates along each axis in turn, between the two input
    elements the output position falls between.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer. Defaults to the class name when None.
    scale_factor : int or tuple of int, optional
        Multiply each spatial extent by this, shared by both axes or given as ``(height, width)``.
        Pass this or ``size``, never both.
    size : int or tuple of int, optional
        Resample to this extent instead, shared by both axes or given as ``(height, width)``. Pass
        this or ``scale_factor``, never both.
    mode : {"nearest", "bilinear"}, optional
        Copy the nearest input element, or interpolate linearly along each axis. Default is
        "nearest".
    align_corners : bool, optional
        Pin the outermost output elements to the outermost input elements, spreading the rest evenly
        between them. When False, each element instead covers a unit interval and output centers map
        to input centers, which shifts the sampled positions by half a pixel and is what torch and
        every diffusion decoder use. Only meaningful for ``mode="bilinear"``. Default is False.

    Examples
    --------
    Double the resolution and then convolve, which is how a decoder ordinarily upsamples. The 3x3
    convolution is what does the learning; the resampling only supplies it a larger grid:

    .. code-block:: python

        from pytensor_ml.layers import Conv2D, Input, Sequential, Upsample2D

        X = Input("X", shape=(None, 32, 32, 64))
        network = Sequential(
            Upsample2D(scale_factor=2),
            Conv2D("conv", in_channels=64, out_channels=64, kernel_size=3, padding="same"),
        )

        features = network(X)
    """

    n_spatial = 2
    linear_mode = "bilinear"


class Upsample1D(_UpsampleNd):
    """
    Resample a sequence to a new extent; see :class:`Upsample2D`.

    Linear interpolation is spelled ``mode="linear"`` at this rank, as it is in torch.

    Examples
    --------
    Stretch a sequence to twice its length, interpolating between neighboring steps rather than
    repeating each one:

    .. code-block:: python

        from pytensor_ml.layers import Input, Upsample1D

        X = Input("X", shape=(None, 128, 16))
        stretched = Upsample1D(scale_factor=2, mode="linear")(X)
    """

    n_spatial = 1
    linear_mode = "linear"
