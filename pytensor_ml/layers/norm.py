import numpy as np
import pytensor.tensor as pt

from pytensor import config

from pytensor_ml.base import Layer, LayerOp, UnaryLayerOp, _resolve_layer_name
from pytensor_ml.params import NonTrainableParameter, TrainableParameter, non_trainable, trainable
from pytensor_ml.state import Initializer, OneInitializer, ZeroInitializer


def _batch_axes(X: pt.TensorVariable) -> tuple[int, ...]:
    """Every axis a batch statistic is taken over, which is all of them but the feature axis at the end."""
    return tuple(range(X.ndim - 1))


def _standardize(X, epsilon, axis, keepdims=False):
    """
    Center and scale ``X`` over ``axis``, returning the statistics used.

    Notes
    -----
    The variance is biased (``ddof=0``), matching :class:`torch.nn.BatchNorm1d` and
    :class:`torch.nn.LayerNorm`; pretrained weights assume it. Do not "correct" it to the unbiased
    estimator -- that would diverge from every pretrained model.

    Returns
    -------
    X_standardized : TensorVariable
        ``X`` centered and scaled to unit variance.
    mu : TensorVariable
        Mean of ``X`` over ``axis``.
    sigma_sq : TensorVariable
        Biased variance of ``X`` over ``axis``.
    """
    # The statistics accumulate at float32 or better whatever X is stored as. Squaring a float16
    # activation overflows past |x| of about 256, and a diffusion decoder's activations reach the
    # thousands -- the variance would go to infinity and standardizing would return NaN.
    accumulator = "float32" if X.dtype in ("float16", "bfloat16") else X.dtype
    wide = X.astype(accumulator)

    mu = wide.mean(axis=axis, keepdims=keepdims)
    sigma_sq = wide.var(axis=axis, keepdims=keepdims)
    standardized = (wide - mu) / pt.sqrt(sigma_sq + epsilon)

    return standardized.astype(X.dtype), mu.astype(X.dtype), sigma_sq.astype(X.dtype)


def _affine_input_count(affine: bool) -> int:
    """Number of inputs the learned affine transform contributes, which every norm op places directly
    after ``X``. Both the graph builders and :meth:`BatchNormLayer.update_map` index around it."""
    return 2 if affine else 0


def _split_affine(affine: bool, rest: tuple) -> tuple[tuple, tuple]:
    """
    Split an op's inputs after ``X`` into the affine pair and whatever that op expects behind it.

    Returns
    -------
    affine_params : tuple
        ``(loc, scale)``, or empty when the layer has no affine transform.
    extras : tuple
        The remaining inputs, such as batch norm's running statistics.
    """
    n_affine = _affine_input_count(affine)
    return rest[:n_affine], rest[n_affine:]


def _rescale(X_normalized, affine_params: tuple):
    if not affine_params:
        return X_normalized

    loc, scale = affine_params
    return X_normalized * scale + loc


def _resolve_n_in(name: str, n_in: int | None, X: pt.TensorVariable | None) -> int | None:
    """Resolve the feature-axis size, or ``None`` while the layer is still waiting for an input to
    infer it from."""
    if X is None:
        return n_in

    inferred = X.type.shape[-1]
    if inferred is None:
        raise ValueError(
            f"{name} cannot infer n_in from an input whose last dimension is unknown. Pass n_in "
            f"explicitly, or give the input a static feature dimension."
        )
    return inferred


def _affine_parameters(
    name: str,
    n_in: int,
    loc_initializer: Initializer | None = None,
    scale_initializer: Initializer | None = None,
) -> tuple[TrainableParameter, TrainableParameter]:
    """Build the learned shift and scale. Returns them in the ``(loc, scale)`` order that every norm
    op unpacks its inputs in, so the two cannot drift apart.

    Both declare their initializer, so a redraw returns them to the identity transform -- normalizing and
    then rescaling by a random factor defeats the point of the layer. A caller who wants something else says
    so, and their choice becomes the declaration.
    """
    resolved_loc = ZeroInitializer() if loc_initializer is None else loc_initializer
    resolved_scale = OneInitializer() if scale_initializer is None else scale_initializer
    loc = trainable(
        resolved_loc.initial_value((n_in,)),
        f"{name}_loc",
        initializer=resolved_loc,
        layer_name=name,
    )
    scale = trainable(
        resolved_scale.initial_value((n_in,)),
        f"{name}_scale",
        initializer=resolved_scale,
        layer_name=name,
    )
    return loc, scale


class BatchNormLayer(LayerOp):
    __props__ = ("n_in", "epsilon", "momentum", "affine")

    def update_map(self):
        # Outputs 1 and 2 are the new running mean and variance; they write back to the running
        # statistics they were computed from, which follow X and the affine pair.
        running_mean_index = 1 + _affine_input_count(self.affine)
        return {1: running_mean_index, 2: running_mean_index + 1}

    def build_inner_graph(self, X, *rest):
        affine_params, (running_mean, running_var) = _split_affine(self.affine, rest)
        X_normalized, mu, sigma_sq = _standardize(X, self.epsilon, axis=_batch_axes(X))

        new_running_mean = self.momentum * mu + (1 - self.momentum) * running_mean
        new_running_var = self.momentum * sigma_sq + (1 - self.momentum) * running_var

        return [_rescale(X_normalized, affine_params), new_running_mean, new_running_var]


class NoRunningStatsBatchNormLayer(LayerOp):
    __props__ = ("n_in", "epsilon", "affine")

    def build_inner_graph(self, X, *rest):
        # Reports the batch statistics to match BatchNormLayer's arity; declaring no update_map is what
        # keeps them from being written back to anything.
        affine_params, _ = _split_affine(self.affine, rest)
        X_normalized, mu, sigma_sq = _standardize(X, self.epsilon, axis=_batch_axes(X))

        return [_rescale(X_normalized, affine_params), mu, sigma_sq]


class PredictionBatchNormLayer(UnaryLayerOp):
    __props__ = ("n_in", "epsilon", "affine")

    def build_inner_graph(self, X, *rest):
        affine_params, (running_mean, running_var) = _split_affine(self.affine, rest)
        X_normalized = (X - running_mean) / pt.sqrt(running_var + self.epsilon)

        return [_rescale(X_normalized, affine_params)]


class BatchNorm(Layer):
    r"""
    Batch normalization over every axis but the last.

    Standardize each feature across the batch, then optionally apply a learned affine transform:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta,

    where the mean and (biased) variance are taken over every axis but the last, and the last axis is
    the feature. On a flat ``(batch, features)`` input that is the batch axis alone, which is what torch
    calls ``BatchNorm1d``; on the ``(batch, height, width, channels)`` a convolution produces it is the
    batch and both spatial axes, giving one statistic per channel, which torch calls ``BatchNorm2d``.
    During training the batch statistics are used and the running mean and variance are updated toward
    them as :math:`(1 - m)\,r + m\,b` from each batch statistic :math:`b`.

    Parameters
    ----------
    name : str, optional
        Name used as a prefix for the layer's parameters. Default is "BatchNorm".
    n_in : int, optional
        Size of the feature axis. Inferred from the input's last dimension on the first call when
        omitted.
    epsilon : float, optional
        Constant :math:`\epsilon` added to the variance for numerical stability. Default is 1e-5.
    momentum : float, optional
        Weight :math:`m` of the current batch statistic in the running-average update. Default is
        0.1.
    affine : bool, optional
        Apply the learned scale :math:`\gamma` and shift :math:`\beta`, starting from the identity
        transform :math:`\gamma = 1`, :math:`\beta = 0`, which a redraw returns them to. Default is True.
    scale_initializer : Initializer, optional
        How :math:`\gamma` is drawn. Ones when omitted, which is the identity transform; drawing a random
        factor to rescale a normalized activation by would defeat the layer.
    loc_initializer : Initializer, optional
        How :math:`\beta` is drawn. Zeros when omitted.
    track_running_stats : bool, optional
        Maintain running mean and variance for use at prediction time. If False, each batch is
        normalized against its own sample statistics at both training and inference time.
        Default is True.

    Notes
    -----
    Batch normalization is not symmetric between training and prediction, so inference needs
    special handling. A plain forward pass -- calling the layer, or compiling with
    :func:`function` -- normalizes with the *current batch's* mean and variance, making each
    output depend on the other samples in the batch. That is what you want while training, but
    wrong at inference, where an example must normalize the same way no matter what it happens to
    be batched with. Compile prediction graphs with :func:`compile_predict`, which applies
    :func:`rewrite_for_prediction` to substitute the accumulated running statistics for the batch
    statistics.

    Examples
    --------
    Normalize each feature over the batch, keeping running statistics so inference does not depend on which
    other rows happen to share the batch. :meth:`~pytensor_ml.model.Model.predict` swaps in those running
    statistics automatically, so the training and inference graphs differ here:

    .. code-block:: python

        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import BatchNorm, Input, Linear, Sequential

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32, bias=False),
            BatchNorm("bn", n_in=32),
            ReLU(),
        )

        activations = network(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_in: int | None = None,
        epsilon: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        scale_initializer: Initializer | None = None,
        loc_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "n_in")
        self.n_in = n_in
        self.epsilon = epsilon
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._scale_initializer = scale_initializer
        self._loc_initializer = loc_initializer

        self.scale: TrainableParameter | None = None
        self.loc: TrainableParameter | None = None
        self.running_mean: NonTrainableParameter | None = None
        self.running_var: NonTrainableParameter | None = None
        self.new_running_mean: pt.TensorVariable | None = None
        self.new_running_var: pt.TensorVariable | None = None

        self.initialized = False
        self._initialize_params(None)

    def _initialize_params(self, X: pt.TensorVariable | None):
        if self.initialized:
            return

        n_in = _resolve_n_in(self.name, self.n_in, X)
        if n_in is None:
            return

        if self.affine:
            self.loc, self.scale = _affine_parameters(
                self.name,
                n_in,
                loc_initializer=self._loc_initializer,
                scale_initializer=self._scale_initializer,
            )

        if self.track_running_stats:
            zeros = np.zeros(n_in, dtype=config.floatX)
            ones = np.ones(n_in, dtype=config.floatX)
            self.running_mean = non_trainable(
                zeros, f"{self.name}_running_mean", layer_name=self.name
            )
            self.running_var = non_trainable(ones, f"{self.name}_running_var", layer_name=self.name)

        self.initialized = True

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)
        if X.ndim < 2:
            raise ValueError(
                f"{self.name} takes statistics over every axis but the last, so it needs at least a "
                f"batch axis and a feature axis; got a {X.ndim}-dimensional input."
            )
        inputs = [X]

        self._initialize_params(X)

        if self.affine:
            assert self.scale is not None and self.loc is not None
            inputs.extend([self.loc, self.scale])

        if self.track_running_stats:
            assert self.running_mean is not None and self.running_var is not None
            # Applying one layer object again reads what the previous application wrote, so every
            # application contributes. The chain is built here because call order is known only while the
            # graph is being built; two finished branches carry no order between them.
            inputs.extend(
                [
                    self.new_running_mean
                    if self.new_running_mean is not None
                    else self.running_mean,
                    self.new_running_var if self.new_running_var is not None else self.running_var,
                ]
            )
            batch_norm_op: LayerOp = BatchNormLayer(
                name=self.name,
                n_in=self.n_in,
                epsilon=self.epsilon,
                momentum=self.momentum,
                affine=self.affine,
            )
        else:
            batch_norm_op = NoRunningStatsBatchNormLayer(
                name=self.name,
                n_in=self.n_in,
                epsilon=self.epsilon,
                affine=self.affine,
            )

        X_transformed, batch_mean, batch_var = batch_norm_op(*inputs)
        if self.track_running_stats:
            self.new_running_mean, self.new_running_var = batch_mean, batch_var

        X_transformed.name = f"{self.name}_output"

        return X_transformed


class LayerNormLayer(UnaryLayerOp):
    __props__ = ("n_in", "epsilon", "affine")

    def build_inner_graph(self, X, *rest):
        affine_params, _ = _split_affine(self.affine, rest)
        X_normalized, _, _ = _standardize(X, self.epsilon, axis=-1, keepdims=True)

        return [_rescale(X_normalized, affine_params)]


class LayerNorm(Layer):
    r"""
    Layer normalization over the last (feature) axis.

    Standardize each sample independently across its features, then optionally apply a learned
    affine transform:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta,

    where the mean and (biased) variance are taken over the last axis. The statistics depend only on
    the current sample, so there are no running statistics and no train/eval distinction.

    Parameters
    ----------
    name : str, optional
        Name used as a prefix for the layer's parameters. Default is "LayerNorm".
    n_in : int, optional
        Size of the normalized feature axis. Inferred from the input's last dimension on the first
        call when omitted.
    epsilon : float, optional
        Constant :math:`\epsilon` added to the variance for numerical stability. Default is 1e-5.
    affine : bool, optional
        Apply the learned scale :math:`\gamma` and shift :math:`\beta`, starting from the identity
        transform :math:`\gamma = 1`, :math:`\beta = 0`, which a redraw returns them to. Default is True.
    scale_initializer : Initializer, optional
        How :math:`\gamma` is drawn. Ones when omitted, which is the identity transform; drawing a random
        factor to rescale a normalized activation by would defeat the layer.
    loc_initializer : Initializer, optional
        How :math:`\beta` is drawn. Zeros when omitted.

    Examples
    --------
    Normalize each row over its own features, so no row depends on the others. That independence is why a
    transformer uses it rather than :class:`BatchNorm`, and it needs no running statistics:

    .. code-block:: python

        from pytensor_ml.layers import Input, LayerNorm, Linear, Sequential

        X = Input("X", shape=(None, 128, 256))
        network = Sequential(
            LayerNorm("ln", n_in=256),
            Linear("fc", n_in=256, n_out=256),
        )

        activations = network(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_in: int | None = None,
        epsilon: float = 1e-5,
        affine: bool = True,
        scale_initializer: Initializer | None = None,
        loc_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "n_in")
        self.n_in = n_in
        self.epsilon = epsilon
        self.affine = affine
        self._scale_initializer = scale_initializer
        self._loc_initializer = loc_initializer

        self.scale: TrainableParameter | None = None
        self.loc: TrainableParameter | None = None

        self.initialized = False
        self._initialize_params(None)

    def _initialize_params(self, X: pt.TensorVariable | None):
        if self.initialized:
            return

        n_in = _resolve_n_in(self.name, self.n_in, X)
        if n_in is None:
            return

        if self.affine:
            self.loc, self.scale = _affine_parameters(
                self.name,
                n_in,
                loc_initializer=self._loc_initializer,
                scale_initializer=self._scale_initializer,
            )

        self.initialized = True

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)
        self._initialize_params(X)

        inputs = [X]
        if self.affine:
            assert self.scale is not None and self.loc is not None
            inputs.extend([self.loc, self.scale])

        X_transformed = LayerNormLayer(
            name=self.name,
            n_in=self.n_in,
            epsilon=self.epsilon,
            affine=self.affine,
        )(*inputs)
        X_transformed.name = f"{self.name}_output"

        return X_transformed


class GroupNormLayer(UnaryLayerOp):
    __props__ = ("n_in", "n_groups", "epsilon", "affine")

    def build_inner_graph(self, X, *rest):
        affine_params, _ = _split_affine(self.affine, rest)
        channels_per_group = X.shape[-1] // self.n_groups
        grouped = pt.split_dims(X, shape=(self.n_groups, channels_per_group), axis=-1)

        # X's spatial axes, which the split leaves in place, and the group's own channels at the end.
        statistic_axes = (*range(1, X.ndim - 1), -1)
        X_grouped, _, _ = _standardize(grouped, self.epsilon, axis=statistic_axes, keepdims=True)
        X_normalized = pt.join_dims(X_grouped, start_axis=-2, n_axes=2)

        return [_rescale(X_normalized, affine_params)]


class GroupNorm(Layer):
    r"""
    Group normalization over a channel grouping and every spatial axis.

    Split the channels into ``n_groups`` contiguous groups, standardize each group of each sample
    independently, then optionally apply a learned per-channel affine transform:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta,

    where the mean and (biased) variance are taken over one group's channels together with every
    spatial axis, giving one statistic per sample per group. Like :class:`LayerNorm` and unlike
    :class:`BatchNorm`, the statistics depend only on the current sample, so there are no running
    statistics and no train/eval distinction, which is why convolutional networks trained at small
    batch sizes reach for it.

    ``n_groups = 1`` puts every channel in one group and normalizes each sample's whole feature map
    at once; ``n_groups = n_in`` gives each channel its own statistic, which is instance
    normalization.

    Parameters
    ----------
    name : str, optional
        Name used as a prefix for the layer's parameters. Default is "GroupNorm".
    n_groups : int
        Number of groups :math:`G` the channels are split into. Must divide the channel count.
    n_in : int, optional
        Size of the channel axis. Inferred from the input's last dimension on the first call when
        omitted.
    epsilon : float, optional
        Constant :math:`\epsilon` added to the variance for numerical stability. Default is 1e-5.
    affine : bool, optional
        Apply the learned scale :math:`\gamma` and shift :math:`\beta`, one entry per channel,
        starting from the identity transform :math:`\gamma = 1`, :math:`\beta = 0`, which a redraw
        returns them to. Default is True.
    scale_initializer : Initializer, optional
        How :math:`\gamma` is drawn. Ones when omitted, which is the identity transform; drawing a
        random factor to rescale a normalized activation by would defeat the layer.
    loc_initializer : Initializer, optional
        How :math:`\beta` is drawn. Zeros when omitted.

    Examples
    --------
    Normalize a convolution's output over 8 groups of its 32 channels. The input is channel-last,
    ``(batch, height, width, channels)``, so each of the 8 statistics covers 4 channels and the whole
    spatial extent of one image:

    .. code-block:: python

        from pytensor_ml.activations import Swish
        from pytensor_ml.layers import Conv2D, GroupNorm, Input, Sequential

        X = Input("X", shape=(None, 64, 64, 3))
        network = Sequential(
            Conv2D("conv", in_channels=3, out_channels=32, kernel_size=3, padding="same"),
            GroupNorm("gn", n_groups=8, n_in=32),
            Swish(),
        )

        activations = network(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_groups: int,
        n_in: int | None = None,
        epsilon: float = 1e-5,
        affine: bool = True,
        scale_initializer: Initializer | None = None,
        loc_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "n_groups")
        if n_groups < 1:
            raise ValueError(f"{self.name} needs at least one group, but got n_groups={n_groups}.")

        self.n_groups = n_groups
        self.n_in = n_in
        self.epsilon = epsilon
        self.affine = affine
        self._scale_initializer = scale_initializer
        self._loc_initializer = loc_initializer

        self.scale: TrainableParameter | None = None
        self.loc: TrainableParameter | None = None

        self.initialized = False
        self._initialize_params(None)

    def _initialize_params(self, X: pt.TensorVariable | None):
        if self.initialized:
            return

        n_in = _resolve_n_in(self.name, self.n_in, X)
        if n_in is None:
            return

        if n_in % self.n_groups:
            raise ValueError(
                f"{self.name} splits the channels into equal groups, so n_groups must divide the "
                f"channel count; {self.n_groups} groups do not divide {n_in} channels."
            )

        if self.affine:
            self.loc, self.scale = _affine_parameters(
                self.name,
                n_in,
                loc_initializer=self._loc_initializer,
                scale_initializer=self._scale_initializer,
            )

        self.initialized = True

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)
        if X.ndim < 2:
            raise ValueError(
                f"{self.name} groups the last axis and reduces over the spatial axes before it, so "
                f"it needs at least a batch axis and a channel axis; got a {X.ndim}-dimensional "
                f"input."
            )

        self._initialize_params(X)

        inputs = [X]
        if self.affine:
            assert self.scale is not None and self.loc is not None
            inputs.extend([self.loc, self.scale])

        X_transformed = GroupNormLayer(
            name=self.name,
            n_in=self.n_in,
            n_groups=self.n_groups,
            epsilon=self.epsilon,
            affine=self.affine,
        )(*inputs)
        X_transformed.name = f"{self.name}_output"

        return X_transformed
