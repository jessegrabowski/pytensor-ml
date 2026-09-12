from typing import Literal, get_args

import pytensor.tensor as pt

from pytensor.tensor.type import float_dtypes
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import UnaryLayerOp, VariadicLayer, _resolve_layer_name

Pairing = Literal["half", "adjacent"]
Scaling = Literal["none", "linear", "ntk"]

_PAIRINGS = get_args(Pairing)
_SCALINGS = get_args(Scaling)


def _validate_options(pairing: str, scaling: str, scaling_factor: float) -> None:
    """Reject unknown pairing and scaling options."""
    if pairing not in _PAIRINGS:
        raise ValueError(f"pairing must be one of {_PAIRINGS}, got {pairing!r}")
    if scaling not in _SCALINGS:
        raise ValueError(f"scaling must be one of {_SCALINGS}, got {scaling!r}")
    if scaling != "none" and scaling_factor <= 0:
        raise ValueError(f"scaling_factor must be positive, got {scaling_factor}")


def _head_dim(x: TensorVariable) -> int | TensorVariable:
    """Return the feature size, symbolic when it is not statically known."""
    if x.type.dtype not in float_dtypes:
        raise ValueError(f"RotaryEmbedding needs a floating-point input, got dtype {x.type.dtype}.")

    static_size = x.type.shape[-1]
    if static_size is None:
        return x.shape[-1]
    if static_size % 2:
        raise ValueError(
            f"RotaryEmbedding rotates channel pairs, so head_dim must be even, got {static_size}."
        )

    return static_size


def _add_head_axes(
    angles: TensorVariable, x: TensorVariable, position_ids: TensorVariable
) -> TensorVariable:
    """Unsqueeze ``angles`` so its sequence axis lines up with ``x``'s.

    ``position_ids`` indexes tokens, so its last axis is the sequence and any leading axes are batch
    axes. ``x`` carries head axes between the two, and the number of them follows from the two ranks --
    which is why the caller does not pass an axis to unsqueeze.
    """
    n_head_axes = (x.type.ndim - 1) - position_ids.type.ndim
    if n_head_axes < 0:
        raise ValueError(
            f"position_ids has {position_ids.type.ndim} dimensions, more than the "
            f"{x.type.ndim - 1} non-feature dimensions of x."
        )
    if n_head_axes == 0:
        return angles

    aligned_angles = angles[(Ellipsis, *(None,) * n_head_axes, slice(None), slice(None))]
    return aligned_angles


def _inverse_frequencies(
    head_dim: int | TensorVariable, dtype: str, base: float, scaling: str, scaling_factor: float
) -> TensorVariable:
    r"""
    Angular frequencies :math:`\theta_i = \mathrm{base}^{-2i/d}`, one per rotated pair.

    Returns
    -------
    inverse_frequencies : TensorVariable
        Shape ``(head_dim // 2,)``, dtype ``dtype``. Folds to a constant when ``head_dim`` is static.
    """
    dimension = pt.cast(head_dim, dtype)
    frequency_base = pt.constant(base, dtype=dtype)
    if scaling == "ntk":
        if isinstance(head_dim, int) and head_dim <= 2:
            raise ValueError(
                f"NTK scaling rescales the base by scaling_factor ** (d / (d - 2)), which is "
                f"undefined for head_dim <= 2; got {head_dim}."
            )
        # Static NTK scaling changes the base independently of the sequence length.
        factor = pt.constant(scaling_factor, dtype=dtype)
        frequency_base = frequency_base * factor ** (dimension / (dimension - 2))

    exponent = pt.arange(0, head_dim, 2, dtype=dtype) / dimension
    inverse_frequencies = frequency_base**-exponent

    if scaling == "linear":
        inverse_frequencies = inverse_frequencies / pt.constant(scaling_factor, dtype=dtype)

    inverse_frequencies.name = "inverse_frequencies"

    return inverse_frequencies


def _split_pairs(
    x: TensorVariable, pairing: str, head_dim: int | TensorVariable
) -> tuple[TensorVariable, TensorVariable]:
    """Split the feature axis into the two members of every rotated pair.

    The conventions are a permutation of the feature axis apart and are not interchangeable: weights
    trained under one produce nonsense under the other.

    ``"half"`` pairs channel ``i`` with ``i + d/2``; ``"adjacent"`` pairs ``2i`` with ``2i + 1``.
    """
    if pairing == "half":
        half = head_dim // 2
        first, second = x[..., :half], x[..., half:]

    else:
        first, second = x[..., 0::2], x[..., 1::2]

    return first, second


def _join_pairs(
    x: TensorVariable, first: TensorVariable, second: TensorVariable, pairing: str
) -> TensorVariable:
    """Reassemble the feature axis, inverting :func:`_split_pairs` for the same ``pairing``."""
    if pairing == "half":
        rotated = pt.concatenate([first, second], axis=-1)
    else:
        rotated = x[..., 0::2].set(first)
        rotated = rotated[..., 1::2].set(second)
    return rotated


class RotaryEmbeddingLayer(UnaryLayerOp):
    __props__ = ("base", "pairing", "scaling", "scaling_factor")

    def build_inner_graph(self, x, position_ids):
        _validate_options(
            pairing=self.pairing, scaling=self.scaling, scaling_factor=self.scaling_factor
        )
        head_dim = _head_dim(x)
        dtype = x.type.dtype

        inverse_frequencies = _inverse_frequencies(
            head_dim=head_dim,
            dtype=dtype,
            base=self.base,
            scaling=self.scaling,
            scaling_factor=self.scaling_factor,
        )
        angles = position_ids[..., None].astype(dtype) * inverse_frequencies
        angles = _add_head_axes(angles=angles, x=x, position_ids=position_ids)
        cos = pt.cos(angles)
        sin = pt.sin(angles)

        first, second = _split_pairs(x=x, pairing=self.pairing, head_dim=head_dim)
        rotated = _join_pairs(
            x=x,
            first=first * cos - second * sin,
            second=second * cos + first * sin,
            pairing=self.pairing,
        )
        rotated.name = "rotary_embedding"

        return [rotated]


def rotary_embedding(
    x: pt.TensorLike,
    position_ids: pt.TensorLike,
    *,
    base: float = 10_000.0,
    pairing: Pairing = "half",
    scaling: Scaling = "none",
    scaling_factor: float = 1.0,
) -> TensorVariable:
    r"""
    Rotary position embedding (RoPE) applied to the trailing feature axis.

    Rotate each two-dimensional subspace of the feature axis by an angle proportional to the token's
    position:

    .. math::

        \begin{pmatrix} x'_a \\ x'_b \end{pmatrix} =
        \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\
                        \sin m\theta_i & \cos m\theta_i \end{pmatrix}
        \begin{pmatrix} x_a \\ x_b \end{pmatrix},
        \qquad \theta_i = \mathrm{base}^{-2i/d},

    where :math:`m` is the position and :math:`(a, b)` is the :math:`i`-th channel pair [1]_.
    Query-key dot products depend only on relative position.

    Apply to queries and keys before
    :func:`~pytensor_ml.layers.attention.scaled_dot_product_attention`, never to values.

    Parameters
    ----------
    x : TensorLike
        Tensor whose last axis is rotated, typically queries or keys of shape
        ``(..., n_head, seq, head_dim)``. ``head_dim`` must be even.
        For JAX with ``pairing="half"`` and for MLX, declare ``head_dim`` in the input's static shape.
    position_ids : TensorLike
        Token positions, shape ``(..., seq)``. Head axes are inserted so ``(seq,)`` and
        ``(batch, seq)`` both broadcast over ``(batch, n_head, seq, head_dim)``.
    base : float, optional
        Geometric base of the frequency ladder. Default 10000.0.
    pairing : str, optional
        Pair channels across halves (``"half"`` [3]_) or consecutively (``"adjacent"`` [4]_).
        Must match the trained weights. Default ``"half"``.
    scaling : str, optional
        Context-extension scheme: ``"none"`` (default), ``"linear"`` for position interpolation [2]_,
        or ``"ntk"`` for static NTK-aware scaling.
    scaling_factor : float, optional
        Context-extension factor, ignored when ``scaling="none"``. Default 1.0.

    Returns
    -------
    rotated : TensorVariable
        ``x`` with its last axis rotated, same shape and dtype.

    Examples
    --------
    Rotate a token at its absolute position in a decoded sequence:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.layers import rotary_embedding

        token = np.ones((1, 8), dtype="float32")
        rotated = rotary_embedding(token, position_ids=np.array([12])).eval()

    References
    ----------
    .. [1] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced
           Transformer with Rotary Position Embedding. arXiv:2104.09864. https://arxiv.org/abs/2104.09864.
    .. [2] Chen, S., Wong, S., Chen, L., & Tian, Y. (2023). Extending Context Window of Large Language
           Models via Positional Interpolation. arXiv:2306.15595. https://arxiv.org/abs/2306.15595.
    .. [3] https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py#L109-L113
    .. [4] https://github.com/meta-pytorch/torchtune/blob/v0.6.1/torchtune/modules/position_embeddings.py#L99-L113
    """
    _validate_options(pairing=pairing, scaling=scaling, scaling_factor=scaling_factor)

    x = pt.as_tensor(x)
    position_ids = pt.as_tensor(position_ids)

    rotated = RotaryEmbeddingLayer(
        name="RotaryEmbedding",
        base=base,
        pairing=pairing,
        scaling=scaling,
        scaling_factor=scaling_factor,
    )(x, position_ids)
    rotated.name = "rotary_embedding_output"

    return rotated


class RotaryEmbedding(VariadicLayer):
    r"""
    Rotary position embeddings as a configured layer.

    Share one frequency configuration between queries and keys [1]_. Call with ``(x, position_ids)``.
    The layer has no learned parameters.

    Parameters
    ----------
    name : str or None, optional
        Name prefix for the layer's output. Defaults to "RotaryEmbedding" when None.
    base : float, optional
        Geometric base of the frequency ladder. Default 10000.0.
    pairing : str, optional
        ``"half"`` (default) or ``"adjacent"``.
    scaling : str, optional
        ``"none"`` (default), ``"linear"``, or ``"ntk"``.
    scaling_factor : float, optional
        Extension factor for the scaled variants. Default 1.0.

    Examples
    --------
    Apply the same rotation to queries and keys before causal attention:

    .. code-block:: python

        import pytensor.tensor as pt

        from pytensor_ml.layers import Input, RotaryEmbedding, scaled_dot_product_attention

        queries = Input("queries", shape=(None, 4, None, 8))
        keys = Input("keys", shape=(None, 4, None, 8))
        values = Input("values", shape=(None, 4, None, 8))
        positions = pt.lvector("positions")
        rope = RotaryEmbedding("rope", pairing="half")
        output = scaled_dot_product_attention(
            rope(queries, positions), rope(keys, positions), values, is_causal=True
        )

    References
    ----------
    .. [1] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced
           Transformer with Rotary Position Embedding. arXiv:2104.09864. https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        base: float = 10_000.0,
        pairing: Pairing = "half",
        scaling: Scaling = "none",
        scaling_factor: float = 1.0,
    ):
        _validate_options(pairing=pairing, scaling=scaling, scaling_factor=scaling_factor)

        self.name = _resolve_layer_name(name, "RotaryEmbedding", "base")
        self.base = base
        self.pairing = pairing
        self.scaling = scaling
        self.scaling_factor = scaling_factor

    def __call__(self, *inputs: pt.TensorLike) -> TensorVariable:
        """Rotate ``(x, position_ids)`` using this layer's configuration."""
        x, position_ids = inputs
        rotated = rotary_embedding(
            x=x,
            position_ids=position_ids,
            base=self.base,
            pairing=self.pairing,
            scaling=self.scaling,
            scaling_factor=self.scaling_factor,
        )
        rotated.name = f"{self.name}_output"

        return rotated


__all__ = [
    "RotaryEmbedding",
    "rotary_embedding",
]
