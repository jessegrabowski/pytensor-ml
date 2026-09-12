import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import Layer, UnaryLayerOp, _resolve_layer_name
from pytensor_ml.layers.linear import Linear
from pytensor_ml.state import Initializer


class AttentionLayer(UnaryLayerOp):
    __props__ = ("is_causal", "scale")

    def build_inner_graph(self, q, k, v, mask=None):
        return [_sdpa_graph(q, k, v, mask, self.is_causal, self.scale)]


def _repeat_kv(x: TensorVariable, n_head: int | None, n_kv_head: int | None) -> TensorVariable:
    """Broadcast ``n_kv_head`` key/value heads up to ``n_head`` query heads (grouped-query attention).

    Each key/value head serves a contiguous group of ``n_head // n_kv_head`` query heads, matching the
    ``repeat_interleave`` convention used by every grouped-query model. When the head counts are equal this
    is a no-op and adds nothing to the graph -- the dense self-attention path stays clean.
    """
    if n_head is None or n_kv_head is None:
        raise ValueError("Attention requires statically known head counts on q and k.")
    if n_head == n_kv_head:
        return x
    return pt.repeat(x, n_head // n_kv_head, axis=-3)


def _sdpa_graph(
    q: TensorVariable,
    k: TensorVariable,
    v: TensorVariable,
    mask: TensorVariable | None,
    is_causal: bool,
    scale: float | None,
) -> TensorVariable:
    k = _repeat_kv(k, q.type.shape[-3], k.type.shape[-3])
    v = _repeat_kv(v, q.type.shape[-3], v.type.shape[-3])

    if scale is None:
        scale_t = 1.0 / pt.sqrt(q.shape[-1].astype(config.floatX))
    else:
        scale_t = pt.as_tensor(scale, dtype=config.floatX)

    scores = (q @ k.swapaxes(-1, -2)) * scale_t

    if is_causal:
        sq, sk = q.shape[-2], k.shape[-2]
        # Position i may attend to j <= i, aligned bottom-right so a partial query block (the decoding
        # case) sees the whole prefix. Reduces to a plain lower triangle when sq == sk.
        q_idx = pt.arange(sq)[:, None]
        k_idx = pt.arange(sk)[None, :]
        causal = pt.where(k_idx <= q_idx + (sk - sq), 0.0, -np.inf).astype(config.floatX)
        scores = scores + causal

    if mask is not None:
        scores = scores + mask

    return pt.special.softmax(scores, axis=-1) @ v


def scaled_dot_product_attention(
    q: TensorVariable,
    k: TensorVariable,
    v: TensorVariable,
    *,
    mask: TensorVariable | None = None,
    is_causal: bool = False,
    scale: float | None = None,
) -> TensorVariable:
    r"""
    Scaled dot-product attention over the trailing ``(head, sequence, feature)`` axes.

    Compute

    .. math::

        \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}} + M\right) V,

    where the softmax is taken over the key axis and :math:`M` is the combined causal and/or additive
    mask. This is the position-agnostic attention kernel: rotary or other positional schemes act on
    ``q`` and ``k`` before they reach this function.

    The query and key/value head counts may differ (grouped-query attention): key/value heads are
    broadcast up to the query head count. The query/key feature size and the value feature size are
    independent -- only the query and key feature sizes must match.

    Parameters
    ----------
    q : TensorVariable
        Queries, shape ``(..., n_head, q_len, qk_dim)``.
    k : TensorVariable
        Keys, shape ``(..., n_kv_head, kv_len, qk_dim)``. ``n_head`` must be a multiple of
        ``n_kv_head``.
    v : TensorVariable
        Values, shape ``(..., n_kv_head, kv_len, v_dim)``.
    mask : TensorVariable, optional
        Additive mask broadcast onto the attention scores of shape ``(..., n_head, q_len, kv_len)``,
        e.g. ``0`` for attended positions and a large negative value for masked ones. Combined with the
        causal mask when both are requested. Default is None.
    is_causal : bool, optional
        Apply a causal mask so each position attends only to itself and earlier positions. Default is
        False.
    scale : float, optional
        Softmax temperature applied to the scores. Defaults to :math:`1/\sqrt{d_k}` when None.

    Returns
    -------
    output : TensorVariable
        Attention output, shape ``(..., n_head, q_len, v_dim)``.

    Examples
    --------
    The bare attention kernel, for building an attention variant of your own. It takes heads as an explicit
    axis -- ``(batch, n_head, time, head_dim)`` -- and knows nothing about positions, so rotary or other
    positional schemes are applied to ``q`` and ``k`` before the call:

    .. code-block:: python

        import pytensor.tensor as pt

        from pytensor_ml.layers import scaled_dot_product_attention

        q = pt.tensor("q", shape=(None, 8, 128, 32))
        k = pt.tensor("k", shape=(None, 8, 128, 32))
        v = pt.tensor("v", shape=(None, 8, 128, 32))

        attended = scaled_dot_product_attention(q, k, v, is_causal=True)
    """
    q, k, v = (pt.as_tensor(t).copy() for t in (q, k, v))
    inputs = [q, k, v]

    if mask is not None:
        mask = pt.as_tensor(mask)
        inputs.append(mask)

    op = AttentionLayer(
        name="ScaledDotProductAttention",
        is_causal=is_causal,
        scale=scale,
    )
    result = op(*inputs)
    result.name = "attention_output"
    return result


class MultiheadAttention(Layer):
    r"""
    Multi-head attention, over one sequence or between two.

    Project to per-head queries, keys, and values, apply :func:`scaled_dot_product_attention`, then
    project the concatenated heads back to the model dimension. Supports grouped-query attention
    through ``n_kv_head``. Keys and values come from the query input unless :meth:`__call__` is given
    a second one, which is cross-attention.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "MultiheadAttention" when None.
    n_embd : int
        Model dimension of the input and output.
    n_head : int
        Number of query heads. Must divide ``n_embd`` evenly.
    n_kv_head : int, optional
        Number of key/value heads, for grouped-query attention. Must divide ``n_head`` evenly. Defaults
        to ``n_head`` (standard multi-head attention).
    kv_dim : int, optional
        Model dimension of the key/value input, when cross-attending to a source of a different width.
        Defaults to ``n_embd``.
    fused_qkv : bool, optional
        Project queries, keys and values with one weight and split the result, instead of three
        separate projections. This is the layout GPT-2 and its descendants store, where ``c_attn`` is
        a single tensor. Incompatible with cross-attention, which projects keys and values from a
        different input. Default is False.
    bias : bool, optional
        Include bias terms in the projections. Default is True.
    is_causal : bool, optional
        Apply a causal mask in the attention. Cannot be combined with cross-attention. Default is
        False.
    out_proj_initializer : Initializer, optional
        How the output projection's weight is drawn. The three input projections are unaffected, which is
        what a scaling applied only to the projection writing back into a residual stream needs. Xavier
        normal when omitted, as for any other weight.

    Examples
    --------
    Attend over a sequence with several heads at once, taking ``(batch, time, n_embd)``. Set
    ``n_kv_head`` below ``n_head`` for grouped-query attention, which shrinks the key/value cache that
    dominates memory at inference:

    .. code-block:: python

        from pytensor_ml.layers import Input, MultiheadAttention

        X = Input("X", shape=(None, 128, 256))
        attended = MultiheadAttention("attn", n_embd=256, n_head=8, n_kv_head=2)(X)

    Cross-attend instead by passing a second input for the keys and values, at its own width:

    .. code-block:: python

        from pytensor_ml.layers import Input, MultiheadAttention

        X = Input("X", shape=(None, 128, 256))
        context = Input("context", shape=(None, 77, 512))
        attended = MultiheadAttention("attn", n_embd=256, n_head=8, kv_dim=512)(X, context)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        kv_dim: int | None = None,
        fused_qkv: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        out_proj_initializer: Initializer | None = None,
    ):
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        n_kv_head = n_kv_head if n_kv_head is not None else n_head
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")

        self.name = _resolve_layer_name(name, type(self).__name__, "n_embd")
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.kv_dim = kv_dim if kv_dim is not None else n_embd
        self.is_causal = is_causal
        self.fused_qkv = fused_qkv

        if fused_qkv and self.kv_dim != n_embd:
            raise ValueError(
                f"{self.name} cannot project a {self.kv_dim}-wide key/value input through a weight "
                f"shared with {n_embd}-wide queries, so fused_qkv and kv_dim are mutually exclusive."
            )

        self.q_width = n_head * self.head_dim
        self.kv_width = n_kv_head * self.head_dim

        self.qkv_proj: Linear | None = None
        self.q_proj: Linear | None = None
        self.k_proj: Linear | None = None
        self.v_proj: Linear | None = None
        if fused_qkv:
            self.qkv_proj = Linear(
                f"{self.name}_qkv_proj",
                n_in=n_embd,
                n_out=self.q_width + 2 * self.kv_width,
                bias=bias,
            )
        else:
            self.q_proj = Linear(f"{self.name}_q_proj", n_in=n_embd, n_out=self.q_width, bias=bias)
            self.k_proj = Linear(
                f"{self.name}_k_proj", n_in=self.kv_dim, n_out=self.kv_width, bias=bias
            )
            self.v_proj = Linear(
                f"{self.name}_v_proj", n_in=self.kv_dim, n_out=self.kv_width, bias=bias
            )
        self.out_proj = Linear(
            f"{self.name}_out_proj",
            n_in=n_head * self.head_dim,
            n_out=n_embd,
            bias=bias,
            weight_initializer=out_proj_initializer,
        )

    def _split_heads(self, x: pt.TensorVariable, n_head: int) -> pt.TensorVariable:
        # (..., seq, n_head * head_dim) -> (..., n_head, seq, head_dim). split_dims keeps the static
        # shape a plain reshape would erase, which lets the numba backend vectorize attention (e.g. under
        # vectorize_graph) instead of falling back to object mode.
        return pt.split_dims(x, shape=(n_head, self.head_dim), axis=-1).swapaxes(-3, -2)

    def __call__(
        self,
        x: pt.TensorLike,
        kv: pt.TensorLike | None = None,
        mask: pt.TensorLike | None = None,
    ) -> pt.TensorVariable:
        """
        Attend over ``x``, of shape ``(batch, seq, n_embd)``.

        Parameters
        ----------
        x : TensorLike
            Input the queries are projected from.
        kv : TensorLike, optional
            Input the keys and values are projected from, of width ``kv_dim`` and any sequence
            length. Defaults to ``x``, which is self-attention.
        mask : TensorLike, optional
            Additive mask broadcast over the attention scores.

        Returns
        -------
        attended : TensorVariable
            Shape ``(batch, seq, n_embd)``.
        """
        if kv is not None and self.fused_qkv:
            raise ValueError(
                f"{self.name} projects keys and values from the same weight as queries, so they "
                f"cannot come from a second input; fused_qkv and kv are mutually exclusive."
            )
        if kv is not None and self.is_causal:
            raise ValueError(
                f"{self.name} cannot mask against earlier positions of a sequence it is not "
                f"attending over, so is_causal and kv are mutually exclusive."
            )

        x = pt.as_tensor(x)
        kv = x if kv is None else pt.as_tensor(kv)

        if self.qkv_proj is not None:
            widths = [self.q_width, self.kv_width, self.kv_width]
            queries, key_states, value_states = pt.split(self.qkv_proj(x), widths, axis=-1)
        else:
            assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None
            queries, key_states, value_states = self.q_proj(x), self.k_proj(kv), self.v_proj(kv)

        q = self._split_heads(queries, self.n_head)
        k = self._split_heads(key_states, self.n_kv_head)
        v = self._split_heads(value_states, self.n_kv_head)

        if mask is not None:
            mask = pt.as_tensor(mask)
        attn = scaled_dot_product_attention(q, k, v, mask=mask, is_causal=self.is_causal)

        # (..., n_head, seq, head_dim) -> (..., seq, n_head * head_dim)
        attn = pt.join_dims(attn.swapaxes(-3, -2), start_axis=-2, n_axes=2)

        out = self.out_proj(attn)
        out.name = f"{self.name}_output"
        return out


class CausalSelfAttention(MultiheadAttention):
    r"""
    Causal multi-head self-attention.

    A :class:`MultiheadAttention` with ``is_causal=True``: each position attends only to itself and
    earlier positions. This is the attention used in GPT-style decoder blocks.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "CausalSelfAttention" when None.
    n_embd : int
        Model dimension of the input and output.
    n_head : int
        Number of query heads. Must divide ``n_embd`` evenly.
    n_kv_head : int, optional
        Number of key/value heads, for grouped-query attention. Defaults to ``n_head``.
    bias : bool, optional
        Include bias terms in the projections. Default is True.
    out_proj_initializer : Initializer, optional
        How the output projection's weight is drawn. See :class:`MultiheadAttention`.

    Examples
    --------
    :class:`MultiheadAttention` with the causal mask always on, so no position sees a later one. This is
    the decoder-side layer a language model stacks:

    .. code-block:: python

        from pytensor_ml.layers import CausalSelfAttention, Input

        X = Input("X", shape=(None, 128, 256))
        attended = CausalSelfAttention("attn", n_embd=256, n_head=8)(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_embd: int,
        n_head: int,
        n_kv_head: int | None = None,
        bias: bool = True,
        out_proj_initializer: Initializer | None = None,
    ):
        super().__init__(
            name,
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            bias=bias,
            is_causal=True,
            out_proj_initializer=out_proj_initializer,
        )


__all__ = [
    "CausalSelfAttention",
    "MultiheadAttention",
    "scaled_dot_product_attention",
]
