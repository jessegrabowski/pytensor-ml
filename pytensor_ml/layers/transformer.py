import pytensor.tensor as pt

from pytensor_ml.activations import GELU, Activation
from pytensor_ml.base import Layer, _resolve_layer_name
from pytensor_ml.layers.attention import MultiheadAttention
from pytensor_ml.layers.dropout import Dropout
from pytensor_ml.layers.linear import Linear
from pytensor_ml.layers.norm import LayerNorm
from pytensor_ml.state import Initializer


def _identity(x: pt.TensorVariable) -> pt.TensorVariable:
    return x


class FeedForward(Layer):
    r"""
    Position-wise feed-forward network.

    Apply two linear layers with an activation between them, expanding to a wider hidden dimension and
    projecting back:

    .. math::

        \mathrm{FFN}(x) = \phi(x W_1 + b_1) W_2 + b_2,

    where :math:`\phi` is the activation. This is the per-token MLP used in transformer blocks, but it is
    a standalone layer usable anywhere a widening-then-narrowing MLP is wanted.

    Gated, the first projection is twice as wide and its halves multiply rather than one passing
    through the activation:

    .. math::

        \mathrm{FFN}(x) = \left(a \odot \phi(g)\right) W_2 + b_2, \qquad
        \left[a, g\right] = x W_1 + b_1.

    That is GEGLU with :class:`~pytensor_ml.activations.GELU` and SwiGLU with
    :class:`~pytensor_ml.activations.Swish`.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "FeedForward" when None.
    d_model : int
        Input and output dimension.
    hidden_dim : int, optional
        Width of the hidden layer. Overrides ``mlp_ratio`` when given; otherwise computed as
        ``mlp_ratio * d_model``.
    mlp_ratio : int, optional
        Hidden-to-model dimension ratio used when ``hidden_dim`` is not given. Default is 4.
    activation : Activation, optional
        Activation applied to the hidden layer. Default is :class:`GELU`.
    gated : bool, optional
        Widen the first projection to :math:`2 d_{hidden}` and gate one half of it by the activation
        applied to the other. The widened projection stays one weight. Default is False.
    bias : bool, optional
        Include bias terms in both linear layers. Default is True.
    fc_out_initializer : Initializer, optional
        How the second layer's weight is drawn. The hidden layer is unaffected. Xavier normal when omitted,
        as for any other weight.

    Examples
    --------
    The position-wise MLP of a transformer block: widen by ``mlp_ratio``, apply a nonlinearity, project
    back. Pass ``hidden_dim`` to set the width directly instead of as a multiple:

    .. code-block:: python

        from pytensor_ml.activations import GELU
        from pytensor_ml.layers import FeedForward, Input

        X = Input("X", shape=(None, 128, 256))
        activations = FeedForward("ff", d_model=256, mlp_ratio=4, activation=GELU())(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        d_model: int,
        hidden_dim: int | None = None,
        mlp_ratio: int = 4,
        activation: Activation | None = None,
        gated: bool = False,
        bias: bool = True,
        fc_out_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "d_model")
        self.d_model = d_model
        self.hidden_dim = hidden_dim if hidden_dim is not None else mlp_ratio * d_model
        self.activation = activation if activation is not None else GELU()
        self.gated = gated

        self.fc_in = Linear(
            f"{self.name}_fc_in",
            n_in=d_model,
            n_out=2 * self.hidden_dim if gated else self.hidden_dim,
            bias=bias,
        )
        self.fc_out = Linear(
            f"{self.name}_fc_out",
            n_in=self.hidden_dim,
            n_out=d_model,
            bias=bias,
            weight_initializer=fc_out_initializer,
        )

    def __call__(self, x: pt.TensorLike) -> pt.TensorVariable:
        x = pt.as_tensor(x)
        projected = self.fc_in(x)
        if self.gated:
            value, gate = pt.split(projected, [self.hidden_dim] * 2, axis=-1)
            hidden = value * self.activation(gate)
        else:
            hidden = self.activation(projected)

        out = self.fc_out(hidden)
        out.name = f"{self.name}_output"
        return out


class TransformerBlock(Layer):
    r"""
    A transformer block: multi-head self-attention and a feed-forward network, each wrapped in a
    residual connection around a layer-normalized sublayer.

    With ``norm_first=True`` (pre-norm, the modern default) the block computes

    .. math::

        x &= x + \mathrm{Attn}(\mathrm{LN}_1(x)) \\
        x &= x + \mathrm{FFN}(\mathrm{LN}_2(x)),

    and with ``norm_first=False`` (post-norm, the original formulation) the normalization is applied to
    the residual sum instead. Set ``is_causal=True`` for a decoder block, leave it False for an encoder
    block; ``n_kv_head`` enables grouped-query attention.

    Parameters
    ----------
    name : str or None
        Name prefix for the block's parameters. Defaults to "TransformerBlock" when None.
    d_model : int
        Model dimension of the input and output.
    n_head : int
        Number of attention heads. Must divide ``d_model`` evenly.
    mlp_ratio : int, optional
        Feed-forward hidden-to-model dimension ratio. Default is 4.
    activation : Activation, optional
        Activation used in the feed-forward network. Default is :class:`GELU`.
    norm_first : bool, optional
        Normalize each sublayer's input (pre-norm) rather than its residual sum (post-norm). Default is
        True.
    dropout : float, optional
        Dropout probability applied to each sublayer's output before the residual add. Dropout is omitted
        from the graph entirely when 0. Default is 0.0.
    bias : bool, optional
        Include bias terms in the attention and feed-forward projections. Default is True.
    is_causal : bool, optional
        Apply a causal mask in the self-attention. Default is False.
    n_kv_head : int, optional
        Number of key/value heads for grouped-query attention. Defaults to ``n_head``.
    epsilon : float, optional
        Constant added to the layer-norm variance for numerical stability. Default is 1e-5.
    residual_initializer : Initializer, optional
        How the two projections that write back into the residual stream are drawn -- attention's output
        projection and the feed-forward's second layer. Their siblings keep the default, which is what
        GPT-style initialization asks for: a residual stream accumulates one contribution per block, so those
        projections are scaled by :math:`1/\sqrt{2 n_\text{layer}}` while the rest are not. The depth is not
        known here, so the scaling belongs in the initializer the caller passes.

    Examples
    --------
    Attention and feed-forward with their residual connections and norms, which is the unit a transformer
    repeats. ``norm_first`` puts the norm inside the residual branch, the pre-norm arrangement that trains
    stably without a warmup:

    .. code-block:: python

        from pytensor_ml.layers import Input, Sequential, TransformerBlock

        X = Input("X", shape=(None, 128, 256))
        network = Sequential(
            TransformerBlock("block1", d_model=256, n_head=8, norm_first=True),
            TransformerBlock("block2", d_model=256, n_head=8, norm_first=True),
        )

        activations = network(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        d_model: int,
        n_head: int,
        mlp_ratio: int = 4,
        activation: Activation | None = None,
        norm_first: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = False,
        n_kv_head: int | None = None,
        fused_qkv: bool = False,
        epsilon: float = 1e-5,
        residual_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "d_model")
        self.d_model = d_model
        self.norm_first = norm_first

        self.norm1 = LayerNorm(f"{self.name}_norm1", n_in=d_model, epsilon=epsilon)
        self.norm2 = LayerNorm(f"{self.name}_norm2", n_in=d_model, epsilon=epsilon)
        self.attn = MultiheadAttention(
            f"{self.name}_attn",
            n_embd=d_model,
            n_head=n_head,
            n_kv_head=n_kv_head,
            fused_qkv=fused_qkv,
            bias=bias,
            is_causal=is_causal,
            out_proj_initializer=residual_initializer,
        )
        self.ff = FeedForward(
            f"{self.name}_ff",
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            activation=activation,
            bias=bias,
            fc_out_initializer=residual_initializer,
        )
        self.attn_dropout = (
            Dropout(f"{self.name}_attn_dropout", p=dropout) if dropout > 0 else _identity
        )
        self.ff_dropout = (
            Dropout(f"{self.name}_ff_dropout", p=dropout) if dropout > 0 else _identity
        )

    def __call__(self, x: pt.TensorLike, mask: pt.TensorLike | None = None) -> pt.TensorVariable:
        x = pt.as_tensor(x)
        if mask is not None:
            mask = pt.as_tensor(mask)

        if self.norm_first:
            x = x + self.attn_dropout(self.attn(self.norm1(x), mask=mask))
            x = x + self.ff_dropout(self.ff(self.norm2(x)))
        else:
            x = self.norm1(x + self.attn_dropout(self.attn(x, mask=mask)))
            x = self.norm2(x + self.ff_dropout(self.ff(x)))

        x.name = f"{self.name}_output"  # type: ignore[union-attr]
        return x  # type: ignore[return-value]


__all__ = [
    "FeedForward",
    "TransformerBlock",
]
