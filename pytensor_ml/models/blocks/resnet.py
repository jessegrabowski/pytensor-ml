import pytensor.tensor as pt

from pytensor_ml.activations import Activation, Swish
from pytensor_ml.base import Layer, _resolve_layer_name
from pytensor_ml.layers import Conv2D, Dropout, GroupNorm, Linear


def _identity(x: pt.TensorVariable) -> pt.TensorVariable:
    return x


class ResnetBlock2D(Layer):
    r"""
    The pre-activation residual block a diffusion U-Net and its VAE are built from.

    Apply :class:`~pytensor_ml.layers.GroupNorm`, an activation and a 3x3 convolution twice, then add
    the input back:

    .. math::

        h &= \mathrm{conv}_1(\phi(\mathrm{norm}_1(x))) \\
        y &= \mathrm{conv}_2(\phi(\mathrm{norm}_2(h))) + s(x),

    where :math:`\phi` is the activation and :math:`s` is the identity, or a learned 1x1 convolution
    when the block changes the channel count. Normalizing before the convolution rather than after
    leaves the skip connection an unmodified path from input to output.

    A U-Net conditions each block on the diffusion timestep. Give ``temb_channels`` to add that
    projection, and pass the embedding to each call; the block adds it per channel between the two
    normalizations, broadcast over the spatial axes.

    Parameters
    ----------
    name : str, optional
        Name used as a prefix for the layer's parameters. Defaults to the class name.
    in_channels : int
        Size of the input's channel axis.
    out_channels : int, optional
        Size of the output's channel axis. Equal to ``in_channels`` when omitted, which is the case
        that needs no shortcut convolution.
    n_groups : int, optional
        Number of groups each normalization splits the channels into. Must divide both channel
        counts. Default is 32.
    epsilon : float, optional
        Constant added to each normalization's variance, which
        :class:`~pytensor_ml.layers.GroupNorm` does not inherit from here. Default is 1e-6.
    activation : Activation, optional
        Applied before each convolution, and to the timestep embedding. Swish when omitted.
    temb_channels : int, optional
        Width of the timestep embedding this block is conditioned on. No projection is built when
        omitted, and passing an embedding to a block without one raises.
    dropout : float, optional
        Dropout probability applied before the second convolution. Default is 0.0, which builds no
        dropout at all.
    conv_shortcut : bool, optional
        Whether the skip connection is a learned 1x1 convolution, which a block that changes the
        channel count needs for its addition to be defined. Inferred from whether the channel count
        changes when omitted.

    Examples
    --------
    Two blocks over a 64x64 feature map, the first widening it from 128 to 256 channels, which is
    what gives that one a shortcut convolution and the second none:

    .. code-block:: python

        from pytensor_ml.layers import Input, Sequential
        from pytensor_ml.models import ResnetBlock2D

        X = Input("X", shape=(None, 64, 64, 128))
        network = Sequential(
            ResnetBlock2D("block_0", in_channels=128, out_channels=256),
            ResnetBlock2D("block_1", in_channels=256),
        )

        activations = network(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        in_channels: int,
        out_channels: int | None = None,
        n_groups: int = 32,
        epsilon: float = 1e-6,
        activation: Activation | None = None,
        temb_channels: int | None = None,
        dropout: float = 0.0,
        conv_shortcut: bool | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "in_channels")
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.activation = Swish() if activation is None else activation

        self.norm1 = GroupNorm(
            f"{self.name}_norm1", n_groups=n_groups, n_in=in_channels, epsilon=epsilon
        )
        self.conv1 = Conv2D(
            f"{self.name}_conv1",
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="same",
        )
        self.norm2 = GroupNorm(
            f"{self.name}_norm2", n_groups=n_groups, n_in=self.out_channels, epsilon=epsilon
        )
        self.conv2 = Conv2D(
            f"{self.name}_conv2",
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="same",
        )

        self.time_emb_proj = (
            None
            if temb_channels is None
            else Linear(f"{self.name}_time_emb_proj", n_in=temb_channels, n_out=self.out_channels)
        )

        self.dropout = Dropout(f"{self.name}_dropout", p=dropout) if dropout > 0 else _identity

        if conv_shortcut is None:
            conv_shortcut = in_channels != self.out_channels
        elif not conv_shortcut and in_channels != self.out_channels:
            raise ValueError(
                f"{self.name} takes {in_channels} channels and returns {self.out_channels}, so "
                f"conv_shortcut=False leaves the residual addition undefined. Omit it, or pass "
                f"out_channels={in_channels}."
            )
        self.conv_shortcut = (
            Conv2D(
                f"{self.name}_conv_shortcut",
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
            )
            if conv_shortcut
            else None
        )

    def __call__(self, X: pt.TensorLike, temb: pt.TensorLike | None = None) -> pt.TensorVariable:
        X = pt.as_tensor(X)

        hidden = self.conv1(self.activation(self.norm1(X)))

        if self.time_emb_proj is not None:
            if temb is None:
                raise ValueError(
                    f"{self.name} was built with temb_channels, so it conditions on a timestep "
                    f"embedding that this call did not pass."
                )
            projected = self.time_emb_proj(self.activation(pt.as_tensor(temb)))
            hidden = hidden + projected[:, None, None, :]
        elif temb is not None:
            raise ValueError(
                f"{self.name} was built without temb_channels, so it has no projection to put this "
                f"timestep embedding through."
            )

        hidden = self.conv2(self.dropout(self.activation(self.norm2(hidden))))

        residual = X if self.conv_shortcut is None else self.conv_shortcut(X)
        output = residual + hidden

        output.name = f"{self.name}_output"
        return output


__all__ = ["ResnetBlock2D"]
