import pytensor.tensor as pt

from pytensor_ml.base import Layer, _resolve_layer_name
from pytensor_ml.layers import GroupNorm, MultiheadAttention


class AttentionBlock2D(Layer):
    r"""
    Self-attention over a feature map's pixels, as a diffusion U-Net and its VAE use at low resolution.

    Normalize, flatten the spatial axes into a sequence of :math:`H \times W` pixels, attend over it,
    then restore the map's shape and add the input back:

    .. math::

        y = x + \mathrm{attend}(\mathrm{norm}(x)).

    Every pixel attends to every other, so the cost is quadratic in :math:`H \times W` and this belongs
    where the map is already small.

    Parameters
    ----------
    name : str, optional
        Name used as a prefix for the layer's parameters. Defaults to the class name.
    channels : int
        Size of the channel axis, which is also the attention's embedding width.
    n_head : int, optional
        Number of attention heads. Default is 1, which is what an autoencoder's mid block uses.
    n_groups : int, optional
        Number of groups the normalization splits the channels into. Default is 32.
    epsilon : float, optional
        Constant added to the normalization's variance. Default is 1e-6.

    Examples
    --------
    The block an autoencoder puts between its two mid-block residual blocks, at one head over all
    512 channels:

    .. code-block:: python

        from pytensor_ml.layers import Input
        from pytensor_ml.models import AttentionBlock2D

        X = Input("X", shape=(None, 32, 32, 512))
        activations = AttentionBlock2D("mid_attn", channels=512)(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        channels: int,
        n_head: int = 1,
        n_groups: int = 32,
        epsilon: float = 1e-6,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "channels")
        self.channels = channels

        self.norm = GroupNorm(
            f"{self.name}_norm", n_groups=n_groups, n_in=channels, epsilon=epsilon
        )
        self.attn = MultiheadAttention(
            f"{self.name}_attn", n_embd=channels, n_head=n_head, bias=True
        )

    def __call__(self, X: pt.TensorLike) -> pt.TensorVariable:
        X = pt.as_tensor(X)

        pixels = pt.join_dims(X, start_axis=1, n_axes=X.ndim - 2)
        attended = self.attn(self.norm(pixels))
        output = X + pt.split_dims(attended, shape=X.shape[1:-1], axis=1)

        output.name = f"{self.name}_output"
        return output


__all__ = ["AttentionBlock2D"]
