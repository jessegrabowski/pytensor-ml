from pytensor_ml.layers import LayerNorm, Linear
from pytensor_ml.models.keys import KeyMap, Transform


def bind_layer_norm(keys: KeyMap, norm: LayerNorm, *parts: str) -> None:
    """Bind an affine norm's scale and shift to the ``weight`` and ``bias`` HuggingFace stores."""
    # The layer types these as optional because a norm without an affine transform owns neither.
    assert norm.scale is not None and norm.loc is not None
    keys.bind(norm.scale, *parts, "weight")
    keys.bind(norm.loc, *parts, "bias")


def bind_linear(
    keys: KeyMap, linear: Linear | None, *parts: str, transform: Transform | None = None
) -> None:
    """
    Bind a dense layer's weight, and its bias when it has one.

    Pass :func:`~pytensor_ml.models.keys.channels_last` for a checkpoint written with ``nn.Linear``,
    which stores ``(out, in)``. Omit it for one written with HuggingFace's ``Conv1D``, which already
    stores ``(in, out)``.
    """
    # The layer types the attention projections as optional because a fused attention owns one weight
    # in their place.
    assert linear is not None
    keys.bind(linear.W, *parts, "weight", transform=transform)
    if linear.bias:
        keys.bind(linear.b, *parts, "bias")
