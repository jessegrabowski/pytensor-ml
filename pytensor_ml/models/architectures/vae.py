import pytensor.tensor as pt

from pytensor.graph.basic import Variable

from pytensor_ml.activations import Swish
from pytensor_ml.layers import Conv2D, GroupNorm, Input, Upsample2D, ZeroPad2D
from pytensor_ml.models.blocks import AttentionBlock2D, ResnetBlock2D
from pytensor_ml.models.loading.binding import bind_linear
from pytensor_ml.models.loading.keys import KeyMap, channels_last
from pytensor_ml.models.loading.registry import register_builder

_EPSILON = 1e-6

_ASSUMED_CONFIG = {
    "act_fn": "silu",
    "norm_type": "group",
    "mid_block_add_attention": True,
}


@register_builder("AutoencoderKL")
def build_autoencoder_kl(config: dict, keys: KeyMap) -> tuple[list[Variable], list[Variable]]:
    """
    Build a diffusion autoencoder and record where each parameter loads from.

    The two directions are independent graphs over one set of weights: ``mean`` and ``log_variance``
    descend only from ``image``, and ``sample`` only from ``latent``. Compile whichever you need over
    the input it uses rather than over both.

    The posterior is returned as its parameters. Drawing from it is the caller's business, since a
    pipeline usually wants the mean rather than a sample, and the ``scaling_factor`` the config
    carries belongs to whoever consumes the latent.

    Parameters
    ----------
    config : dict
        Parsed ``config.json`` of an ``AutoencoderKL``.
    keys : KeyMap
        Filled with the checkpoint key of every parameter built.

    Returns
    -------
    data_inputs : list of Variable
        The ``image`` and ``latent`` placeholders, channel-last.
    outputs : list of Variable
        The posterior's ``mean`` and ``log_variance``, at an eighth of the image's spatial extent,
        then the decoded ``sample`` at eight times the latent's.

    Examples
    --------
    Compile each direction over the input it actually reads:

    .. code-block:: python

        import pytensor

        from pytensor_ml import from_pretrained

        (image, latent), (mean, log_variance, sample) = from_pretrained("sdxl/vae", variant="fp16")

        encode = pytensor.function([image], mean)
        decode = pytensor.function([latent], sample)
    """
    _check_config(config)

    image, mean, log_variance = _build_encoder(config, keys)
    latent, sample = _build_decoder(config, keys)

    return [image, latent], [mean, log_variance, sample]


def _check_config(config: dict) -> None:
    """Refuse a config naming an autoencoder this builder does not describe."""
    for key, assumed in _ASSUMED_CONFIG.items():
        if config.get(key, assumed) != assumed:
            raise ValueError(
                f"AutoencoderKL's {key} is {config[key]!r}, which builds a different autoencoder "
                f"than the one implemented here."
            )

    unknown = sorted(set(config.get("up_block_types", ["UpDecoderBlock2D"])) - {"UpDecoderBlock2D"})
    if unknown:
        raise ValueError(
            f"AutoencoderKL's up_block_types names {unknown}, and only 'UpDecoderBlock2D' is built "
            f"here."
        )


def _build_encoder(config: dict, keys: KeyMap) -> tuple[Variable, Variable, Variable]:
    """The image-to-posterior direction: its input, and the two parameters it produces."""
    widths = config["block_out_channels"]
    n_groups = config.get("norm_num_groups", 32)
    latent_channels = config.get("latent_channels", 4)
    image_channels = config.get("in_channels", 3)

    image = Input("image", shape=(None, None, None, image_channels))

    conv_in = Conv2D(
        "encoder_conv_in",
        in_channels=image_channels,
        out_channels=widths[0],
        kernel_size=3,
        padding="same",
    )
    stages = _down_stages(widths, n_resnets=config.get("layers_per_block", 2), n_groups=n_groups)
    mid = _MidBlock(widths[-1], n_groups=n_groups, name="encoder_mid")
    norm_out = GroupNorm("encoder_norm_out", n_groups=n_groups, n_in=widths[-1], epsilon=_EPSILON)
    conv_out = Conv2D(
        "encoder_conv_out",
        in_channels=widths[-1],
        out_channels=2 * latent_channels,
        kernel_size=3,
        padding="same",
    )
    quant_conv = Conv2D(
        "quant_conv",
        in_channels=2 * latent_channels,
        out_channels=2 * latent_channels,
        kernel_size=1,
    )

    _bind_conv(keys, quant_conv, "quant_conv")
    with keys.scope("encoder"):
        _bind_conv(keys, conv_in, "conv_in")
        _bind_stages(keys, stages, "down_blocks", "downsamplers")
        _bind_mid_block(keys, mid)
        _bind_group_norm(keys, norm_out, "conv_norm_out")
        _bind_conv(keys, conv_out, "conv_out")

    hidden = conv_in(image)
    for stage in stages:
        hidden = stage(hidden)
    moments = quant_conv(conv_out(Swish()(norm_out(mid(hidden)))))

    mean, log_variance = pt.split(moments, [latent_channels, latent_channels], n_splits=2, axis=-1)
    mean.name, log_variance.name = "mean", "log_variance"
    return image, mean, log_variance


def _build_decoder(config: dict, keys: KeyMap) -> tuple[Variable, Variable]:
    """The latent-to-image direction: its input, and the image it produces."""
    widths = config["block_out_channels"]
    n_groups = config.get("norm_num_groups", 32)
    latent_channels = config.get("latent_channels", 4)

    latent = Input("latent", shape=(None, None, None, latent_channels))

    post_quant_conv = Conv2D(
        "post_quant_conv", in_channels=latent_channels, out_channels=latent_channels, kernel_size=1
    )
    conv_in = Conv2D(
        "decoder_conv_in",
        in_channels=latent_channels,
        out_channels=widths[-1],
        kernel_size=3,
        padding="same",
    )
    mid = _MidBlock(widths[-1], n_groups=n_groups, name="decoder_mid")
    stages = _up_stages(widths, n_resnets=config.get("layers_per_block", 2) + 1, n_groups=n_groups)
    norm_out = GroupNorm("decoder_norm_out", n_groups=n_groups, n_in=widths[0], epsilon=_EPSILON)
    conv_out = Conv2D(
        "decoder_conv_out",
        in_channels=widths[0],
        out_channels=config.get("out_channels", 3),
        kernel_size=3,
        padding="same",
    )

    _bind_conv(keys, post_quant_conv, "post_quant_conv")
    with keys.scope("decoder"):
        _bind_conv(keys, conv_in, "conv_in")
        _bind_mid_block(keys, mid)
        _bind_stages(keys, stages, "up_blocks", "upsamplers")
        _bind_group_norm(keys, norm_out, "conv_norm_out")
        _bind_conv(keys, conv_out, "conv_out")

    hidden = mid(conv_in(post_quant_conv(latent)))
    for stage in stages:
        hidden = stage(hidden)
    sample = conv_out(Swish()(norm_out(hidden)))

    sample.name = "sample"
    return latent, sample


class _MidBlock:
    """The pair of residual blocks each half puts an attention block between."""

    def __init__(self, width: int, *, n_groups: int, name: str):
        self.resnets = [
            ResnetBlock2D(
                f"{name}_resnet_{i}", in_channels=width, n_groups=n_groups, epsilon=_EPSILON
            )
            for i in range(2)
        ]
        self.attention = AttentionBlock2D(
            f"{name}_attn", channels=width, n_groups=n_groups, epsilon=_EPSILON
        )

    def __call__(self, X: pt.TensorVariable) -> pt.TensorVariable:
        return self.resnets[1](self.attention(self.resnets[0](X)))


class _Stage:
    """One resolution's residual blocks, and the resampling that follows them where there is any.

    The stage already at its target resolution resamples nothing, so it owns no convolution and
    binds none, which is why ``resample`` is absent rather than an identity.
    """

    def __init__(self, resnets: list[ResnetBlock2D], resample: "_Downsample | _Upsample | None"):
        self.resnets = resnets
        self.resample = resample

    def __call__(self, X: pt.TensorVariable) -> pt.TensorVariable:
        for resnet in self.resnets:
            X = resnet(X)
        return X if self.resample is None else self.resample(X)


class _Downsample:
    """A stride-2 convolution over a map padded on its bottom and right only.

    Padding symmetrically instead shifts the whole map half a pixel, with nothing to raise.
    """

    def __init__(self, name: str, width: int):
        self.pad = ZeroPad2D(f"{name}_pad", padding=((0, 1), (0, 1)))
        self.conv = Conv2D(
            f"{name}_conv", in_channels=width, out_channels=width, kernel_size=3, stride=2
        )

    def __call__(self, X: pt.TensorVariable) -> pt.TensorVariable:
        return self.conv(self.pad(X))


class _Upsample:
    """A nearest-neighbor doubling, then a convolution, in the order diffusers applies them."""

    def __init__(self, name: str, width: int):
        self.upsample = Upsample2D(f"{name}_upsample", scale_factor=2, mode="nearest")
        self.conv = Conv2D(
            f"{name}_conv",
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            padding="same",
        )

    def __call__(self, X: pt.TensorVariable) -> pt.TensorVariable:
        return self.conv(self.upsample(X))


def _stages(
    widths: list[int], *, prefix: str, n_resnets: int, n_groups: int, resample
) -> list[_Stage]:
    """One stage per width, the last resampling nothing since it is already at its own resolution."""
    stages, in_width = [], widths[0]
    for index, out_width in enumerate(widths):
        name = f"{prefix}_{index}"
        resnets = [
            ResnetBlock2D(
                f"{name}_resnet_{i}",
                in_channels=in_width if i == 0 else out_width,
                out_channels=out_width,
                n_groups=n_groups,
                epsilon=_EPSILON,
            )
            for i in range(n_resnets)
        ]
        is_last = index == len(widths) - 1
        stages.append(_Stage(resnets, None if is_last else resample(name, out_width)))
        in_width = out_width
    return stages


def _down_stages(widths: list[int], *, n_resnets: int, n_groups: int) -> list[_Stage]:
    """Finest first, each widening the channel count the config asks it to."""
    return _stages(
        widths, prefix="encoder_down", n_resnets=n_resnets, n_groups=n_groups, resample=_Downsample
    )


def _up_stages(widths: list[int], *, n_resnets: int, n_groups: int) -> list[_Stage]:
    """Coarsest first, each narrowing the channel count the config asks it to."""
    return _stages(
        list(reversed(widths)),
        prefix="decoder_up",
        n_resnets=n_resnets,
        n_groups=n_groups,
        resample=_Upsample,
    )


def _bind_stages(keys: KeyMap, stages: list[_Stage], scope: str, resampler_scope: str) -> None:
    with keys.scope(scope):
        for index, stage in enumerate(stages):
            with keys.scope(str(index)):
                for i, resnet in enumerate(stage.resnets):
                    with keys.scope("resnets", str(i)):
                        _bind_resnet_block(keys, resnet)
                if stage.resample is not None:
                    _bind_conv(keys, stage.resample.conv, resampler_scope, "0", "conv")


def _bind_mid_block(keys: KeyMap, mid: _MidBlock) -> None:
    with keys.scope("mid_block"):
        with keys.scope("attentions", "0"):
            _bind_attention_block(keys, mid.attention)
        for i, resnet in enumerate(mid.resnets):
            with keys.scope("resnets", str(i)):
                _bind_resnet_block(keys, resnet)


def _bind_conv(keys: KeyMap, conv: Conv2D, *parts: str) -> None:
    keys.bind(conv.W, *parts, "weight", transform=channels_last)
    keys.bind(conv.b, *parts, "bias")


def _bind_group_norm(keys: KeyMap, norm: GroupNorm, *parts: str) -> None:
    assert norm.scale is not None and norm.loc is not None
    keys.bind(norm.scale, *parts, "weight")
    keys.bind(norm.loc, *parts, "bias")


def _bind_resnet_block(keys: KeyMap, block: ResnetBlock2D) -> None:
    _bind_group_norm(keys, block.norm1, "norm1")
    _bind_conv(keys, block.conv1, "conv1")
    _bind_group_norm(keys, block.norm2, "norm2")
    _bind_conv(keys, block.conv2, "conv2")
    if block.conv_shortcut is not None:
        _bind_conv(keys, block.conv_shortcut, "conv_shortcut")


def _bind_attention_block(keys: KeyMap, block: AttentionBlock2D) -> None:
    _bind_group_norm(keys, block.norm, "group_norm")
    for name, projection in [
        ("to_q", block.attn.q_proj),
        ("to_k", block.attn.k_proj),
        ("to_v", block.attn.v_proj),
    ]:
        bind_linear(keys, projection, name, transform=channels_last)
    bind_linear(keys, block.attn.out_proj, "to_out", "0", transform=channels_last)
