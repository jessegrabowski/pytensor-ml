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
    for key, assumed in _ASSUMED_CONFIG.items():
        if config.get(key, assumed) != assumed:
            raise ValueError(
                f"AutoencoderKL's {key} is {config[key]!r}, which builds a different decoder than "
                f"the one implemented here."
            )

    up_blocks = config.get("up_block_types", ["UpDecoderBlock2D"])
    unknown = sorted(set(up_blocks) - {"UpDecoderBlock2D"})
    if unknown:
        raise ValueError(
            f"AutoencoderKL's up_block_types names {unknown}, and only 'UpDecoderBlock2D' is built "
            f"here."
        )

    widths = config["block_out_channels"]
    n_groups = config.get("norm_num_groups", 32)
    n_resnets = config.get("layers_per_block", 2) + 1
    latent_channels = config.get("latent_channels", 4)

    latent = Input("latent", shape=(None, None, None, latent_channels))
    image = Input("image", shape=(None, None, None, config.get("in_channels", 3)))
    activation = Swish()

    encoder_conv_in = Conv2D(
        "encoder_conv_in",
        in_channels=config.get("in_channels", 3),
        out_channels=widths[0],
        kernel_size=3,
        padding="same",
    )
    down_stages = _down_stages(
        widths, n_resnets=config.get("layers_per_block", 2), n_groups=n_groups
    )
    encoder_mid = _MidBlock(widths[-1], n_groups=n_groups, name="encoder_mid")
    encoder_norm_out = GroupNorm(
        "encoder_norm_out", n_groups=n_groups, n_in=widths[-1], epsilon=_EPSILON
    )
    encoder_conv_out = Conv2D(
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
    stages = _up_stages(widths, n_resnets=n_resnets, n_groups=n_groups)
    norm_out = GroupNorm("decoder_norm_out", n_groups=n_groups, n_in=widths[0], epsilon=_EPSILON)
    conv_out = Conv2D(
        "decoder_conv_out",
        in_channels=widths[0],
        out_channels=config.get("out_channels", 3),
        kernel_size=3,
        padding="same",
    )

    _bind_encoder(
        keys,
        encoder_conv_in,
        down_stages,
        encoder_mid,
        encoder_norm_out,
        encoder_conv_out,
        quant_conv,
    )
    _bind_decoder(keys, post_quant_conv, conv_in, mid, stages, norm_out, conv_out)

    hidden = encoder_conv_in(image)
    for down_stage in down_stages:
        hidden = down_stage(hidden)
    hidden = encoder_mid(hidden)
    moments = quant_conv(encoder_conv_out(activation(encoder_norm_out(hidden))))
    mean, log_variance = pt.split(moments, [latent_channels, latent_channels], n_splits=2, axis=-1)

    hidden = conv_in(post_quant_conv(latent))
    hidden = mid(hidden)
    for up_stage in stages:
        hidden = up_stage(hidden)
    sample = conv_out(activation(norm_out(hidden)))

    mean.name, log_variance.name, sample.name = "mean", "log_variance", "sample"
    return [image, latent], [mean, log_variance, sample]


class _MidBlock:
    """The pair of residual blocks the decoder puts an attention block between."""

    def __init__(self, width: int, *, n_groups: int, name: str):
        self.resnets = [
            ResnetBlock2D(
                f"{name}_resnet_{i}",
                in_channels=width,
                n_groups=n_groups,
                epsilon=_EPSILON,
            )
            for i in range(2)
        ]
        self.attention = AttentionBlock2D(
            f"{name}_attn", channels=width, n_groups=n_groups, epsilon=_EPSILON
        )

    def __call__(self, X: pt.TensorVariable) -> pt.TensorVariable:
        return self.resnets[1](self.attention(self.resnets[0](X)))


class _UpStage:
    """One resolution's residual blocks, and the nearest-neighbor upsample that follows them."""

    def __init__(self, in_width: int, out_width: int, *, index: int, n_resnets: int, n_groups: int):
        self.resnets = [
            ResnetBlock2D(
                f"decoder_up_{index}_resnet_{i}",
                in_channels=in_width if i == 0 else out_width,
                out_channels=out_width,
                n_groups=n_groups,
                epsilon=_EPSILON,
            )
            for i in range(n_resnets)
        ]
        self.upsample = Upsample2D(f"decoder_up_{index}_upsample", scale_factor=2, mode="nearest")
        self.conv = Conv2D(
            f"decoder_up_{index}_conv",
            in_channels=out_width,
            out_channels=out_width,
            kernel_size=3,
            padding="same",
        )

    def __call__(self, X: pt.TensorVariable) -> pt.TensorVariable:
        for resnet in self.resnets:
            X = resnet(X)
        return self.conv(self.upsample(X))


class _FinalUpStage(_UpStage):
    """The last stage, which is already at full resolution and so does not upsample."""

    def __call__(self, X: pt.TensorVariable) -> pt.TensorVariable:
        for resnet in self.resnets:
            X = resnet(X)
        return X


class _DownStage:
    """One resolution's residual blocks, and the strided convolution that halves the map after them."""

    def __init__(self, in_width: int, out_width: int, *, index: int, n_resnets: int, n_groups: int):
        self.resnets = [
            ResnetBlock2D(
                f"encoder_down_{index}_resnet_{i}",
                in_channels=in_width if i == 0 else out_width,
                out_channels=out_width,
                n_groups=n_groups,
                epsilon=_EPSILON,
            )
            for i in range(n_resnets)
        ]
        # Padded on the bottom and right only, so the stride-2 window starts at the top-left pixel.
        # Padding symmetrically instead shifts the whole map half a pixel with nothing to raise.
        self.pad = ZeroPad2D(f"encoder_down_{index}_pad", padding=((0, 1), (0, 1)))
        self.conv = Conv2D(
            f"encoder_down_{index}_conv",
            in_channels=out_width,
            out_channels=out_width,
            kernel_size=3,
            stride=2,
        )

    def __call__(self, X: pt.TensorVariable) -> pt.TensorVariable:
        for resnet in self.resnets:
            X = resnet(X)
        return self.conv(self.pad(X))


class _FinalDownStage(_DownStage):
    """The last stage, which is already at the latent's resolution and so does not downsample."""

    def __call__(self, X: pt.TensorVariable) -> pt.TensorVariable:
        for resnet in self.resnets:
            X = resnet(X)
        return X


def _down_stages(widths: list[int], *, n_resnets: int, n_groups: int) -> list[_DownStage]:
    """One stage per width, finest first, each widening the channel count the config asks it to."""
    stages, in_width = [], widths[0]
    for index, out_width in enumerate(widths):
        stage_type = _FinalDownStage if index == len(widths) - 1 else _DownStage
        stages.append(
            stage_type(in_width, out_width, index=index, n_resnets=n_resnets, n_groups=n_groups)
        )
        in_width = out_width
    return stages


def _up_stages(widths: list[int], *, n_resnets: int, n_groups: int) -> list[_UpStage]:
    """One stage per width, coarsest first, each halving the channel count the config asks it to."""
    reversed_widths = list(reversed(widths))
    stages, in_width = [], reversed_widths[0]
    for index, out_width in enumerate(reversed_widths):
        stage_type = _FinalUpStage if index == len(reversed_widths) - 1 else _UpStage
        stages.append(
            stage_type(in_width, out_width, index=index, n_resnets=n_resnets, n_groups=n_groups)
        )
        in_width = out_width
    return stages


def _bind_decoder(
    keys: KeyMap,
    post_quant_conv: Conv2D,
    conv_in: Conv2D,
    mid: _MidBlock,
    stages: list[_UpStage],
    norm_out: GroupNorm,
    conv_out: Conv2D,
) -> None:
    _bind_conv(keys, post_quant_conv, "post_quant_conv")
    with keys.scope("decoder"):
        _bind_conv(keys, conv_in, "conv_in")
        _bind_mid_block(keys, mid)
        with keys.scope("up_blocks"):
            for index, stage in enumerate(stages):
                with keys.scope(str(index)):
                    for i, resnet in enumerate(stage.resnets):
                        with keys.scope("resnets", str(i)):
                            _bind_resnet_block(keys, resnet)
                    if not isinstance(stage, _FinalUpStage):
                        _bind_conv(keys, stage.conv, "upsamplers", "0", "conv")
        _bind_group_norm(keys, norm_out, "conv_norm_out")
        _bind_conv(keys, conv_out, "conv_out")


def _bind_encoder(
    keys: KeyMap,
    conv_in: Conv2D,
    stages: list[_DownStage],
    mid: _MidBlock,
    norm_out: GroupNorm,
    conv_out: Conv2D,
    quant_conv: Conv2D,
) -> None:
    _bind_conv(keys, quant_conv, "quant_conv")
    with keys.scope("encoder"):
        _bind_conv(keys, conv_in, "conv_in")
        with keys.scope("down_blocks"):
            for index, stage in enumerate(stages):
                with keys.scope(str(index)):
                    for i, resnet in enumerate(stage.resnets):
                        with keys.scope("resnets", str(i)):
                            _bind_resnet_block(keys, resnet)
                    if not isinstance(stage, _FinalDownStage):
                        _bind_conv(keys, stage.conv, "downsamplers", "0", "conv")
        _bind_mid_block(keys, mid)
        _bind_group_norm(keys, norm_out, "conv_norm_out")
        _bind_conv(keys, conv_out, "conv_out")


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
