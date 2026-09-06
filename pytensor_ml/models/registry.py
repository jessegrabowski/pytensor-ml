from collections.abc import Callable

from pytensor.graph.basic import Variable

from pytensor_ml.models.keys import KeyMap

Builder = Callable[[dict, KeyMap], tuple[list[Variable], Variable | list[Variable]]]

_BUILDERS: dict[str, Builder] = {}


def architecture_name(config: dict) -> str | None:
    """
    Name of the class a HuggingFace config declares, or None when it declares none.

    Diffusers writes ``_class_name``; transformers writes the class into ``architectures`` alongside a
    lowercase ``model_type`` slug. Both spellings resolve to the same class name, which is what a
    builder registers under.

    Parameters
    ----------
    config : dict
        Parsed contents of a HuggingFace ``config.json``.

    Returns
    -------
    name : str or None
        Declared class name, such as "AutoencoderKL" or "CLIPTextModel".
    """
    declared = config.get("_class_name")
    if declared is not None:
        return declared

    architectures = config.get("architectures")
    if architectures:
        return architectures[0]
    return None


def register_builder(architecture: str) -> Callable[[Builder], Builder]:
    """
    Register a builder for the architecture a config names.

    Parameters
    ----------
    architecture : str
        Class name to dispatch on, matching what :func:`architecture_name` reads off a config.

    Returns
    -------
    decorator : callable
        Decorator registering the builder it is applied to and returning it unchanged.

    Examples
    --------
    A builder takes the parsed config and returns the graph it describes, in the same
    ``(data_inputs, outputs)`` shape :func:`~pytensor_ml.pretrained.from_pretrained` returns:

    .. code-block:: python

        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.models import channels_last, register_builder


        @register_builder("MyEncoder")
        def build_my_encoder(config, keys):
            X = Input("X", shape=(None, config["hidden_size"]))
            fc = Linear("fc", n_in=config["hidden_size"], n_out=config["out_size"])
            keys.bind(fc.W, "fc.weight", transform=channels_last)
            return [X], fc(X)
    """

    def decorator(builder: Builder) -> Builder:
        registered = _BUILDERS.get(architecture)
        if registered is not None:
            raise ValueError(
                f"{architecture!r} already has a builder, {registered.__module__}."
                f"{registered.__qualname__}. Two builders for one architecture would make which "
                f"one runs depend on import order."
            )
        _BUILDERS[architecture] = builder
        return builder

    return decorator


def build_from_config(
    config: dict,
) -> tuple[list[Variable], Variable | list[Variable], KeyMap]:
    """
    Build the graph a HuggingFace config describes.

    Parameters
    ----------
    config : dict
        Parsed contents of a HuggingFace ``config.json``.

    Returns
    -------
    data_inputs : list of Variable
        The graph's data inputs.
    outputs : Variable or list of Variable
        The graph the builder produced.
    keys : KeyMap
        The checkpoint key each parameter loads from, recorded as the builder ran.
    """
    architecture = architecture_name(config)
    if architecture is None:
        raise ValueError(
            "This config names no architecture: it has neither _class_name (diffusers) nor "
            "architectures (transformers), so there is nothing to dispatch on."
        )
    builder = _BUILDERS.get(architecture)
    if builder is None:
        raise ValueError(
            f"No builder is registered for {architecture!r}. Registered architectures: "
            f"{sorted(_BUILDERS)}."
        )
    keys = KeyMap()
    return (*builder(config, keys), keys)
