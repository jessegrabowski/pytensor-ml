import json

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytensor

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Variable
from pytensor.tensor.random.type import RandomGeneratorType
from safetensors import safe_open

from pytensor_ml.checkpoint import (
    bit_generator_kind,
    generator_from_state,
    holds_generator,
    jsonable_rng_state,
    load_state,
    save_state,
)
from pytensor_ml.json_serialize import (
    deserialize_graph,
    props_from_json,
    props_to_json,
    qualname,
    resolve_class,
    serialize_graph,
    type_from_json,
)
from pytensor_ml.models import build_from_config
from pytensor_ml.params import NonTrainableParameter, TrainableParameter, non_trainable, trainable
from pytensor_ml.pytensorf import (
    as_output_list,
    collect_data_inputs,
    collect_shared_variables,
)
from pytensor_ml.state import (
    EmptyInitializer,
    Initializer,
    UnrecordedInitializer,
    initial_values_from,
)

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"

# Marks a config as a pytensor_ml graph (vs a HuggingFace config, which shares the config.json filename but
# is a hyperparameter sheet, not a serialized graph). The version guards future schema changes.
GRAPH_FORMAT = "pytensor_ml.graph"
GRAPH_FORMAT_VERSION = 4

Format = Literal["auto", "pytensor", "huggingface"]


class InputKind(StrEnum):
    """How a serialized graph input is rebuilt: as a data placeholder or a kind of shared variable."""

    DATA = "data"
    TRAINABLE = "trainable"
    NON_TRAINABLE = "non_trainable"
    SHARED = "shared"
    RNG = "rng"


def _looks_like_huggingface(config: dict) -> bool:
    # Diffusers writes _class_name; transformers writes model_type and architectures. A directory with
    # any of them is somebody else's checkpoint rather than one of ours.
    return any(key in config for key in ("_class_name", "model_type", "architectures"))


def _huggingface_weights(directory: Path, variant: str | None) -> Path:
    """
    The safetensors file in a HuggingFace component directory.

    The name varies by toolchain and precision -- transformers writes ``model.safetensors``, diffusers
    writes ``diffusion_pytorch_model.safetensors``, and a half-precision download inserts a variant
    before the extension -- so it is resolved rather than assumed.
    """
    shards = sorted(directory.glob("*.safetensors.index.json"))
    if shards:
        raise NotImplementedError(
            f"{directory} holds a sharded checkpoint ({shards[0].name}). Sharded checkpoints are not "
            f"read yet; no single shard is the whole model."
        )

    candidates = sorted(directory.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"No .safetensors weights in {directory}. Only safetensors is read; a .bin checkpoint "
            f"has to be converted first."
        )

    if variant is not None:
        # Matched on the stem rather than on suffixes, which would read the "1.0" of a name like
        # sd-xl-1.0.safetensors as a variant.
        candidates = [path for path in candidates if path.stem.endswith(f".{variant}")]
        if not candidates:
            raise FileNotFoundError(f"No *.{variant}.safetensors in {directory}.")

    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"{directory} holds several weight files ({names}), so which to load is ambiguous. Pass "
            f"variant= to choose one."
        )
    return candidates[0]


def _detect_format(config: dict) -> Format:
    if config.get("format") == GRAPH_FORMAT:
        return "pytensor"
    if _looks_like_huggingface(config):
        return "huggingface"
    raise ValueError("Unrecognized config: not a pytensor_ml graph or a HuggingFace model.")


def _weight_variables(outputs: Variable | Sequence[Variable]) -> list[SharedVariable]:
    # Random generators are excluded: their state rides in the config as JSON, not as a tensor.
    return [
        variable
        for variable in collect_shared_variables(as_output_list(outputs))
        if not holds_generator(variable)
    ]


def _input_kind(variable: Variable) -> InputKind:
    if isinstance(variable.type, RandomGeneratorType) and isinstance(variable, SharedVariable):
        return InputKind.RNG
    if isinstance(variable, TrainableParameter):
        return InputKind.TRAINABLE
    if isinstance(variable, NonTrainableParameter):
        return InputKind.NON_TRAINABLE
    if isinstance(variable, SharedVariable):
        return InputKind.SHARED
    return InputKind.DATA


def _recordable(initializer: Initializer) -> dict | None:
    """Return the encoding of ``initializer``, or None when it cannot be written down and read back."""
    class_path = qualname(initializer)
    try:
        props = props_to_json(initializer)
    except TypeError:
        return None  # a parameter that is not JSON, such as an array or a fitted model
    try:
        restored_class = resolve_class(class_path)
    except (ImportError, AttributeError):
        return None  # defined where an import cannot reach it, such as inside a function

    if restored_class is not type(initializer):
        return None  # the name at that path now means something else
    return {"class": class_path, "props": props}


def _initializer_to_json(initializer: Initializer) -> dict:
    """
    Encode the law a parameter is drawn from, as a class path and its ``__props__``.

    An initializer that cannot be written down is recorded as an :class:`UnrecordedInitializer` naming it.
    Saving still succeeds, because restoring values is the usual reason to save and that needs no law at
    all; only a redraw needs one, and that is where the loss is reported.
    """
    encoded = _recordable(initializer)
    if encoded is not None:
        return encoded

    lost = UnrecordedInitializer(type(initializer).__name__)
    return {"class": qualname(lost), "props": props_to_json(lost)}


def _initializer_from_json(initializer_dict: dict) -> Initializer:
    return resolve_class(initializer_dict["class"])(**props_from_json(initializer_dict["props"]))


def _input_meta(variable: Variable) -> dict:
    meta: dict[str, Any] = {"name": variable.name, "kind": _input_kind(variable)}
    if isinstance(variable, SharedVariable) and meta["kind"] == InputKind.RNG:
        # Captured for exact reproducibility even though load does not restore it by default.
        state = variable.get_value(borrow=True).bit_generator.state
        meta["rng_state"] = jsonable_rng_state(state)
    if isinstance(variable, TrainableParameter) and variable.initializer is not None:
        meta["initializer"] = _initializer_to_json(variable.initializer)
    return meta


def _rebuild_trainable(graph_type, meta: dict) -> TrainableParameter:
    """
    Rebuild one trainable parameter, holding a draw from the initializer the config recorded for it.

    Drawn rather than zero-filled, so a rebuilt parameter holds a value for the same reason a freshly
    constructed one does. An initializer the config could not record has nothing to draw from, and says so
    on the redraw that needs it rather than here, where restoring saved values is the point.
    """
    initializer_dict = meta.get("initializer")
    initializer = None if initializer_dict is None else _initializer_from_json(initializer_dict)

    if initializer is None or isinstance(initializer, UnrecordedInitializer):
        value = np.zeros(graph_type.shape, dtype=graph_type.dtype)
    else:
        value = initializer.initial_value(graph_type.shape)

    return trainable(value, meta["name"], initializer=initializer)


def _rebuild_input(type_json: dict, meta: dict, restore_rng: bool):
    kind, name = meta["kind"], meta["name"]
    if kind == InputKind.DATA:
        return type_from_json(type_json)(name=name)
    if kind == InputKind.RNG:
        if restore_rng:
            generator = generator_from_state(meta["rng_state"])
        else:
            # A fresh stream, but of the kind the network was saved with: the default kind silently
            # rebuilds a different architecture, and any later load_state then fails on the kind.
            # Only the recorded kind is read, so state this call discards cannot make it fail.
            generator = np.random.Generator(bit_generator_kind(meta["rng_state"])())
        return pytensor.shared(generator, name=name)

    graph_type = type_from_json(type_json)
    if kind == InputKind.TRAINABLE:
        return _rebuild_trainable(graph_type, meta)

    placeholder = np.zeros(graph_type.shape, dtype=graph_type.dtype)
    if kind == InputKind.NON_TRAINABLE:
        return non_trainable(placeholder, name)
    return pytensor.shared(placeholder, name=name)


def save_network(
    outputs: Variable | Sequence[Variable],
    path: str | Path,
    *,
    inputs: Sequence[Variable] | None = None,
) -> None:
    """
    Serialize a network's architecture (its graph) to a JSON config file, without parameter values.

    Records each input's name and kind (data, trainable, non-trainable, or plain shared) so
    :func:`load_network` can rebuild the graph with the right variable identities. The data inputs are
    collected from ``outputs`` unless given explicitly.

    A random generator is the one input whose *value* is recorded here, since it has no tensor to ride in
    a weights archive. That makes the config depend on how far the generators have been drawn, so two
    saves of one architecture differ; :func:`load_network` reads it back only when asked.

    Parameters
    ----------
    outputs : Variable or sequence of Variable
        The network's output(s).
    path : str or pathlib.Path
        Destination JSON file.
    inputs : sequence of Variable, optional
        The network's data inputs, in call order. Collected from ``outputs`` when omitted; pass explicitly
        when call order matters.

    Examples
    --------
    Write the architecture alone, as JSON. The weights are not in it, so pair it with
    :func:`~pytensor_ml.checkpoint.save_state` or use :func:`save_pretrained`, which writes both.
    The destination directory has to exist already:

    .. code-block:: python

        from pytensor_ml import save_network
        from pytensor_ml.layers import Input, Linear

        X = Input("X", shape=(None, 64))
        logits = Linear("logits", n_in=64, n_out=10)(X)

        save_network(logits, "config.json")
    """
    output_list = as_output_list(outputs)
    data_inputs = list(inputs) if inputs is not None else collect_data_inputs(output_list)
    leaves = [*data_inputs, *collect_shared_variables(output_list)]

    config = {"format": GRAPH_FORMAT, "format_version": GRAPH_FORMAT_VERSION}
    config.update(serialize_graph(leaves, output_list))
    config["input_meta"] = [_input_meta(leaf) for leaf in leaves]
    config["n_outputs"] = len(output_list)
    Path(path).write_text(json.dumps(config))


def load_network(
    path: str | Path, *, restore_rng: bool = False
) -> tuple[list[Variable], Variable | list[Variable]]:
    """
    Rebuild a network's graph from a :func:`save_network` config file.

    Shared variables keep their original names and kinds, so a subsequent :func:`load_state` (or
    :func:`from_pretrained`) can fill them by name. This restores the architecture only -- the values are
    not the saved ones.

    A trainable parameter comes back holding a draw from the initializer it was built with, as it would if
    the layer had constructed it, so a loaded architecture is trainable without further calls and a batch
    norm layer returns to its identity transform. Other shared state is zero-filled, having no law to
    redraw from.

    Parameters
    ----------
    path : str or pathlib.Path
        A config file written by :func:`save_network`.
    restore_rng : bool
        If True, restore each random generator to its saved state for exact reproducibility. By default a
        fresh generator is created, so stochastic layers draw new randomness. Default False.

    Returns
    -------
    inputs : list of Variable
        The network's data inputs, in call order.
    outputs : Variable or list of Variable
        The rebuilt output(s) -- a single variable when the network has one output, otherwise a list.

    Examples
    --------
    Rebuild a saved architecture with freshly drawn weights, which is what you want when the values are
    about to be trained or loaded separately:

    .. code-block:: python

        from pytensor_ml import load_network, save_network
        from pytensor_ml.layers import Input, Linear

        X = Input("X", shape=(None, 64))
        save_network(Linear("logits", n_in=64, n_out=10)(X), "config.json")

        inputs, outputs = load_network("config.json")
    """
    config = json.loads(Path(path).read_text())
    if config.get("format") != GRAPH_FORMAT:
        hint = " (this looks like a HuggingFace config)" if _looks_like_huggingface(config) else ""
        raise ValueError(f"{path} is not a pytensor_ml network config{hint}.")

    # Before rebuilding: op classes are recorded by import path, so a stale layout would otherwise fail
    # deep inside class resolution.
    written_version = config.get("format_version")
    if written_version != GRAPH_FORMAT_VERSION:
        raise ValueError(
            f"{path} is graph format version {written_version}, but this pytensor_ml reads version "
            f"{GRAPH_FORMAT_VERSION}. Rebuild the network and call save_network again."
        )

    leaves = [
        _rebuild_input(type_json, meta, restore_rng)
        for type_json, meta in zip(config["inputs"], config["input_meta"])
    ]
    _, outputs = deserialize_graph(config, inputs=leaves)

    data_inputs = [
        leaf for leaf, meta in zip(leaves, config["input_meta"]) if meta["kind"] == InputKind.DATA
    ]
    return data_inputs, (outputs[0] if config["n_outputs"] == 1 else outputs)


def save_pretrained(
    outputs: Variable | Sequence[Variable],
    directory: str | Path,
    *,
    inputs: Sequence[Variable] | None = None,
) -> None:
    """
    Save a complete network -- architecture and weights -- to a directory.

    Writes ``config.json`` (the graph) and ``model.safetensors`` (the parameter values), the layout
    :func:`from_pretrained` expects.

    Parameters
    ----------
    outputs : Variable or sequence of Variable
        The network's output(s).
    directory : str or pathlib.Path
        Destination directory, created if needed.
    inputs : sequence of Variable, optional
        The network's data inputs, in call order. Collected from ``outputs`` when omitted.

    Examples
    --------
    Write the architecture and the weights together, so the directory reloads into a runnable network
    without the Python that defined it:

    .. code-block:: python

        from pytensor_ml import save_pretrained
        from pytensor_ml.activations import ReLU
        from pytensor_ml.layers import Input, Linear, Sequential
        from pytensor_ml.model import Model

        X = Input("X", shape=(None, 64))
        network = Sequential(
            Linear("fc", n_in=64, n_out=32),
            ReLU(),
            Linear("logits", n_in=32, n_out=10),
        )
        logits = network(X)
        Model(logits).initialize(seed=0)

        save_pretrained(logits, "artifacts/model")
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_network(outputs, directory / CONFIG_FILENAME, inputs=inputs)
    save_state(_weight_variables(outputs), directory / WEIGHTS_FILENAME)


def from_pretrained(
    directory: str | Path,
    source_format: Format = "auto",
    *,
    restore_rng: bool = False,
    variant: str | None = None,
) -> tuple[list[Variable], Variable | list[Variable]]:
    """
    Load a complete network -- architecture and weights -- from a directory.

    For a pytensor_ml directory, rebuilds the graph from ``config.json`` and fills its parameters from
    ``model.safetensors``. The format is detected from the config by default, since a pytensor_ml graph and
    a HuggingFace model share the same filenames but not the same schema.

    For a HuggingFace directory, dispatches the config to a registered builder and fills the graph it
    returns from the component's safetensors file. Each weight casts to its parameter's dtype, which
    ``floatX`` fixed when the layer built it. A checkpoint tensor no parameter loads from is logged
    rather than treated as an error, since every parameter still got a value.

    Parameters
    ----------
    directory : str or pathlib.Path
        A directory holding ``config.json`` and ``model.safetensors``.
    source_format : {'auto', 'pytensor', 'huggingface'}
        Which loader to use. ``'auto'`` detects the format from the config's marker. Default 'auto'.
    restore_rng : bool
        If True, restore each random generator to its saved state for exact reproducibility. Only a
        pytensor_ml directory carries generator state. Default False.
    variant : str, optional
        Picks between weight files when a directory holds several, as in ``model.fp16.safetensors``.
        Unnecessary when there is only one.

    Returns
    -------
    inputs : list of Variable
        The network's data inputs, in call order.
    outputs : Variable or list of Variable
        The rebuilt, weight-filled output(s).

    Examples
    --------
    Rebuild a saved network and fill its weights, returning the data inputs and the outputs. Nothing that
    defined the network has to be importable:

    .. code-block:: python

        import numpy as np

        from pytensor_ml import from_pretrained, save_pretrained
        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.model import Model
        from pytensor_ml.pytensorf import function

        X = Input("X", shape=(None, 64))
        logits = Linear("logits", n_in=64, n_out=10)(X)
        Model(logits).initialize(seed=0)
        save_pretrained(logits, "artifacts/model")

        inputs, outputs = from_pretrained("artifacts/model")
        predictions = function(inputs, outputs)(np.zeros((4, 64)))
    """
    directory = Path(directory)
    if source_format == "auto":
        source_format = _detect_format(json.loads((directory / CONFIG_FILENAME).read_text()))
    if source_format == "huggingface":
        if restore_rng:
            raise ValueError(
                "restore_rng restores a pytensor_ml graph's saved generator state, and a HuggingFace "
                "checkpoint carries none."
            )
        config = json.loads((directory / CONFIG_FILENAME).read_text())
        # Every parameter the builder makes is bound to a checkpoint key, and load fills all of them or
        # raises, so drawing them first is work thrown away.
        with initial_values_from(EmptyInitializer()):
            data_inputs, outputs, keys = build_from_config(config)
        with safe_open(_huggingface_weights(directory, variant), framework="numpy") as weights:
            keys.load(weights.get_tensor, weights.keys())
        return data_inputs, outputs

    # load_state fills every weight or raises, so the draws load_network would make are thrown away too.
    with initial_values_from(EmptyInitializer()):
        data_inputs, outputs = load_network(directory / CONFIG_FILENAME, restore_rng=restore_rng)
    load_state(_weight_variables(outputs), directory / WEIGHTS_FILENAME)
    return data_inputs, outputs
