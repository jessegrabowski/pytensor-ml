import json

from collections import Counter
from collections.abc import Mapping, Sequence
from os import fspath
from pathlib import Path
from typing import Any

import numpy as np

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.tensor.random.type import RandomGeneratorType
from safetensors import safe_open
from safetensors.numpy import save_file

# Generator state rides in the archive's metadata, which is a flat str-to-str map shared with whatever
# wrote the file, so our entries are namespaced away from a foreign writer's (HuggingFace stores a
# "format" key there).
_RNG_KEY_PREFIX = "rng/"


def holds_generator(variable: SharedVariable) -> bool:
    """Report whether a shared variable holds a random generator rather than a tensor."""
    return isinstance(variable.type, RandomGeneratorType)


def jsonable_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a bit generator's state with its arrays tagged, so it survives a JSON round trip.

    MT19937 carries its key as an array and Philox its counter, neither of which JSON takes directly.
    """

    def tag(value):
        if isinstance(value, np.ndarray):
            return {"__array__": value.tolist(), "dtype": value.dtype.name}
        if isinstance(value, Mapping):
            return {key: tag(item) for key, item in value.items()}
        return value

    return tag(state)


def rng_state_from_jsonable(data: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a bit generator's state from :func:`jsonable_rng_state`, restoring its arrays."""

    def untag(value):
        if isinstance(value, Mapping):
            if "__array__" in value:
                return np.array(value["__array__"], dtype=value["dtype"])
            return {key: untag(item) for key, item in value.items()}
        return value

    return untag(data)


def bit_generator_kind(state: Mapping[str, Any]) -> type[np.random.BitGenerator]:
    """
    Resolve the bit generator class ``state`` was written by.

    Parameters
    ----------
    state : mapping
        A bit generator's state. Only the recorded kind is read, so a state whose contents are
        unusable still resolves.

    Returns
    -------
    kind : type
        The :class:`numpy.random.BitGenerator` subclass the state names.
    """
    name = state["bit_generator"]
    kind = getattr(np.random, name, None)
    if not (isinstance(kind, type) and issubclass(kind, np.random.BitGenerator)):
        raise ValueError(
            f"Cannot rebuild a {name!r} bit generator: numpy has no bit generator by that name."
        )
    return kind


def generator_from_state(state: Mapping[str, Any]) -> np.random.Generator:
    """
    Build a generator of the kind ``state`` came from, restored to it.

    Parameters
    ----------
    state : mapping
        A bit generator's state, either as :attr:`numpy.random.BitGenerator.state` reports it or as
        :func:`jsonable_rng_state` renders it for storage.

    Returns
    -------
    generator : numpy.random.Generator
        A generator whose bit generator holds ``state``.
    """
    state = rng_state_from_jsonable(state)
    # Seeded rather than left to OS entropy, which the next line discards anyway -- MT19937 would
    # otherwise run a full 624-word init per rebuild.
    generator = np.random.Generator(bit_generator_kind(state)(0))
    generator.bit_generator.state = dict(state)
    return generator


def _numbered(variable: SharedVariable, name: str, ordinal: int) -> str:
    """Number the layer a variable belongs to, so ``Linear_W`` keys as ``Linear_1_W`` not ``Linear_W_1``."""
    layer_name = getattr(variable, "layer_name", None)
    if layer_name and name.startswith(layer_name):
        return f"{layer_name}_{ordinal}{name[len(layer_name) :]}"
    return f"{name}_{ordinal}"


def _index_by_key(shared_variables: Sequence[SharedVariable]) -> dict[str, SharedVariable]:
    """
    Index shared variables by their archive key, numbering a name that repeats.

    Layers left unnamed take their class name, so a stack of several of a kind offers several variables
    of one name. Each repeat is numbered after its layer -- ``Linear_1_W``, ``Linear_2_W`` -- in the
    order given, so the keys hold only for that order. Collecting from the same graph reproduces it.
    """
    names: list[str] = []
    for variable in shared_variables:
        if variable.name is None:
            raise ValueError(
                f"Cannot checkpoint the unnamed shared variable {variable!r}: the name is the only "
                f"handle at the serialization boundary, so every variable must be named."
            )
        names.append(variable.name)

    occurrences = Counter(names)
    indexed: dict[str, SharedVariable] = {}
    ordinals: Counter[str] = Counter()
    for name, variable in zip(names, shared_variables):
        if occurrences[name] == 1:
            indexed[name] = variable
            continue
        ordinals[name] += 1
        key = _numbered(variable, name, ordinals[name])
        if key in occurrences:
            raise ValueError(
                f"Numbering the shared variables named {name!r} produces {key!r}, which another "
                f"variable already uses. Name those layers explicitly."
            )
        indexed[key] = variable
    return indexed


def save_state(shared_variables: Sequence[SharedVariable], path: str | Path) -> None:
    """
    Save the current values of shared variables to a name-keyed ``.safetensors`` archive.

    The variables' names become the archive keys. Variables sharing a name -- as layers left unnamed
    do -- are numbered after the layer that built them (``Linear_1_W``, ``Linear_2_W``) in the order
    given, so collect them the same way at save and at load. Pass the parameters together with the
    optimizer state to capture a complete training checkpoint; both are ordinary shared variables and
    carry self-describing names (e.g. ``"fc1/weight"``, ``"fc1/weight/adam/first_moment"``).
    :func:`~pytensor_ml.pytensorf.collect_optimizer_state` finds the half no walk of the graph reaches.
    A random
    generator has no tensor to store, so its state is written to the archive's metadata under the same
    key, which is what lets a stochastic network checkpoint at all.

    Parameters
    ----------
    shared_variables : sequence of SharedVariable
        Variables whose values to save, random generators included. Every variable must be named; repeats
        are numbered by position.
    path : str or pathlib.Path
        Destination archive, written verbatim.

    Examples
    --------
    Write parameter values to safetensors, keyed by name. This is the checkpoint half of a run: it
    saves what the parameters hold, not what built them. The destination directory has to exist
    already:

    .. code-block:: python

        from pytensor_ml import save_state
        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.model import Model
        from pytensor_ml.pytensorf import collect_shared_variables

        X = Input("X", shape=(None, 64))
        logits = Linear("logits", n_in=64, n_out=10)(X)
        Model(logits).initialize(seed=0)

        save_state(collect_shared_variables(logits), "weights.safetensors")
    """
    indexed = _index_by_key(shared_variables)
    tensors: dict[str, np.ndarray] = {}
    generator_states: dict[str, str] = {}
    for key, variable in indexed.items():
        if holds_generator(variable):
            state = variable.get_value(borrow=True).bit_generator.state
            generator_states[_RNG_KEY_PREFIX + key] = json.dumps(jsonable_rng_state(state))
        else:
            tensors[key] = _as_saveable_array(variable.get_value(), variable.type.dtype)
    save_file(tensors, fspath(path), metadata=generator_states or None)


def _as_saveable_array(value: Any, dtype: str) -> np.ndarray:
    """
    Return a value as a C-contiguous numpy array of ``dtype`` that safetensors will accept, preserving its
    rank.

    A JIT backend stores its own array type in the shared variables a compiled function updates, so a
    trained model's values are ``jax.Array`` or ``mlx.core.array``; both convert through the array
    protocol. It may also narrow them -- JAX computes in 32-bit unless ``floatX`` is ``float64`` -- so the
    archive records the declared dtype to stay portable across precisions.
    """
    array = np.asarray(value, dtype=dtype)
    # Not np.ascontiguousarray: it forces ndim >= 1, reshaping a rank-0 array such as the step count.
    return array if array.flags["C_CONTIGUOUS"] else np.array(array, order="C")


def load_state(
    shared_variables: Sequence[SharedVariable],
    path: str | Path,
    name_map: Mapping[str, str] | None = None,
) -> None:
    """
    Load values from a name-keyed ``.safetensors`` archive into shared variables, in place.

    Match is by name: each variable's name (optionally remapped through ``name_map``) selects the archive
    entry to load into it. A random generator takes its state from the archive's metadata, and its bit
    generator has to be the kind the archive holds state for. The archive's keys and the target names
    must correspond exactly -- a missing or extra key, or any shape or dtype mismatch, raises rather than
    loading partial state. Every target is validated before any value is written, so a failed load leaves
    all variables untouched.

    Parameters
    ----------
    shared_variables : sequence of SharedVariable
        Variables to restore. Every variable must be named; repeats are numbered by position, so pass
        them in the order they were saved in.
    path : str or pathlib.Path
        A ``.safetensors`` archive, e.g. one written by :func:`save_state` or a HuggingFace checkpoint.
    name_map : mapping of str to str, optional
        Maps a variable's name to the archive key to read it from, for loading a checkpoint saved under
        different names (such as HuggingFace's). The mapping must be injective. Names absent from the map
        are matched directly.

    Examples
    --------
    Fill an existing graph's parameters from a checkpoint, matching them by name. Build the same network
    first -- this loads values into it rather than reconstructing it:

    .. code-block:: python

        from pytensor_ml import load_state, save_state
        from pytensor_ml.layers import Input, Linear
        from pytensor_ml.model import Model
        from pytensor_ml.pytensorf import collect_shared_variables

        X = Input("X", shape=(None, 64))
        logits = Linear("logits", n_in=64, n_out=10)(X)
        Model(logits).initialize(seed=0)

        shared = collect_shared_variables(logits)
        save_state(shared, "weights.safetensors")
        load_state(shared, "weights.safetensors")
    """
    indexed = _index_by_key(shared_variables)
    name_map = name_map or {}

    target_by_key: dict[str, SharedVariable] = {}
    source_by_key: dict[str, str] = {}
    for source, variable in indexed.items():
        key = name_map.get(source, source)
        collision = source_by_key.get(key)
        if collision is not None:
            raise ValueError(
                f"name_map is not injective: both {collision!r} and {source!r} map to {key!r}."
            )
        source_by_key[key] = source
        target_by_key[key] = variable

    with safe_open(fspath(path), framework="numpy") as archive:
        values = {key: archive.get_tensor(key) for key in archive.keys()}
        metadata = archive.metadata() or {}
    archived_states = {
        key.removeprefix(_RNG_KEY_PREFIX): encoded
        for key, encoded in metadata.items()
        if key.startswith(_RNG_KEY_PREFIX)
    }
    generator_targets = {k: v for k, v in target_by_key.items() if holds_generator(v)}
    tensor_targets = {k: v for k, v in target_by_key.items() if not holds_generator(v)}

    missing = (set(tensor_targets) - set(values)) | (set(generator_targets) - set(archived_states))
    unexpected = (set(values) - set(tensor_targets)) | (
        set(archived_states) - set(generator_targets)
    )
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing from archive: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected in archive: {sorted(unexpected)}")
        raise ValueError(f"Archive keys do not match the targets ({'; '.join(details)}).")

    mismatches = []
    # Built here rather than at the write below: numpy only validates a state's contents when it is
    # assigned, and a state that fails there would leave the tensors already written.
    restored_generators: dict[str, np.random.Generator] = {}
    for key, variable in generator_targets.items():
        try:
            restored_generators[key] = generator_from_state(json.loads(archived_states[key]))
        except Exception as failure:
            mismatches.append(
                f"  {variable.name!r}: the state under {_RNG_KEY_PREFIX + key!r} is not usable "
                f"generator state ({type(failure).__name__}: {failure})"
            )
            continue
        target_kind = type(variable.get_value(borrow=True).bit_generator).__name__
        archived_kind = type(restored_generators[key].bit_generator).__name__
        if archived_kind != target_kind:
            mismatches.append(
                f"  {variable.name!r}: target is a {target_kind} generator, archive holds "
                f"{archived_kind} state"
            )
    for key, variable in tensor_targets.items():
        value = values[key]
        # The declared dtype, not the stored value's, which a backend may have narrowed (see
        # _as_saveable_array); set_value filters to the declaration anyway. Shape has to come from the
        # value, whose runtime dimensions the declared type may leave unknown.
        current = np.asarray(variable.get_value(borrow=True))
        declared_dtype = np.dtype(variable.type.dtype)
        if value.shape != current.shape or value.dtype != declared_dtype:
            mismatches.append(
                f"  {variable.name!r}: target is {declared_dtype} of shape {current.shape}, "
                f"archive has {value.dtype} of shape {value.shape}"
            )
    if mismatches:
        raise ValueError("Archive values do not match their targets:\n" + "\n".join(mismatches))

    for key, variable in tensor_targets.items():
        variable.set_value(values[key])
    for key, variable in generator_targets.items():
        variable.get_value(borrow=True).bit_generator.state = restored_generators[
            key
        ].bit_generator.state
