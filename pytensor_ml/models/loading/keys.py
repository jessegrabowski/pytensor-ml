import logging

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import NamedTuple

import numpy as np

from pytensor.compile.sharedvalue import SharedVariable

_log = logging.getLogger(__name__)

Transform = Callable[[np.ndarray], np.ndarray]


class _Binding(NamedTuple):
    key: str
    transform: Transform | None


def channels_last(array: np.ndarray) -> np.ndarray:
    """
    Move a checkpoint's leading output and input channel axes to the end, input before output.

    HuggingFace stores weights channel-first -- ``(out, in)`` for a dense layer and ``(out, in,
    *kernel)`` for a convolution -- and this library stores them channel-last. One move covers every
    rank: a dense weight comes back transposed, and a convolution comes back as ``(*kernel, in, out)``.
    """
    return np.moveaxis(array, (0, 1), (-1, -2))


class KeyMap:
    """
    Pairs of a parameter and the checkpoint key it loads from, recorded while a builder runs.

    A builder knows the module path of what it is building, so it records each correspondence as it
    goes rather than a loader rediscovering it afterwards from names. The map holds the parameter
    object itself, so two parameters that happen to share a name stay distinct.

    Examples
    --------
    Nest :meth:`scope` the way the module path nests, and bind the leaf names under it:

    .. code-block:: python

        from pytensor_ml.layers import Linear
        from pytensor_ml.models import KeyMap

        keys = KeyMap()
        with keys.scope("text_model", "encoder"):
            for i in range(2):
                block = Linear(f"fc_{i}", n_in=4, n_out=4)
                with keys.scope("layers", str(i), "mlp", "fc1"):
                    keys.bind(block.W, "weight")
                    keys.bind(block.b, "bias")

        assert keys.key_for(block.b) == "text_model.encoder.layers.1.mlp.fc1.bias"
    """

    def __init__(self) -> None:
        self._prefix: tuple[str, ...] = ()
        self._bindings: dict[SharedVariable, _Binding] = {}
        self._by_key: dict[str, SharedVariable] = {}

    @contextmanager
    def scope(self, *parts: str) -> Iterator[None]:
        """Append ``parts`` to the module path every :meth:`bind` inside the block builds on."""
        outer = self._prefix
        self._prefix = outer + tuple(parts)
        try:
            yield
        finally:
            self._prefix = outer

    def bind(
        self, parameter: SharedVariable, *parts: str, transform: Transform | None = None
    ) -> None:
        """
        Record that ``parameter`` loads from the key ``parts`` names under the current scope.

        Parameters
        ----------
        parameter : SharedVariable
            The parameter a layer built, held by identity rather than by name.
        *parts : str
            Path components below the current scope, joined with dots.
        transform : callable, optional
            Applied to the checkpoint array before it is stored, typically :func:`channels_last`.
            Stored unchanged when omitted.
        """
        key = ".".join(self._prefix + tuple(parts))

        bound = self._bindings.get(parameter)
        if bound is not None:
            raise ValueError(
                f"{parameter.name or parameter} is already bound to {bound.key!r}, so binding it to "
                f"{key!r} would leave one of the two keys silently unloaded."
            )
        claimed_by = self._by_key.get(key)
        if claimed_by is not None:
            raise ValueError(
                f"{key!r} is already bound to {claimed_by.name or claimed_by}. One checkpoint tensor "
                f"cannot load into two parameters."
            )

        self._bindings[parameter] = _Binding(key, transform)
        self._by_key[key] = parameter

    def key_for(self, parameter: SharedVariable) -> str:
        """Checkpoint key ``parameter`` loads from."""
        bound = self._bindings.get(parameter)
        if bound is None:
            raise KeyError(f"{parameter.name or parameter} was never bound to a checkpoint key.")
        return bound.key

    def parameter_for(self, key: str) -> SharedVariable:
        """Parameter the checkpoint key ``key`` loads into."""
        parameter = self._by_key.get(key)
        if parameter is None:
            raise KeyError(f"No parameter is bound to {key!r}.")
        return parameter

    def keys(self) -> set[str]:
        """Every checkpoint key the builder bound."""
        return set(self._by_key)

    def __len__(self) -> int:
        return len(self._bindings)

    def _checkpoint_names(self, checkpoint: set[str]) -> dict[str, str]:
        """
        Every bound key paired with the name the checkpoint gives it.

        Whether a checkpoint carries the model's own leading segment depends on the class it was
        saved from: transformers drops it for a bare ``CLIPTextModel`` and keeps it for a
        ``GPT2LMHeadModel``, and the published files invert both. One segment is added or dropped to
        match, and the bound names stand when neither resolves the whole set.
        """
        bound = self.keys()
        if bound <= checkpoint:
            return {key: key for key in bound}

        leading = {key.split(".", 1)[0] for key in bound}
        for prefix in sorted({key.split(".", 1)[0] for key in checkpoint} - leading):
            renamed = {key: f"{prefix}.{key}" for key in bound}
            if set(renamed.values()) <= checkpoint:
                return renamed

        if len(leading) == 1:
            prefix = f"{leading.pop()}."
            renamed = {key: key.removeprefix(prefix) for key in bound}
            if set(renamed.values()) <= checkpoint:
                return renamed

        return {key: key for key in bound}

    def load(self, read: Callable[[str], np.ndarray], available: Iterable[str]) -> list[str]:
        """
        Fill every bound parameter from the checkpoint.

        Tensors are handled one at a time, so the peak cost is one tensor rather than a second copy
        of the whole checkpoint. Each is cast to its parameter's dtype, which the layer fixed from
        ``floatX`` when it built it and loading cannot change.

        A bound parameter the checkpoint cannot fill raises before anything is stored -- it would
        otherwise keep its initialization, giving a wrong model that runs. A checkpoint tensor no
        parameter wants is returned instead, since every parameter still got a value. A tensor whose
        shape is wrong raises mid-load, leaving the parameters before it filled.

        Parameters
        ----------
        read : callable
            Returns the checkpoint array for a key.
        available : iterable of str
            Every key the checkpoint holds.

        Returns
        -------
        surplus : list of str
            Checkpoint keys no parameter loads from, sorted.
        """
        checkpoint = set(available)
        names = self._checkpoint_names(checkpoint)

        missing = sorted(set(names.values()) - checkpoint)
        if missing:
            shown = ", ".join(repr(key) for key in missing[:5])
            more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
            raise ValueError(
                f"The checkpoint has no tensor for {len(missing)} bound parameter(s): {shown}{more}. "
                f"Loading anyway would leave them at their initialization, which is a wrong model "
                f"that runs."
            )

        for parameter, (bound_key, transform) in self._bindings.items():
            key = names[bound_key]
            try:
                array = read(key)
            except TypeError as error:
                raise TypeError(
                    f"{key!r} is stored in a dtype numpy cannot read ({error}). Re-save the "
                    f"checkpoint as float16 or float32."
                ) from error

            if transform is not None:
                array = transform(array)

            expected = parameter.get_value(borrow=True).shape
            if array.shape != expected:
                raise ValueError(
                    f"{key!r} holds {array.shape} but {parameter.name or parameter} needs {expected}. "
                    f"A transform is missing or wrong -- a square kernel transposed the wrong way has "
                    f"the right shape and the wrong numbers, so this check is the only one that fires."
                )
            parameter.set_value(array.astype(parameter.type.dtype, copy=False), borrow=True)

        surplus = sorted(checkpoint - set(names.values()))
        if surplus:
            _log.info(
                "%d checkpoint tensor(s) went unused: %s",
                len(surplus),
                ", ".join(surplus[:5]) + (" ..." if len(surplus) > 5 else ""),
            )
        return surplus
