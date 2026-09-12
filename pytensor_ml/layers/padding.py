from collections.abc import Sequence

import pytensor.tensor as pt

from pytensor.tensor.pad import PadMode
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.base import Layer, _check_input_rank, _resolve_layer_name


def _pad_spatial(
    X: TensorVariable,
    padding: tuple[tuple[int, int], ...],
    mode: PadMode,
    constant_value: float = 0.0,
) -> TensorVariable:
    """Pad the spatial axes of ``(batch, *spatial, channels)``, leaving batch and channels alone."""
    if not any(before or after for before, after in padding):
        return X
    pad_width = [(0, 0), *padding, (0, 0)]
    # Spelled out per branch rather than unpacked from a dict: only `constant` takes a fill value, and
    # `pad` is overloaded per mode, so a `**kwargs` call cannot be matched against those overloads.
    if mode == "constant":
        return pt.pad(X, pad_width, mode="constant", constant_values=constant_value)
    return pt.pad(X, pad_width, mode=mode)


def _resolve_pad_amounts(
    padding: int | Sequence[int] | Sequence[tuple[int, int]], n_spatial: int
) -> tuple[tuple[int, int], ...]:
    """
    Turn a padding argument into an explicit ``(before, after)`` pair per spatial axis.

    Parameters
    ----------
    padding : int, sequence of int, or sequence of pair of int
        The same amount on every side, one amount per axis applied to both of its sides, or an
        explicit ``(before, after)`` for each axis. Axes are given in the order they appear in the
        input, so a 2-D layer reads ``((top, bottom), (left, right))``. Over one spatial axis a bare
        ``(before, after)`` is taken as that axis's two ends.
    n_spatial : int
        Number of spatial axes the layer pads.
    """
    amounts = _as_pairs(padding, n_spatial)
    for axis, pair in enumerate(amounts):
        if any(amount < 0 for amount in pair):
            raise ValueError(
                f"Padding adds elements, so it cannot be negative; spatial axis {axis} got {pair}. "
                "To use less of the input than it has, take a slice of it instead."
            )
    return amounts


def _as_pairs(
    padding: int | Sequence[int] | Sequence[tuple[int, int]], n_spatial: int
) -> tuple[tuple[int, int], ...]:
    """Broadcast whichever shape the argument came in as to one ``(before, after)`` pair per axis."""
    if isinstance(padding, int):
        return ((padding, padding),) * n_spatial

    given: list[int | tuple[int, int]] = list(padding)
    # Over a single axis a bare pair is unambiguous, and reads the way torch's does: the two
    # numbers are its two ends rather than two axes.
    if n_spatial == 1 and len(given) == 2:
        before, after = given
        if isinstance(before, int) and isinstance(after, int):
            return ((before, after),)

    if len(given) != n_spatial:
        raise ValueError(
            f"Padding over {n_spatial} spatial axes needs one amount per axis, but got "
            f"{len(given)}. Torch takes a flat sequence ordered from the last axis; this takes "
            f"one entry per axis, in the order the axes appear in the input."
        )

    amounts: list[tuple[int, int]] = []
    for axis, amount in enumerate(given):
        if isinstance(amount, int):
            amounts.append((amount, amount))
            continue
        if len(amount) != 2:
            raise ValueError(
                f"An explicit padding is a (before, after) pair, but spatial axis {axis} got "
                f"{len(amount)} values."
            )
        before, after = amount
        amounts.append((before, after))
    return tuple(amounts)


class _PadNd(Layer):
    """
    Everything padding does that does not depend on how many spatial axes it has.

    Subclasses set :attr:`n_spatial` and :attr:`pad_mode`; see :class:`ZeroPad2D` for the arguments,
    which are shared.
    """

    n_spatial: int
    pad_mode: PadMode
    constant_value: float = 0.0

    def __init__(
        self,
        name: str | None = None,
        *,
        padding: int | Sequence[int] | Sequence[tuple[int, int]] = 0,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "padding")
        self.padding = _resolve_pad_amounts(padding, self.n_spatial)

    def __call__(self, X: pt.TensorLike) -> TensorVariable:
        """
        Grow the spatial axes of ``X``, of shape ``(batch, *spatial, channels)``.

        Returns
        -------
        padded : TensorVariable
            Shape ``(batch, *padded_spatial, channels)``, with batch and channels untouched.
        """
        X = pt.as_tensor(X)
        _check_input_rank(X, self.name, self.n_spatial)

        out = _pad_spatial(X, self.padding, self.pad_mode, self.constant_value)
        out.name = f"{self.name}_output"
        return out


class _ConstantPadNd(_PadNd):
    """Padding that fills with a value the caller chooses."""

    pad_mode: PadMode = "constant"

    def __init__(
        self,
        name: str | None = None,
        *,
        padding: int | Sequence[int] | Sequence[tuple[int, int]] = 0,
        value: float = 0.0,
    ):
        super().__init__(name, padding=padding)
        self.constant_value = float(value)


class ZeroPad1D(_PadNd):
    """
    Grow a sequence with zeros on both ends.

    Takes ``(batch, time, channels)`` and returns ``(batch, padded_time, channels)``.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's output. Defaults to the class name when None.
    padding : int or pair of int, optional
        Elements added before and after the time axis, either shared by both ends or given as
        ``(before, after)``. Default is 0.

    Examples
    --------
    Pad a sequence with zeros on both ends, most often to keep a convolution's output length:

    .. code-block:: python

        from pytensor_ml.layers import Conv1D, Input, Sequential, ZeroPad1D

        X = Input("X", shape=(None, 128, 16))
        network = Sequential(
            ZeroPad1D(padding=2),
            Conv1D("conv", in_channels=16, out_channels=32, kernel_size=5),
        )

        features = network(X)
    """

    n_spatial = 1
    pad_mode: PadMode = "constant"


class ZeroPad2D(_PadNd):
    """
    Grow an image with zeros on every side.

    Takes ``(batch, height, width, channels)`` and returns ``(batch, padded_height, padded_width,
    channels)``. This is what a convolution's ``padding="same"`` does internally, exposed as a layer
    for the cases where the padding and the convolution want separate control.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's output. Defaults to the class name when None.
    padding : int, pair of int, or pair of pair of int, optional
        The same amount on every side, one amount per axis, or an explicit ``(before, after)`` for
        each. Axes read in input order, so ``((top, bottom), (left, right))`` -- not torch's flat
        last-axis-first sequence. Default is 0.

    Examples
    --------
    Surround an image with zeros so a following convolution keeps its extents. One number pads every
    side equally; a pair per axis pads each side separately:

    .. code-block:: python

        from pytensor_ml.layers import Conv2D, Input, Sequential, ZeroPad2D

        X = Input("X", shape=(None, 32, 32, 3))
        network = Sequential(
            ZeroPad2D(padding=1),
            Conv2D("conv", in_channels=3, out_channels=16, kernel_size=3),
        )

        features = network(X)
    """

    n_spatial = 2
    pad_mode: PadMode = "constant"


class ConstantPad1D(_ConstantPadNd):
    """
    Grow a sequence with a chosen value on both ends.

    Takes ``(batch, time, channels)``. See :class:`ZeroPad1D` for ``name`` and ``padding``.

    Parameters
    ----------
    value : float, optional
        What the added elements hold. Default is 0.0, which makes this :class:`ZeroPad1D`.

    Examples
    --------
    Pad a sequence with a chosen constant, for data where zero is a real value rather than absence:

    .. code-block:: python

        from pytensor_ml.layers import Conv1D, Input, Sequential, ConstantPad1D

        X = Input("X", shape=(None, 128, 16))
        network = Sequential(
            ConstantPad1D(padding=2, value=0.5),
            Conv1D("conv", in_channels=16, out_channels=32, kernel_size=5),
        )

        features = network(X)
    """

    n_spatial = 1


class ConstantPad2D(_ConstantPadNd):
    """
    Grow an image with a chosen value on every side.

    Takes ``(batch, height, width, channels)``. See :class:`ZeroPad2D` for ``name`` and ``padding``.

    Parameters
    ----------
    value : float, optional
        What the added elements hold. Default is 0.0, which makes this :class:`ZeroPad2D`.

    Examples
    --------
    Pad with a value of your choosing rather than zero, which matters when zero is a meaningful level
    in the data rather than a neutral one:

    .. code-block:: python

        from pytensor_ml.layers import Conv2D, Input, Sequential, ConstantPad2D

        X = Input("X", shape=(None, 32, 32, 3))
        network = Sequential(
            ConstantPad2D(padding=1, value=0.5),
            Conv2D("conv", in_channels=3, out_channels=16, kernel_size=3),
        )

        features = network(X)
    """

    n_spatial = 2


class ReflectionPad1D(_PadNd):
    """
    Grow a sequence by mirroring it, without repeating the end element.

    Takes ``(batch, time, channels)``. Padding a signal by reflection avoids the step change a
    constant fill introduces at the boundary, which a downstream convolution would otherwise read as
    an edge. The reflection excludes the end element itself, so ``abcd`` padded by two becomes
    ``cbabcdcb``; :class:`ReplicationPad1D` repeats it instead.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's output. Defaults to the class name when None.
    padding : int or pair of int, optional
        Elements added before and after the time axis. Padding wider than the axis keeps reflecting
        back and forth rather than failing, which is what numpy does and where torch raises.
        Default is 0.

    Examples
    --------
    Mirror a sequence across its ends, so the padded region continues the signal rather than cutting it
    to a constant:

    .. code-block:: python

        from pytensor_ml.layers import Conv1D, Input, Sequential, ReflectionPad1D

        X = Input("X", shape=(None, 128, 16))
        network = Sequential(
            ReflectionPad1D(padding=2),
            Conv1D("conv", in_channels=16, out_channels=32, kernel_size=5),
        )

        features = network(X)
    """

    n_spatial = 1
    pad_mode: PadMode = "reflect"


class ReflectionPad2D(_PadNd):
    """
    Grow an image by mirroring it, without repeating the edge row or column.

    Takes ``(batch, height, width, channels)``. See :class:`ReflectionPad1D` for what reflection
    means and :class:`ZeroPad2D` for how ``padding`` is read.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's output. Defaults to the class name when None.
    padding : int, pair of int, or pair of pair of int, optional
        Padding wider than the axis it reflects keeps reflecting back and forth rather than failing.
        Default is 0.

    Examples
    --------
    Mirror the image across its edges instead of inventing a constant, which avoids the hard border a
    zero pad introduces. The padding must be smaller than the extent it mirrors:

    .. code-block:: python

        from pytensor_ml.layers import Conv2D, Input, Sequential, ReflectionPad2D

        X = Input("X", shape=(None, 32, 32, 3))
        network = Sequential(
            ReflectionPad2D(padding=1),
            Conv2D("conv", in_channels=3, out_channels=16, kernel_size=3),
        )

        features = network(X)
    """

    n_spatial = 2
    pad_mode: PadMode = "reflect"


class ReplicationPad1D(_PadNd):
    """
    Grow a sequence by repeating its end elements.

    Takes ``(batch, time, channels)``. Unlike :class:`ReflectionPad1D` this holds the boundary value
    constant, so ``abcd`` padded by two becomes ``aaabcddd``, and unlike a constant fill it never
    introduces a value the signal did not already have.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's output. Defaults to the class name when None.
    padding : int or pair of int, optional
        Elements added before and after the time axis. Default is 0.

    Examples
    --------
    Hold the first and last values of a sequence flat across the padded region:

    .. code-block:: python

        from pytensor_ml.layers import Conv1D, Input, Sequential, ReplicationPad1D

        X = Input("X", shape=(None, 128, 16))
        network = Sequential(
            ReplicationPad1D(padding=2),
            Conv1D("conv", in_channels=16, out_channels=32, kernel_size=5),
        )

        features = network(X)
    """

    n_spatial = 1
    pad_mode: PadMode = "edge"


class ReplicationPad2D(_PadNd):
    """
    Grow an image by repeating its edge rows and columns.

    Takes ``(batch, height, width, channels)``. See :class:`ReplicationPad1D` for what replication
    means and :class:`ZeroPad2D` for how ``padding`` is read.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's output. Defaults to the class name when None.
    padding : int, pair of int, or pair of pair of int, optional
        Default is 0.

    Examples
    --------
    Repeat the edge pixel outwards. Like reflection it avoids a hard border, but it holds the boundary
    value flat rather than folding the interior back:

    .. code-block:: python

        from pytensor_ml.layers import Conv2D, Input, Sequential, ReplicationPad2D

        X = Input("X", shape=(None, 32, 32, 3))
        network = Sequential(
            ReplicationPad2D(padding=1),
            Conv2D("conv", in_channels=3, out_channels=16, kernel_size=3),
        )

        features = network(X)
    """

    n_spatial = 2
    pad_mode: PadMode = "edge"
