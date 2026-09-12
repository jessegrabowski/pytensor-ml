from typing import NamedTuple

import numpy as np

from pytensor.link.numba.cache import compile_numba_function_src
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    default_hash_key_from_props,
    register_funcify_and_cache_key,
    register_funcify_default_op_cache_key,
)
from pytensor.link.numba.dispatch.string_codegen import (
    CODE_TOKEN,
    build_source_code,
    create_tuple_string,
)

from pytensor_ml.layers.conv import Col2Im, Im2Col, PoolLayer, PoolLayerGrad

# The generated kernels are compiled against this namespace rather than the module's, so the only
# name they can reach is spelled out rather than being whatever the module happens to import.
_KERNEL_GLOBALS = {"np": np}

_Source = list[str | CODE_TOKEN]


def _window_position(op, axis: int) -> str:
    """Where tap ``t{axis}`` of window ``w{axis}`` sits along one spatial axis."""
    return f"w{axis} * {op.stride[axis]} + t{axis} * {op.dilation[axis]}"


def _window_loops(window_counts: list[str]) -> _Source:
    """Open the batch loop and one loop per spatial axis over the windows that fit along it."""
    lines: _Source = ["for b in range(batch):", CODE_TOKEN.INDENT]
    for axis, count in enumerate(window_counts):
        lines += [f"for w{axis} in range({count}):", CODE_TOKEN.INDENT]
    return lines


def _tap_loops(op) -> _Source:
    """Open one loop per spatial axis over the taps of a window, each extent written in as a literal."""
    lines: _Source = []
    for axis, taps in enumerate(op.kernel_size):
        lines += [f"for t{axis} in range({taps}):", CODE_TOKEN.INDENT]
    return lines


def _window_counts(op, extents: list[str]) -> tuple[_Source, list[str]]:
    """
    Count the windows that fit along each spatial axis.

    Returns
    -------
    lines : list of str
        One assignment per spatial axis.
    names : list of str
        The variable each assignment binds.
    """
    lines: _Source = []
    names = []
    for axis, (taps, spacing) in enumerate(zip(op.kernel_size, op.dilation)):
        span = spacing * (taps - 1) + 1
        lines.append(f"windows_{axis} = ({extents[axis]} - {span}) // {op.stride[axis]} + 1")
        names.append(f"windows_{axis}")
    return lines, names


@register_funcify_default_op_cache_key(Im2Col)
def numba_funcify_Im2Col(op, node=None, **kwargs):
    """
    Gather every window a kernel visits, one channel at a time.

    The loop nest is generated rather than walked: the spatial rank is a property of the op, so every
    window count, stride and dilation reaches the compiler as a constant it can fold into the index
    arithmetic.
    """
    n_spatial = len(op.kernel_size)
    extents = [f"X.shape[{1 + axis}]" for axis in range(n_spatial)]
    windows = [f"w{axis}" for axis in range(n_spatial)]
    taps = [f"t{axis}" for axis in range(n_spatial)]

    source: _Source = [
        "def im2col(X):",
        CODE_TOKEN.INDENT,
        "batch = X.shape[0]",
        f"channels = X.shape[{1 + n_spatial}]",
    ]
    count_lines, counts = _window_counts(op, extents)
    source += count_lines
    out_shape = create_tuple_string(
        ["batch", *counts, *(str(extent) for extent in op.kernel_size), "channels"]
    )
    source.append(f"out = np.empty({out_shape}, dtype=X.dtype)")
    source += _window_loops(counts) + _tap_loops(op)
    # One channel at a time: a row copy would memcpy `in_channels` elements, too few to pay for itself
    source += ["for c in range(channels):", CODE_TOKEN.INDENT]
    read = ", ".join(_window_position(op, axis) for axis in range(n_spatial))
    source.append(f"out[b, {', '.join(windows)}, {', '.join(taps)}, c] = X[b, {read}, c]")
    source += [CODE_TOKEN.DEDENT] * (2 * n_spatial + 2)
    source.append("return out")

    kernel = numba_basic.numba_njit(
        compile_numba_function_src(
            build_source_code(source), function_name="im2col", global_env=_KERNEL_GLOBALS
        )
    )
    # Bump whenever the generated source changes: the default key covers the op's props, not this
    cache_version = 1
    return kernel, cache_version


@register_funcify_default_op_cache_key(Col2Im)
def numba_funcify_Col2Im(op, node=None, **kwargs):
    """
    Scatter every window's contribution back where it was gathered from, one channel at a time.

    Windows overlap, so a position several of them reach accumulates in loop order.
    """
    n_spatial = len(op.kernel_size)
    arguments = [f"extent_{axis}" for axis in range(n_spatial)]
    extents = [f"{name}_item" for name in arguments]
    windows = [f"w{axis}" for axis in range(n_spatial)]
    taps = [f"t{axis}" for axis in range(n_spatial)]

    source: _Source = [
        f"def col2im(patches, {', '.join(arguments)}):",
        CODE_TOKEN.INDENT,
        "batch = patches.shape[0]",
        f"channels = patches.shape[{1 + 2 * n_spatial}]",
        # Shape inputs arrive as zero-dimensional arrays, as they do everywhere else in numba-land
        *(f"{item} = {name}.item()" for item, name in zip(extents, arguments)),
    ]
    count_lines, counts = _window_counts(op, extents)
    source += count_lines
    out_shape = create_tuple_string(["batch", *extents, "channels"])
    source.append(f"out = np.zeros({out_shape}, dtype=patches.dtype)")
    source += _window_loops(counts) + _tap_loops(op)
    # Scalar accumulation rather than a slice `+=`, which allocates a temporary per window visit
    source += ["for c in range(channels):", CODE_TOKEN.INDENT]
    write = ", ".join(_window_position(op, axis) for axis in range(n_spatial))
    source.append(f"out[b, {write}, c] += patches[b, {', '.join(windows)}, {', '.join(taps)}, c]")
    source += [CODE_TOKEN.DEDENT] * (2 * n_spatial + 2)
    source.append("return out")

    kernel = numba_basic.numba_njit(
        compile_numba_function_src(
            build_source_code(source), function_name="col2im", global_env=_KERNEL_GLOBALS
        )
    )
    # Bump whenever the generated source changes: the default key covers the op's props, not this
    cache_version = 1
    return kernel, cache_version


class _Accumulator(NamedTuple):
    """How one reduction seeds an accumulator, folds one tap into it, and finishes it."""

    seed: str
    fold: _Source
    finish: str


# Max seeds from the window's own first tap rather than from an infinity, so the kernel never depends on
# the input's dtype having one.
_ACCUMULATORS = {
    "max": _Accumulator(
        seed="acc = X[b, {first_tap}, c]",
        fold=["if value > acc:", CODE_TOKEN.INDENT, "acc = value", CODE_TOKEN.DEDENT],
        finish="acc",
    ),
    "mean": _Accumulator(
        seed="acc = X.dtype.type(0)",
        fold=["acc += value"],
        finish="acc / {taps}",
    ),
}


# Registered against the keyed dispatcher rather than the default-key helper: a `PoolLayer` is an
# `OpFromGraph`, and the generic OpFromGraph entry is the more specific match for the dispatcher the
# linker actually calls, so it would otherwise win and inline the inner graph.
@register_funcify_and_cache_key(PoolLayer)
def numba_funcify_PoolLayer(op, node=None, **kwargs):
    """
    Reduce each window in place, without materializing the windows.

    The inner graph gathers every window and then reduces the gathered copy, which writes and re-reads a
    buffer several times the input. Reading each window where it lies costs one pass and no buffer.
    """
    n_spatial = len(op.kernel_size)
    extents = [f"X.shape[{1 + axis}]" for axis in range(n_spatial)]
    windows = [f"w{axis}" for axis in range(n_spatial)]
    accumulator = _ACCUMULATORS[op.reduction]
    read = ", ".join(_window_position(op, axis) for axis in range(n_spatial))
    first_tap = ", ".join(f"w{axis} * {op.stride[axis]}" for axis in range(n_spatial))

    source: _Source = [
        "def pool(X):",
        CODE_TOKEN.INDENT,
        "batch = X.shape[0]",
        f"channels = X.shape[{1 + n_spatial}]",
    ]
    count_lines, counts = _window_counts(op, extents)
    source += count_lines
    out_shape = create_tuple_string(["batch", *counts, "channels"])
    source.append(f"out = np.empty({out_shape}, dtype=X.dtype)")

    source += _window_loops(counts)
    source += ["for c in range(channels):", CODE_TOKEN.INDENT]
    source.append(accumulator.seed.format(first_tap=first_tap))
    source += _tap_loops(op)
    source.append(f"value = X[b, {read}, c]")
    source += accumulator.fold
    source += [CODE_TOKEN.DEDENT] * n_spatial
    reduced = accumulator.finish.format(taps=int(np.prod(op.kernel_size)))
    source.append(f"out[b, {', '.join(windows)}, c] = {reduced}")
    source += [CODE_TOKEN.DEDENT] * (n_spatial + 2)
    source.append("return out")

    kernel = numba_basic.numba_njit(
        compile_numba_function_src(
            build_source_code(source), function_name="pool", global_env=_KERNEL_GLOBALS
        )
    )
    cache_version = 1
    return kernel, default_hash_key_from_props(op, cache_version=cache_version)


@register_funcify_and_cache_key(PoolLayerGrad)
def numba_funcify_PoolLayerGrad(op, node=None, **kwargs):
    """
    Route each window's cotangent straight to the positions that earned it.

    Differentiating the forward instead gathers every window, reduces it a second time to find where its
    maximum was, and scatters -- three passes over a buffer larger than the input, to move one value per
    window.
    """
    n_spatial = len(op.kernel_size)
    extents = [f"X.shape[{1 + axis}]" for axis in range(n_spatial)]
    windows = [f"w{axis}" for axis in range(n_spatial)]
    read = ", ".join(_window_position(op, axis) for axis in range(n_spatial))
    first_tap = ", ".join(f"w{axis} * {op.stride[axis]}" for axis in range(n_spatial))
    cotangent = f"cotangent[b, {', '.join(windows)}, c]"

    source: _Source = [
        "def pool_grad(X, cotangent):",
        CODE_TOKEN.INDENT,
        "batch = X.shape[0]",
        f"channels = X.shape[{1 + n_spatial}]",
    ]
    count_lines, counts = _window_counts(op, extents)
    source += count_lines
    source.append("out = np.zeros(X.shape, dtype=X.dtype)")
    source += _window_loops(counts)
    source += ["for c in range(channels):", CODE_TOKEN.INDENT]

    if op.reduction == "max":
        # Locate the winner first, then credit it once. Ties go to the earliest tap, matching the
        # `>` the forward kernel uses to keep its own running maximum.
        source.append(f"best = X[b, {first_tap}, c]")
        source += [f"best_{axis} = 0" for axis in range(n_spatial)]
        source += _tap_loops(op)
        source.append(f"value = X[b, {read}, c]")
        source += ["if value > best:", CODE_TOKEN.INDENT, "best = value"]
        source += [f"best_{axis} = t{axis}" for axis in range(n_spatial)]
        source.append(CODE_TOKEN.DEDENT)
        source += [CODE_TOKEN.DEDENT] * n_spatial
        winner = ", ".join(
            f"w{axis} * {op.stride[axis]} + best_{axis} * {op.dilation[axis]}"
            for axis in range(n_spatial)
        )
        source.append(f"out[b, {winner}, c] += {cotangent}")
    else:
        taps = int(np.prod(op.kernel_size))
        source.append(f"share = {cotangent} / {taps}")
        source += _tap_loops(op)
        source.append(f"out[b, {read}, c] += share")
        source += [CODE_TOKEN.DEDENT] * n_spatial

    source += [CODE_TOKEN.DEDENT] * (n_spatial + 2)
    source.append("return out")

    kernel = numba_basic.numba_njit(
        compile_numba_function_src(
            build_source_code(source), function_name="pool_grad", global_env=_KERNEL_GLOBALS
        )
    )
    cache_version = 1
    return kernel, default_hash_key_from_props(op, cache_version=cache_version)
