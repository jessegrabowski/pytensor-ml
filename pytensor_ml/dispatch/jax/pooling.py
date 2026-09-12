import math

from functools import partial

import jax
import jax.numpy as jnp

from pytensor.link.jax.dispatch import jax_funcify

from pytensor_ml.layers.conv import PoolLayer, PoolLayerGrad


def _pooling(op):
    """The pooling both dispatches here run, so the gradient differentiates what the forward does."""
    # Windows span the spatial axes only; batch and channels take a width of one and are never reduced.
    window = (1, *op.kernel_size, 1)
    strides = (1, *op.stride, 1)
    dilation = (1, *op.dilation, 1)
    taps = math.prod(op.kernel_size)

    # Padding is a `pt.pad` node ahead of the op, so what arrives here is already padded.
    reduce_window = partial(
        jax.lax.reduce_window,
        window_dimensions=window,
        window_strides=strides,
        padding="VALID",
        window_dilation=dilation,
    )

    if op.reduction == "max":

        def pool(X):
            return reduce_window(X, -jnp.inf, jax.lax.max)

    else:

        def pool(X):
            return reduce_window(X, 0.0, jax.lax.add) / taps

    return pool


@jax_funcify.register(PoolLayer)
def jax_funcify_PoolLayer(op, node=None, **kwargs):
    """Dispatch the pooling marker to ``jax.lax.reduce_window``."""
    return _pooling(op)


@jax_funcify.register(PoolLayerGrad)
def jax_funcify_PoolLayerGrad(op, node=None, **kwargs):
    """Let jax differentiate its own pooling rather than spelling the routing out."""
    pool = _pooling(op)

    def pool_grad(X, cotangent):
        # An op with one output is dispatched to a function returning that output, not a list of one.
        _, pullback = jax.vjp(pool, X)
        (gradient,) = pullback(cotangent)
        return gradient

    return pool_grad
