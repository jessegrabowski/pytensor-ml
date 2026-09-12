import mlx.core as mx
import mlx.nn as mnn

from pytensor.link.mlx.dispatch import mlx_funcify

from pytensor_ml.layers.conv import PoolLayer, PoolLayerGrad

_POOLS = {
    ("max", 1): mnn.MaxPool1d,
    ("max", 2): mnn.MaxPool2d,
    ("max", 3): mnn.MaxPool3d,
    ("mean", 1): mnn.AvgPool1d,
    ("mean", 2): mnn.AvgPool2d,
    ("mean", 3): mnn.AvgPool3d,
}


def _pooling(op):
    """The pooling both dispatches here run, so the gradient differentiates what the forward does."""
    n_spatial = len(op.kernel_size)
    if any(spacing != 1 for spacing in op.dilation):
        raise NotImplementedError(
            f"mlx pools over adjacent positions only, so it cannot take the dilation {op.dilation}. "
            "Pool without dilation, or run this graph on jax or the default backend."
        )
    if (op.reduction, n_spatial) not in _POOLS:
        raise NotImplementedError(
            f"mlx has no pooling over {n_spatial} spatial axes; it goes up to three."
        )

    # mlx pools channels-last as we store it, and takes no padding here because a `pt.pad` node ahead
    # of the op has already applied it.
    return _POOLS[(op.reduction, n_spatial)](kernel_size=op.kernel_size, stride=op.stride)


@mlx_funcify.register(PoolLayer)
def mlx_funcify_PoolLayer(op, node=None, **kwargs):
    """Dispatch the pooling marker to mlx's own pooling (a fused Metal kernel)."""
    pool = _pooling(op)

    def pooled(X):
        return pool(X)

    return pooled


@mlx_funcify.register(PoolLayerGrad)
def mlx_funcify_PoolLayerGrad(op, node=None, **kwargs):
    """Let mlx differentiate its own pooling rather than spelling the routing out."""
    pool = _pooling(op)

    def pool_grad(X, cotangent):
        # An op with one output is dispatched to a function returning that output, not a list of one.
        _, pullback = mx.vjp(pool, [X], [cotangent])
        return pullback[0]

    return pool_grad
