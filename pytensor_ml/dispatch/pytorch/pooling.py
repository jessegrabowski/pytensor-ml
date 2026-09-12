import torch
import torch.nn.functional as F

from pytensor.link.pytorch.dispatch import pytorch_funcify

from pytensor_ml.layers.conv import PoolLayer, PoolLayerGrad

_MAX_POOLS = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}
_AVERAGE_POOLS = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}


def _pooling(op):
    """The pooling both dispatches here run, so the gradient differentiates what the forward does."""
    n_spatial = len(op.kernel_size)
    dilated = any(spacing != 1 for spacing in op.dilation)
    pools = _MAX_POOLS if op.reduction == "max" else _AVERAGE_POOLS
    if n_spatial not in pools:
        raise NotImplementedError(
            f"Torch has no pooling over {n_spatial} spatial axes; it goes up to three."
        )
    if dilated and op.reduction == "mean":
        raise NotImplementedError(
            f"Torch averages over adjacent positions only, so it cannot take the dilation "
            f"{op.dilation}. Pool without dilation, or run this graph on jax or the default backend."
        )
    pool = pools[n_spatial]
    extra = {"dilation": op.dilation} if op.reduction == "max" else {}

    def pooled(X):
        # Torch pools channels-first where we store channels last, so the activation is moved either
        # way around the call. Padding is a `pt.pad` node ahead of the op.
        channels_first = X.permute(0, n_spatial + 1, *range(1, n_spatial + 1))
        out = pool(channels_first, op.kernel_size, op.stride, **extra)
        return out.permute(0, *range(2, n_spatial + 2), 1)

    return pooled


@pytorch_funcify.register(PoolLayer)
def pytorch_funcify_PoolLayer(op, node=None, **kwargs):
    """Dispatch the pooling marker to ``torch.nn.functional.{max,avg}_pool{1,2,3}d``."""
    return _pooling(op)


@pytorch_funcify.register(PoolLayerGrad)
def pytorch_funcify_PoolLayerGrad(op, node=None, **kwargs):
    """Let torch differentiate its own pooling rather than spelling the routing out."""
    pool = _pooling(op)

    def pool_grad(X, cotangent):
        # An op with one output is dispatched to a function returning that output, not a list of one.
        X = X.detach().requires_grad_(True)
        (gradient,) = torch.autograd.grad(pool(X), X, grad_outputs=cotangent)
        return gradient

    return pool_grad
