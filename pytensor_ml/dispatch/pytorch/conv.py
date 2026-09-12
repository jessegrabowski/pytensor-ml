import torch.nn.functional as F
import torch.nn.grad as G

from pytensor.link.pytorch.dispatch import pytorch_funcify

from pytensor_ml.layers.conv import ConvLayer, ConvLayerGrad

_CONVOLUTIONS = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
_INPUT_GRADIENTS = {1: G.conv1d_input, 2: G.conv2d_input, 3: G.conv3d_input}
_KERNEL_GRADIENTS = {1: G.conv1d_weight, 2: G.conv2d_weight, 3: G.conv3d_weight}


def _convolution(op):
    """The convolution both dispatches here run, so the gradient differentiates what the forward does.

    Torch is the one backend whose layouts disagree with ours at both ends: it takes the activation
    channels-first and the kernel output-channels-first, where we store both the other way round. The
    kernel permute is negligible, but the two activation moves are on the largest tensor in the graph.
    """
    n_spatial = len(op.kernel_size)
    if n_spatial not in _CONVOLUTIONS:
        raise NotImplementedError(f"Torch has no convolution over {n_spatial} spatial axes.")
    convolution = _CONVOLUTIONS[n_spatial]
    stride, dilation = op.stride, op.dilation

    def convolve(X, W):
        # Padding is a `pt.pad` node ahead of the op, so what arrives here is already padded.
        out = convolution(
            X.movedim(-1, 1),
            W.permute(n_spatial + 1, n_spatial, *range(n_spatial)),
            stride=stride,
            dilation=dilation,
        )
        return out.movedim(1, -1)

    return convolve


@pytorch_funcify.register(ConvLayer)
def pytorch_funcify_ConvLayer(op, node=None, **kwargs):
    """Dispatch the convolution marker to ``torch.nn.functional.conv{1,2,3}d``."""
    convolve = _convolution(op)

    def conv(X, W, *bias):
        out = convolve(X, W)
        return out + bias[0] if bias else out

    return conv


@pytorch_funcify.register(ConvLayerGrad)
def pytorch_funcify_ConvLayerGrad(op, node=None, **kwargs):
    """Dispatch each gradient to the convolution backward torch already ships.

    Differentiating our own forward call would work, but it puts an ``autograd.grad`` inside a region
    the caller may be compiling, and inductor cannot always resolve shapes through that. These take
    the same kernels autograd would have reached, without the nested tape.
    """
    n_spatial = len(op.kernel_size)
    if n_spatial not in _INPUT_GRADIENTS:
        raise NotImplementedError(
            f"Torch has no convolution gradient over {n_spatial} spatial axes."
        )
    grad_input = _INPUT_GRADIENTS[n_spatial]
    grad_kernel = _KERNEL_GRADIENTS[n_spatial]
    stride, dilation = op.stride, op.dilation
    compute_dX, compute_dW = op.compute_dX, op.compute_dW
    # Our kernel runs (taps, in, out); torch's runs (out, in, taps), and back again for its gradient.
    to_torch_kernel = (n_spatial + 1, n_spatial, *range(n_spatial))
    from_torch_kernel = (*range(2, 2 + n_spatial), 1, 0)

    def conv_grad(X, W, cotangent):
        # Padding is a `pt.pad` node ahead of the op, so what arrives here is already padded.
        X_channels_first = X.movedim(-1, 1)
        W_channels_first = W.permute(*to_torch_kernel)
        cotangent_channels_first = cotangent.movedim(-1, 1)

        gradients = []
        if compute_dX:
            dX = grad_input(
                X_channels_first.shape,
                W_channels_first,
                cotangent_channels_first,
                stride=stride,
                dilation=dilation,
            )
            gradients.append(dX.movedim(1, -1))
        if compute_dW:
            dW = grad_kernel(
                X_channels_first,
                W_channels_first.shape,
                cotangent_channels_first,
                stride=stride,
                dilation=dilation,
            )
            gradients.append(dW.permute(*from_torch_kernel))

        # An op with one output is dispatched to a function returning that output, not a list of one.
        return tuple(gradients) if len(gradients) > 1 else gradients[0]

    return conv_grad
