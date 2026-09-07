# Regenerate the committed reference activations. diffusers is read from the clone under
# references/ rather than installed, so it goes on the path rather than into the environment:
#
#     PYTHONPATH=references/diffusers/src python tests/data/generate_reference_layers.py
#
# Each archive holds a layer's inputs, its weights under the names HuggingFace stores them, and the
# activations diffusers computes, so the test comparing against them needs neither library installed.

from pathlib import Path

import numpy as np
import torch

from diffusers.models.resnet import ResnetBlock2D

HERE = Path(__file__).parent


def channels_first(array):
    """A channel-last batch of images as torch stores it."""
    return np.moveaxis(array, -1, 1)


def resnet_block_2d():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    cases = {}
    for label, out_channels, temb_channels in [
        ("same_width", None, None),
        ("wider", 6, None),
        ("conditioned", 6, 5),
    ]:
        block = ResnetBlock2D(
            in_channels=4,
            out_channels=out_channels,
            temb_channels=temb_channels,
            groups=2,
            eps=1e-6,
            non_linearity="swish",
        ).eval()

        X = rng.normal(size=(2, 8, 8, 4)).astype("float32")
        temb = (
            None if temb_channels is None else rng.normal(size=(2, temb_channels)).astype("float32")
        )
        with torch.no_grad():
            output = block(
                torch.from_numpy(channels_first(X)),
                None if temb is None else torch.from_numpy(temb),
            )

        cases[f"{label}/X"] = X
        cases[f"{label}/output"] = np.moveaxis(output.numpy(), 1, -1)
        if temb is not None:
            cases[f"{label}/temb"] = temb
        for key, value in block.state_dict().items():
            cases[f"{label}/weights/{key}"] = value.numpy()

    np.savez(HERE / "resnet_block_2d.npz", **cases)


if __name__ == "__main__":
    resnet_block_2d()
