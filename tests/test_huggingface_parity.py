from pathlib import Path

import numpy as np
import pytensor
import pytest

from pytensor_ml.pretrained import from_pretrained

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "name, wanted",
    [
        ("tiny_clip", {"last_hidden_state": 0, "penultimate_hidden_state": -2}),
        ("tiny_clip_with_projection", {"text_embeds": 0, "last_hidden_state": 1}),
        ("tiny_gpt2", {"logits": 0, "last_hidden_state": 1}),
    ],
)
def test_a_loaded_model_computes_what_transformers_computes(name, wanted):
    """Every binding, transpose and layer in these builders can be individually plausible and still
    produce a different model, and only the numbers say otherwise. The checkpoints under tests/data
    are the files HuggingFace itself writes -- see the script beside them to regenerate."""
    directory = DATA / name
    expected = np.load(directory / "expected_outputs.npz")

    inputs, outputs = from_pretrained(directory)
    computed = pytensor.function(inputs, [outputs[index] for index in wanted.values()])(
        expected["input_ids"]
    )

    for key, result in zip(wanted, computed, strict=True):
        np.testing.assert_allclose(result, expected[key], rtol=1e-4, atol=1e-5, err_msg=key)
