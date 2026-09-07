import numpy as np
import pytensor
import pytest

from pytensor_ml.models import build_from_config
from pytensor_ml.pretrained import from_pretrained
from pytensor_ml.pytensorf import collect_trainable_params
from tests.conftest import REFERENCE_DATA

TINY_GPT2 = {
    "architectures": ["GPT2LMHeadModel"],
    "n_embd": 8,
    "n_head": 2,
    "n_layer": 2,
    "n_positions": 16,
    "vocab_size": 20,
    "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-5,
    "n_inner": None,
}


def test_gpt2_builder_binds_one_key_for_the_fused_attention():
    """GPT-2 stores q, k and v in a single c_attn tensor, so the graph must own one weight there
    rather than three. Its Conv1D already stores (in, out), so nothing transposes."""
    _, _, keys = build_from_config(TINY_GPT2)

    per_layer = [
        "ln_1.weight",
        "ln_1.bias",
        "ln_2.weight",
        "ln_2.bias",
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    ]
    assert keys.keys() == {
        "wte.weight",
        "wpe.weight",
        "ln_f.weight",
        "ln_f.bias",
        *(f"h.{i}.{name}" for i in range(2) for name in per_layer),
    }
    assert keys.parameter_for("h.0.attn.c_attn.weight").get_value().shape == (8, 24)


def test_gpt2_builder_ties_the_head_to_the_token_embedding():
    """The checkpoint carries no lm_head weight, so the head has to reuse wte rather than bind a
    parameter the file cannot fill."""
    inputs, outputs, keys = build_from_config(TINY_GPT2)
    assert not any("lm_head" in key for key in keys.keys())

    rng = np.random.default_rng(0)
    for parameter in collect_trainable_params(outputs[0]):
        parameter.set_value(rng.normal(size=parameter.get_value().shape))

    logits, final = pytensor.function(inputs, [outputs[0], outputs[1]])(
        np.array([[1, 2, 3]], dtype="int32")
    )
    token_embedding = keys.parameter_for("wte.weight").get_value()

    np.testing.assert_allclose(logits, final @ token_embedding.T, rtol=1e-5, atol=1e-5)


def test_gpt2_builder_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="activation_function is 'swiglu'"):
        build_from_config({**TINY_GPT2, "activation_function": "swiglu"})


@pytest.mark.parametrize(
    "flag, value",
    [
        ("scale_attn_weights", False),
        ("scale_attn_by_inverse_layer_idx", True),
        ("reorder_and_upcast_attn", True),
    ],
)
def test_gpt2_builder_rejects_a_config_that_changes_the_attention_arithmetic(flag, value):
    """These flags load cleanly and return wrong numbers, which is the one failure the key map cannot
    catch for itself."""
    with pytest.raises(ValueError, match=flag):
        build_from_config({**TINY_GPT2, flag: value})


def test_a_token_id_input_is_an_integer_matrix():
    """Token ids index the embedding table, and pt.matrix defaults to floatX, so the dtype has to be
    asked for rather than inherited."""
    inputs, _, _ = build_from_config(TINY_GPT2)

    assert inputs[0].type.dtype == "int64"


def test_a_sequence_longer_than_the_position_table_raises():
    """Past the table the position gather reads out of bounds and returns whatever memory it finds,
    which is a wrong model that runs."""
    inputs, outputs, _ = build_from_config(TINY_GPT2)
    predict = pytensor.function(inputs, outputs[0])

    with pytest.raises(AssertionError, match="longer than the 16 positions"):
        predict(np.zeros((1, 17), dtype="int64"))

    assert predict(np.zeros((1, 16), dtype="int64")).shape[1] == 16


def test_a_loaded_model_computes_what_transformers_computes():
    """Every binding, the fused attention split and the tied head can be individually plausible and
    still produce a different model, and only the numbers say otherwise. The checkpoint under
    tests/data is the file HuggingFace itself writes -- see the script beside it to regenerate."""
    directory = REFERENCE_DATA / "tiny_gpt2"
    expected = np.load(directory / "expected_outputs.npz")

    inputs, outputs = from_pretrained(directory)
    logits, final = pytensor.function(inputs, [outputs[0], outputs[1]])(expected["input_ids"])

    np.testing.assert_allclose(logits, expected["logits"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(final, expected["last_hidden_state"], rtol=1e-4, atol=1e-5)
