import numpy as np
import pytensor
import pytest

from pytensor.graph.traversal import ancestors

from pytensor_ml.models import build_from_config
from pytensor_ml.pretrained import from_pretrained
from pytensor_ml.pytensorf import collect_trainable_params
from tests.conftest import REFERENCE_DATA, TINY_CLIP, TINY_CLIP_CHECKPOINT


def test_clip_builder_binds_the_checkpoint_key_of_every_parameter():
    """The bound keys are HuggingFace's own module paths, so this is what decides whether a real
    checkpoint loads. Two layers is enough to pin the numbering and the nesting."""
    _, _, keys = build_from_config(TINY_CLIP)

    per_layer = [
        "layer_norm1.weight",
        "layer_norm1.bias",
        "layer_norm2.weight",
        "layer_norm2.bias",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
    ]
    assert keys.keys() == {
        "text_model.embeddings.token_embedding.weight",
        "text_model.embeddings.position_embedding.weight",
        "text_model.final_layer_norm.weight",
        "text_model.final_layer_norm.bias",
        *(f"text_model.encoder.layers.{i}.{name}" for i in range(2) for name in per_layer),
    }


def test_the_clip_builder_binds_exactly_what_a_checkpoint_holds():
    """TINY_CLIP_CHECKPOINT is only worth trusting if it is the checkpoint HuggingFace would write,
    so the builder's bindings are checked against it rather than the other way round."""
    _, _, keys = build_from_config(TINY_CLIP)

    assert keys.keys() == set(TINY_CLIP_CHECKPOINT)
    for key, checkpoint_shape in TINY_CLIP_CHECKPOINT.items():
        # nn.Linear stores a dense weight channel-first, so every 2-D tensor but the two embedding
        # tables is the parameter's transpose.
        stored_as_built = key.endswith(("token_embedding.weight", "position_embedding.weight"))
        expected = (
            checkpoint_shape[::-1]
            if len(checkpoint_shape) == 2 and not stored_as_built
            else checkpoint_shape
        )
        assert keys.parameter_for(key).get_value().shape == expected, key


def test_clip_builder_returns_the_final_state_then_every_layer():
    """SDXL conditions on the second-to-last layer, so the per-layer states are outputs rather than
    internals a caller would have to rebuild the model to reach."""
    inputs, outputs, _ = build_from_config(TINY_CLIP)

    assert [variable.name for variable in inputs] == ["input_ids"]
    assert len(outputs) == TINY_CLIP["num_hidden_layers"] + 1
    assert outputs[0].name == "last_hidden_state"
    assert all(output.type.shape == (None, None, 8) for output in outputs)


@pytest.mark.parametrize(
    "hidden_act, expected",
    [("quick_gelu", "QuickGELU"), ("gelu", "GELU")],
    ids=["clip_l", "clip_big_g"],
)
def test_clip_builder_uses_the_configured_activation(hidden_act, expected):
    """SDXL's two encoders differ here -- CLIP-L is quick_gelu and bigG is gelu -- so hardcoding
    either one gets half the conditioning quietly wrong."""
    _, outputs, _ = build_from_config({**TINY_CLIP, "hidden_act": hidden_act})

    names = {variable.name for variable in ancestors(outputs)}
    assert expected in names


def test_clip_builder_rejects_an_intermediate_size_that_is_not_a_multiple():
    with pytest.raises(ValueError, match="not a whole multiple of hidden_size"):
        build_from_config({**TINY_CLIP, "intermediate_size": 30})


def test_clip_builder_rejects_an_unknown_activation():
    with pytest.raises(ValueError, match="hidden_act is 'silu'"):
        build_from_config({**TINY_CLIP, "hidden_act": "silu"})


def test_clip_projection_builder_adds_only_the_projection_head():
    """The projection sits at the checkpoint's top level, not under text_model, and is the sole
    difference from the base architecture."""
    _, _, base = build_from_config(TINY_CLIP)
    _, _, projected = build_from_config(
        {**TINY_CLIP, "architectures": ["CLIPTextModelWithProjection"]}
    )

    assert projected.keys() - base.keys() == {"text_projection.weight"}
    assert base.keys() - projected.keys() == set()


def test_clip_projection_builder_returns_the_pooled_embedding_first():
    inputs, outputs, _ = build_from_config(
        {**TINY_CLIP, "architectures": ["CLIPTextModelWithProjection"]}
    )

    assert outputs[0].name == "text_embeds"
    assert outputs[0].type.shape == (None, TINY_CLIP["projection_dim"])
    assert outputs[1].name == "last_hidden_state"
    assert len(outputs) == TINY_CLIP["num_hidden_layers"] + 2


@pytest.mark.parametrize(
    "eos_token_id, ids, expected_position",
    [(2, [5, 40, 9, 40], 1), (9, [5, 40, 9, 40], 2)],
    ids=["legacy_takes_the_largest_id", "matches_the_configured_id"],
)
def test_clip_pools_at_the_end_of_stream_token(eos_token_id, ids, expected_position):
    """Configs written before transformers#24773 carry eos_token_id 2 whatever the tokenizer uses, so
    they locate the token by taking the largest id instead. SDXL's are of that vintage."""
    config = {
        **TINY_CLIP,
        "architectures": ["CLIPTextModelWithProjection"],
        "eos_token_id": eos_token_id,
    }
    inputs, outputs, keys = build_from_config(config)
    rng = np.random.default_rng(0)
    for parameter in collect_trainable_params(outputs[0]):
        parameter.set_value(rng.normal(size=parameter.get_value().shape))

    pooled_output, final = pytensor.function(inputs, [outputs[0], outputs[1]])(
        np.array([ids], dtype="int32")
    )
    projection = keys.parameter_for("text_projection.weight").get_value()

    np.testing.assert_allclose(
        pooled_output[0], final[0, expected_position] @ projection, rtol=1e-5, atol=1e-5
    )


def test_a_token_id_input_is_an_integer_matrix():
    """Token ids index the embedding table, and pt.matrix defaults to floatX, so the dtype has to be
    asked for rather than inherited."""
    inputs, _, _ = build_from_config(TINY_CLIP)

    assert inputs[0].type.dtype == "int64"


def test_a_sequence_longer_than_the_position_table_raises():
    """Past the table the position gather reads out of bounds and returns whatever memory it finds,
    which is a wrong model that runs."""
    inputs, outputs, _ = build_from_config(TINY_CLIP)
    predict = pytensor.function(inputs, outputs[0])

    with pytest.raises(AssertionError, match="longer than the 16 positions"):
        predict(np.zeros((1, 17), dtype="int64"))

    assert predict(np.zeros((1, 16), dtype="int64")).shape[1] == 16


@pytest.mark.parametrize(
    "name, wanted",
    [
        ("tiny_clip", {"last_hidden_state": 0, "penultimate_hidden_state": -2}),
        ("tiny_clip_with_projection", {"text_embeds": 0, "last_hidden_state": 1}),
    ],
)
def test_a_loaded_encoder_computes_what_transformers_computes(name, wanted):
    """Every binding, transpose and layer here can be individually plausible and still produce a
    different model, and only the numbers say otherwise. The checkpoints under tests/data are the
    files HuggingFace itself writes -- see the script beside them to regenerate."""
    directory = REFERENCE_DATA / name
    expected = np.load(directory / "expected_outputs.npz")

    inputs, outputs = from_pretrained(directory)
    computed = pytensor.function(inputs, [outputs[index] for index in wanted.values()])(
        expected["input_ids"]
    )

    for key, result in zip(wanted, computed, strict=True):
        np.testing.assert_allclose(result, expected[key], rtol=1e-4, atol=1e-5, err_msg=key)
