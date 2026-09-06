# Regenerate the committed reference checkpoints, in an environment holding torch and transformers:
#
#     python tests/data/generate_reference_checkpoints.py
#
# Each architecture gets a directory holding the config and weights HuggingFace itself writes, plus the
# outputs transformers computes from them, so the parity test needs neither library installed.

from pathlib import Path

import numpy as np
import torch

from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    GPT2Config,
    GPT2LMHeadModel,
)

HERE = Path(__file__).parent
IDS = np.array([[1, 5, 9, 2, 0], [3, 3, 7, 2, 0]], dtype=np.int64)

CLIP_CONFIG = dict(
    hidden_size=8,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=16,
    vocab_size=50,
    hidden_act="quick_gelu",
    layer_norm_eps=1e-5,
    projection_dim=4,
    # SDXL's vintage: the config says 2 whatever the tokenizer uses, so pooling falls back to
    # taking the largest id rather than matching this value.
    eos_token_id=2,
)

GPT2_CONFIG = dict(
    n_embd=8,
    n_head=2,
    n_layer=2,
    n_positions=16,
    vocab_size=20,
    activation_function="gelu_new",
    layer_norm_epsilon=1e-5,
)


def write(name, model, outputs):
    directory = HERE / name
    # save_pretrained writes the config and the key names HuggingFace ships, which differ from the
    # state_dict's -- GPT-2's live under transformer. there and at the top level in the file.
    model.save_pretrained(directory, safe_serialization=True)
    np.savez(directory / "expected_outputs.npz", input_ids=IDS, **outputs)


def main():
    torch.manual_seed(0)
    ids = torch.from_numpy(IDS)

    text_model = CLIPTextModel(CLIPTextConfig(**CLIP_CONFIG)).eval()
    with torch.no_grad():
        result = text_model(ids, output_hidden_states=True)
    write(
        "tiny_clip",
        text_model,
        {
            "last_hidden_state": result.last_hidden_state.numpy(),
            "penultimate_hidden_state": result.hidden_states[-2].numpy(),
        },
    )

    projection_model = CLIPTextModelWithProjection(CLIPTextConfig(**CLIP_CONFIG)).eval()
    with torch.no_grad():
        result = projection_model(ids)
    write(
        "tiny_clip_with_projection",
        projection_model,
        {
            "text_embeds": result.text_embeds.numpy(),
            "last_hidden_state": result.last_hidden_state.numpy(),
        },
    )

    gpt2 = GPT2LMHeadModel(GPT2Config(**GPT2_CONFIG)).eval()
    with torch.no_grad():
        result = gpt2(ids, output_hidden_states=True)
    write(
        "tiny_gpt2",
        gpt2,
        {
            "logits": result.logits.numpy(),
            "last_hidden_state": result.hidden_states[-1].numpy(),
        },
    )


if __name__ == "__main__":
    main()
