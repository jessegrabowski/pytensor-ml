import json
import logging

from pathlib import Path

import numpy as np
import pytensor.tensor as pt
import pytest

from safetensors.numpy import save_file

from pytensor_ml.activations import ReLU
from pytensor_ml.layers import BatchNorm, Dropout, Linear, Sequential
from pytensor_ml.state import fans, initializer

REFERENCE_DATA = Path(__file__).parent / "data"


@initializer
def he_normal(rng, shape):
    """A fan-scaled draw, written the way a user would: the fans come from `fans(shape)` rather than being
    handed over. Takes no parameters at all, so what a config records of it is the class alone."""
    fan_in, _ = fans(shape)
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)


@initializer
def constant(rng, shape, value):
    """Fill every element with ``value``. A draw no other initializer produces, so a parameter holding it
    says which initializer reached it. Defined here rather than inline so it also survives a round trip
    through a saved config, which a locally defined one cannot."""
    return np.full(shape, value)


@pytest.fixture
def simple_network():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", n_in=64, n_out=32),
        ReLU(),
        Linear("fc2", n_in=32, n_out=10),
    )
    y = network(X)
    return X, y


@pytest.fixture
def network_with_batchnorm():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", n_in=64, n_out=32),
        BatchNorm("bn1", n_in=32),
        ReLU(),
        Linear("fc2", n_in=32, n_out=10),
    )
    y = network(X)
    return X, y


@pytest.fixture
def network_with_dropout():
    X = pt.tensor("X", shape=(None, 64))
    network = Sequential(
        Linear("fc1", n_in=64, n_out=32),
        Dropout(p=0.5),
        ReLU(),
        Linear("fc2", n_in=32, n_out=10),
    )
    y = network(X)
    return X, y


@pytest.fixture(autouse=True)
def fail_on_swallowed_rewrite_errors(caplog):
    """Fail if a node rewriter raised while a test was rewriting a graph.

    Pytensor catches exceptions from a node rewriter, reports them through ``logger.error``, and leaves
    the graph untouched. Nothing else notices: ``filterwarnings = ["error"]`` only sees warnings, and a
    rewrite that crashes on the first node it touches looks exactly like one that correctly declined to
    match -- so a scan keeps the dropout an inference graph is supposed to drop, and the test still
    passes.
    """
    yield

    failures = [
        record.getMessage()
        for record in caplog.get_records("call")
        if record.name.startswith("pytensor.graph.rewriting") and record.levelno >= logging.ERROR
    ]

    assert not failures, "a node rewriter raised and pytensor swallowed it:\n" + "\n".join(failures)


TINY_CLIP = {
    "architectures": ["CLIPTextModel"],
    "hidden_size": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 16,
    "vocab_size": 50,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "projection_dim": 4,
}

# The key names and checkpoint shapes HuggingFace writes for a two-layer CLIP of TINY_CLIP's size,
# spelled out rather than read back from the builder. A fixture derived from the builder cannot fail
# on a wrong key, a wrong scope or a missing transpose, which is most of what the loader can get
# wrong.
TINY_CLIP_CHECKPOINT = {
    "text_model.embeddings.token_embedding.weight": (50, 8),
    "text_model.embeddings.position_embedding.weight": (16, 8),
    "text_model.final_layer_norm.weight": (8,),
    "text_model.final_layer_norm.bias": (8,),
    **{
        f"text_model.encoder.layers.{layer}.{name}": shape
        for layer in range(2)
        for name, shape in {
            "layer_norm1.weight": (8,),
            "layer_norm1.bias": (8,),
            "layer_norm2.weight": (8,),
            "layer_norm2.bias": (8,),
            "self_attn.q_proj.weight": (8, 8),
            "self_attn.q_proj.bias": (8,),
            "self_attn.k_proj.weight": (8, 8),
            "self_attn.k_proj.bias": (8,),
            "self_attn.v_proj.weight": (8, 8),
            "self_attn.v_proj.bias": (8,),
            "self_attn.out_proj.weight": (8, 8),
            "self_attn.out_proj.bias": (8,),
            "mlp.fc1.weight": (32, 8),
            "mlp.fc1.bias": (32,),
            "mlp.fc2.weight": (8, 32),
            "mlp.fc2.bias": (8,),
        }.items()
    },
}


def write_huggingface_component(directory, config, tensors, filename):
    """A HuggingFace component directory: a config and one safetensors file."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    save_file({key: np.asarray(value) for key, value in tensors.items()}, directory / filename)
    return directory


def tiny_clip_tensors():
    rng = np.random.default_rng(0)
    return {
        key: rng.normal(size=shape).astype("float16") for key, shape in TINY_CLIP_CHECKPOINT.items()
    }
