import pytensor.tensor as pt

from pytensor.graph.basic import Variable
from pytensor.raise_op import Assert

from pytensor_ml.activations import GELU, Activation, ReLU
from pytensor_ml.layers import Embedding, LayerNorm, TransformerBlock
from pytensor_ml.models.binding import bind_layer_norm, bind_linear
from pytensor_ml.models.keys import KeyMap
from pytensor_ml.models.registry import register_builder

# Config flags that alter the attention arithmetic. A checkpoint setting one differently would load
# cleanly and return wrong numbers, which is the failure KeyMap.load refuses for weights.
_ASSUMED_CONFIG = {
    "scale_attn_weights": True,
    "scale_attn_by_inverse_layer_idx": False,
    "reorder_and_upcast_attn": False,
}

_ACTIVATIONS: dict[str, Activation] = {
    "gelu_new": GELU(approximate=True),
    "gelu": GELU(approximate=False),
    "relu": ReLU(),
}


@register_builder("GPT2LMHeadModel")
def build_gpt2(config: dict, keys: KeyMap) -> tuple[list[Variable], list[Variable]]:
    """
    Build GPT-2 and record where each parameter loads from.

    Parameters
    ----------
    config : dict
        Parsed ``config.json`` of a ``GPT2LMHeadModel``.
    keys : KeyMap
        Filled with the checkpoint key of every parameter built.

    Returns
    -------
    data_inputs : list of Variable
        The ``input_ids`` placeholder, of shape ``(batch, sequence)``.
    outputs : list of Variable
        The vocabulary logits, the final hidden state, then one entry per layer.
    """
    width = config["n_embd"]
    activation_name = config.get("activation_function", "gelu_new")
    activation = _ACTIVATIONS.get(activation_name)
    if activation is None:
        raise ValueError(
            f"GPT-2's activation_function is {activation_name!r}, which has no activation here. "
            f"Known: {sorted(_ACTIVATIONS)}."
        )

    hidden_dim = config.get("n_inner") or 4 * width
    mlp_ratio, remainder = divmod(hidden_dim, width)
    if remainder:
        raise ValueError(
            f"GPT-2's n_inner ({hidden_dim}) is not a whole multiple of n_embd ({width}), which "
            f"TransformerBlock cannot express."
        )

    for key, assumed in _ASSUMED_CONFIG.items():
        if config.get(key, assumed) != assumed:
            raise ValueError(
                f"GPT-2's {key} is {config[key]!r}, which changes the attention arithmetic and is "
                f"not implemented here. Loading anyway would give a wrong model that runs."
            )

    ids = pt.matrix("input_ids", dtype="int64")

    token_embedding = Embedding("wte", n_embeddings=config["vocab_size"], n_features=width)
    position_embedding = Embedding("wpe", n_embeddings=config["n_positions"], n_features=width)
    blocks = [
        TransformerBlock(
            f"h_{i}",
            d_model=width,
            n_head=config["n_head"],
            mlp_ratio=mlp_ratio,
            activation=activation,
            is_causal=True,
            fused_qkv=True,
            epsilon=config["layer_norm_epsilon"],
        )
        for i in range(config["n_layer"])
    ]
    final_layer_norm = LayerNorm("ln_f", n_in=width, epsilon=config["layer_norm_epsilon"])

    keys.bind(token_embedding.W, "wte.weight")
    keys.bind(position_embedding.W, "wpe.weight")
    bind_layer_norm(keys, final_layer_norm, "ln_f")
    with keys.scope("h"):
        for i, block in enumerate(blocks):
            with keys.scope(str(i)):
                _bind_block(keys, block)

    # Past the position table the gather reads out of bounds and returns whatever memory it finds,
    # so the length is checked rather than trusted.
    n_positions = Assert(
        f"input_ids is longer than the {config['n_positions']} positions this checkpoint was "
        f"trained with."
    )(ids.shape[1], ids.shape[1] <= config["n_positions"])
    hidden = token_embedding(ids) + position_embedding(pt.arange(n_positions))

    hidden_states = []
    for block in blocks:
        hidden = block(hidden)
        hidden_states.append(hidden)

    final = final_layer_norm(hidden)
    final.name = "last_hidden_state"

    # The language-modelling head is the token embedding transposed, which is why the checkpoint
    # carries no lm_head weight of its own.
    logits = final @ token_embedding.W.T
    logits.name = "logits"
    return [ids], [logits, final, *hidden_states]


def _bind_block(keys: KeyMap, block: TransformerBlock) -> None:
    # GPT-2's Conv1D already stores (in, out), so nothing here transposes.
    bind_layer_norm(keys, block.norm1, "ln_1")
    bind_layer_norm(keys, block.norm2, "ln_2")
    bind_linear(keys, block.attn.qkv_proj, "attn", "c_attn")
    bind_linear(keys, block.attn.out_proj, "attn", "c_proj")
    bind_linear(keys, block.ff.fc_in, "mlp", "c_fc")
    bind_linear(keys, block.ff.fc_out, "mlp", "c_proj")
