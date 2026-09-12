import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph.replace import vectorize_graph
from pytensor.graph.traversal import ancestors
from pytensor.tensor.extra_ops import Repeat

from pytensor_ml.layers.attention import (
    AttentionLayer,
    CausalSelfAttention,
    MultiheadAttention,
    scaled_dot_product_attention,
)

floatX = pytensor.config.floatX


@pytest.fixture(scope="module")
def rng():
    seed = sum(map(ord, "Attention Test"))
    return np.random.default_rng(seed)


def softmax_np(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def sdpa_np(q, k, v, *, is_causal=False, mask=None, scale=None):
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = np.repeat(k, n_rep, axis=1)
        v = np.repeat(v, n_rep, axis=1)
    # np.sqrt(int) is a float64 zero-dim array, not a weak scalar, and would upcast the scores.
    scale = 1.0 / np.sqrt(q.shape[-1]) if scale is None else scale
    scores = (q @ k.swapaxes(-1, -2)) * np.asarray(scale, dtype=q.dtype)
    if is_causal:
        sq, sk = q.shape[-2], k.shape[-2]
        causal = np.arange(sk)[None, :] <= np.arange(sq)[:, None] + (sk - sq)
        scores = np.where(causal, scores, -np.inf)
    if mask is not None:
        scores = scores + mask
    return softmax_np(scores) @ v


@pytest.mark.parametrize("scale", [None, 0.5], ids=["default_scale", "custom_scale"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["full", "causal"])
def test_sdpa_matches_reference(is_causal, scale, rng):
    b, n_head, seq, qk_dim, v_dim = 2, 3, 5, 4, 6
    q = rng.normal(size=(b, n_head, seq, qk_dim)).astype(floatX)
    k = rng.normal(size=(b, n_head, seq, qk_dim)).astype(floatX)
    v = rng.normal(size=(b, n_head, seq, v_dim)).astype(floatX)

    out = scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale).eval()

    assert out.shape == (b, n_head, seq, v_dim)
    np.testing.assert_allclose(out, sdpa_np(q, k, v, is_causal=is_causal, scale=scale), atol=1e-5)


def test_sdpa_grouped_query(rng):
    b, n_head, n_kv_head, seq, dim = 2, 6, 2, 4, 5
    q = rng.normal(size=(b, n_head, seq, dim)).astype(floatX)
    k = rng.normal(size=(b, n_kv_head, seq, dim)).astype(floatX)
    v = rng.normal(size=(b, n_kv_head, seq, dim)).astype(floatX)

    out = scaled_dot_product_attention(q, k, v, is_causal=True).eval()

    np.testing.assert_allclose(out, sdpa_np(q, k, v, is_causal=True), atol=1e-5)


@pytest.mark.parametrize("is_causal", [False, True], ids=["mask_only", "mask_and_causal"])
def test_sdpa_additive_mask(is_causal, rng):
    b, n_head, seq, dim = 2, 3, 5, 4
    q = rng.normal(size=(b, n_head, seq, dim)).astype(floatX)
    k = rng.normal(size=(b, n_head, seq, dim)).astype(floatX)
    v = rng.normal(size=(b, n_head, seq, dim)).astype(floatX)
    mask = np.where(rng.normal(size=(b, n_head, seq, seq)) > 0, 0.0, -1e9).astype(floatX)

    out = scaled_dot_product_attention(q, k, v, mask=mask, is_causal=is_causal).eval()

    np.testing.assert_allclose(out, sdpa_np(q, k, v, mask=mask, is_causal=is_causal), atol=1e-5)


@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "full"])
def test_causal_mask_severs_gradient_to_future(is_causal, rng):
    """Under a causal mask, output position i has exactly zero gradient w.r.t. the key and value at
    any future position j > i -- the structural statement of 'no peeking ahead'. Without the mask the
    same gradient is nonzero, since every position attends everywhere."""
    b, n_head, seq, dim = 1, 2, 5, 3
    q = pt.tensor("q", shape=(b, n_head, seq, dim))
    k = pt.tensor("k", shape=(b, n_head, seq, dim))
    v = pt.tensor("v", shape=(b, n_head, seq, dim))
    out = scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    query_pos = 1
    dk, dv = pt.grad(out[:, :, query_pos, :].sum(), [k, v])
    f = pytensor.function([q, k, v], [dk, dv])
    dk_val, dv_val = f(*(rng.normal(size=(b, n_head, seq, dim)).astype(floatX) for _ in range(3)))

    future = slice(query_pos + 1, None)
    if is_causal:
        assert np.all(dk_val[:, :, future, :] == 0)
        assert np.all(dv_val[:, :, future, :] == 0)
    else:
        assert np.abs(dv_val[:, :, future, :]).max() > 0


def test_additive_mask_severs_gradient(rng):
    """A key position masked with -inf contributes nothing, so the output's gradient w.r.t. that
    position's key and value is exactly zero while unmasked positions still receive gradient."""
    b, n_head, seq, dim = 1, 2, 4, 3
    masked_key = 2
    mask = np.zeros((b, n_head, seq, seq), dtype=floatX)
    mask[:, :, :, masked_key] = -np.inf

    q = pt.tensor("q", shape=(b, n_head, seq, dim))
    k = pt.tensor("k", shape=(b, n_head, seq, dim))
    v = pt.tensor("v", shape=(b, n_head, seq, dim))
    out = scaled_dot_product_attention(q, k, v, mask=mask)

    dk, dv = pt.grad(out.sum(), [k, v])
    f = pytensor.function([q, k, v], [dk, dv])
    dk_val, dv_val = f(*(rng.normal(size=(b, n_head, seq, dim)).astype(floatX) for _ in range(3)))

    assert np.all(dk_val[:, :, masked_key, :] == 0)
    assert np.all(dv_val[:, :, masked_key, :] == 0)
    assert np.abs(dv_val[:, :, 0, :]).max() > 0


def test_sdpa_is_marker_op(rng):
    q = pt.tensor("q", shape=(2, 2, 4, 3))
    out = scaled_dot_product_attention(q, q, q, is_causal=True, scale=0.5)
    op = out.owner.op
    assert isinstance(op, AttentionLayer)
    assert op.is_causal is True
    assert op.scale == 0.5


def test_dense_attention_has_no_repeat(rng):
    """Equal query and key/value head counts must not introduce a Repeat into the graph."""
    q = pt.tensor("q", shape=(2, 3, 4, 5))
    out = scaled_dot_product_attention(q, q, q)
    inner_ops = [node.op for node in out.owner.op.fgraph.toposort()]
    assert not any(isinstance(op, Repeat) for op in inner_ops)


def set_random_weights(layer, rng):
    for proj in [layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj]:
        proj.W.set_value(rng.normal(size=proj.W.get_value().shape).astype(floatX))
        proj.b.set_value(rng.normal(size=proj.b.get_value().shape).astype(floatX))


def test_multihead_attention_shape_and_value(rng):
    n_embd, n_head = 12, 3
    mha = MultiheadAttention("mha", n_embd=n_embd, n_head=n_head)
    set_random_weights(mha, rng)

    X = pt.tensor("X", shape=(None, None, n_embd))
    out = mha(X)
    assert out.type.shape == (None, None, n_embd)
    assert out.name == "mha_output"

    batch, seq = 2, 5
    X_np = rng.normal(size=(batch, seq, n_embd)).astype(floatX)
    result = out.eval({X: X_np})

    head_dim = n_embd // n_head

    def project(proj, x):
        return x @ proj.W.get_value() + proj.b.get_value()

    def split(x):
        return x.reshape(batch, seq, n_head, head_dim).transpose(0, 2, 1, 3)

    q = split(project(mha.q_proj, X_np))
    k = split(project(mha.k_proj, X_np))
    v = split(project(mha.v_proj, X_np))
    attn = sdpa_np(q, k, v).transpose(0, 2, 1, 3).reshape(batch, seq, n_embd)
    expected = project(mha.out_proj, attn)

    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_cross_attention_shape_and_value(rng):
    n_embd, kv_dim, n_head = 12, 8, 3
    mha = MultiheadAttention("mha", n_embd=n_embd, n_head=n_head, kv_dim=kv_dim)
    set_random_weights(mha, rng)

    assert mha.q_proj.W.get_value().shape[0] == n_embd
    assert mha.k_proj.W.get_value().shape[0] == kv_dim
    assert mha.v_proj.W.get_value().shape[0] == kv_dim

    X = pt.tensor("X", shape=(None, None, n_embd))
    context = pt.tensor("context", shape=(None, None, kv_dim))
    out = mha(X, context)
    assert out.type.shape == (None, None, n_embd)

    batch, seq, seq_kv = 2, 5, 3
    X_np = rng.normal(size=(batch, seq, n_embd)).astype(floatX)
    context_np = rng.normal(size=(batch, seq_kv, kv_dim)).astype(floatX)
    result = out.eval({X: X_np, context: context_np})

    head_dim = n_embd // n_head

    def project(proj, x):
        return x @ proj.W.get_value() + proj.b.get_value()

    def split(x, length):
        return x.reshape(batch, length, n_head, head_dim).transpose(0, 2, 1, 3)

    q = split(project(mha.q_proj, X_np), seq)
    k = split(project(mha.k_proj, context_np), seq_kv)
    v = split(project(mha.v_proj, context_np), seq_kv)
    attn = sdpa_np(q, k, v).transpose(0, 2, 1, 3).reshape(batch, seq, n_embd)
    expected = project(mha.out_proj, attn)

    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_causal_cross_attention_raises():
    mha = MultiheadAttention("mha", n_embd=12, n_head=3, is_causal=True)
    X = pt.tensor("X", shape=(2, 5, 12))

    with pytest.raises(ValueError, match="mutually exclusive"):
        mha(X, X)


@pytest.mark.parametrize("n_kv_head", [None, 2], ids=["multi_head", "grouped_query"])
def test_fused_qkv_matches_three_separate_projections(n_kv_head, rng):
    """The fused weight is the three stacked in q, k, v order, so splitting it has to reproduce them
    exactly. Grouped-query attention makes the three widths unequal, which is where an even split
    would go wrong."""
    n_embd, n_head = 12, 4
    separate = MultiheadAttention("sep", n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
    fused = MultiheadAttention(
        "fused", n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head, fused_qkv=True
    )
    set_random_weights(separate, rng)

    stacked = np.concatenate(
        [
            separate.q_proj.W.get_value(),
            separate.k_proj.W.get_value(),
            separate.v_proj.W.get_value(),
        ],
        axis=-1,
    )
    stacked_bias = np.concatenate(
        [
            separate.q_proj.b.get_value(),
            separate.k_proj.b.get_value(),
            separate.v_proj.b.get_value(),
        ]
    )
    fused.qkv_proj.W.set_value(stacked)
    fused.qkv_proj.b.set_value(stacked_bias)
    fused.out_proj.W.set_value(separate.out_proj.W.get_value())
    fused.out_proj.b.set_value(separate.out_proj.b.get_value())

    X = pt.tensor("X", shape=(2, 5, n_embd))
    X_np = rng.normal(size=(2, 5, n_embd)).astype(floatX)

    np.testing.assert_allclose(
        fused(X).eval({X: X_np}), separate(X).eval({X: X_np}), rtol=1e-6, atol=1e-6
    )


def test_fused_qkv_owns_one_weight_of_the_stacked_width():
    """GPT-2 stores c_attn as a single tensor, so a loader must find one parameter rather than three."""
    fused = MultiheadAttention("fused", n_embd=12, n_head=3, fused_qkv=True)

    assert fused.qkv_proj.W.get_value().shape == (12, 36)
    assert (fused.q_proj, fused.k_proj, fused.v_proj) == (None, None, None)


def test_fused_qkv_with_a_separate_kv_width_raises_at_construction():
    """A differently-sized key/value input cannot share a weight with the queries, and there is no
    point building the layer to find out."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        MultiheadAttention("fused", n_embd=12, n_head=3, fused_qkv=True, kv_dim=8)


def test_fused_qkv_with_a_second_input_raises_at_call():
    """Keys and values come from the query's own weight, so a second input has nowhere to project.
    The layer is legal to build -- only using it this way is not."""
    attention = MultiheadAttention("fused", n_embd=12, n_head=3, fused_qkv=True)
    X = pt.tensor("X", shape=(2, 5, 12))

    with pytest.raises(ValueError, match="cannot come from a second input"):
        attention(X, X)


def test_causal_self_attention_is_causal(rng):
    csa = CausalSelfAttention("csa", n_embd=12, n_head=3)
    assert csa.is_causal is True
    out = csa(pt.tensor("X", shape=(2, 5, 12)))
    attn_ops = [
        var.owner.op
        for var in ancestors([out])
        if var.owner is not None and isinstance(var.owner.op, AttentionLayer)
    ]
    assert len(attn_ops) == 1
    assert attn_ops[0].is_causal is True


def test_attention_gradient_flows_to_all_projections(rng):
    n_embd, n_head = 8, 2
    csa = CausalSelfAttention("csa", n_embd=n_embd, n_head=n_head)
    set_random_weights(csa, rng)

    X = pt.tensor("X", shape=(2, 4, n_embd))
    loss = csa(X).sum()
    X_np = rng.normal(size=(2, 4, n_embd)).astype(floatX)

    for proj in [csa.q_proj, csa.k_proj, csa.v_proj, csa.out_proj]:
        grad = pt.grad(loss, proj.W).eval({X: X_np})
        assert np.abs(grad).sum() > 0, f"no gradient into {proj.name}"


def test_attention_vectorizes_over_independent_inputs(rng):
    """vectorize_graph adds a leading batch of independent forward passes (multi-prediction)."""
    n_embd, n_head = 8, 2
    mha = MultiheadAttention("mha", n_embd=n_embd, n_head=n_head, is_causal=True)
    set_random_weights(mha, rng)

    X = pt.tensor("X", shape=(2, 4, n_embd))
    out = mha(X)

    n_models = 3
    X_batched = pt.tensor("X_batched", shape=(n_models, 2, 4, n_embd))
    out_batched = vectorize_graph(out, replace={X: X_batched})

    X_np = rng.normal(size=(n_models, 2, 4, n_embd)).astype(floatX)
    batched_result = out_batched.eval({X_batched: X_np})
    stacked_result = np.stack([out.eval({X: X_np[i]}) for i in range(n_models)])

    np.testing.assert_allclose(batched_result, stacked_result, atol=1e-5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_embd": 10, "n_head": 3}, "must be divisible"),
        ({"n_embd": 12, "n_head": 4, "n_kv_head": 3}, "must be divisible"),
    ],
    ids=["n_embd_not_divisible", "n_kv_head_not_divisible"],
)
def test_multihead_attention_validates_head_counts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MultiheadAttention("mha", **kwargs)
