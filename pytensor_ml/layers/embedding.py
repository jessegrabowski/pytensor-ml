import pytensor.tensor as pt

from pytensor_ml.base import Layer, UnaryLayerOp, _resolve_layer_name
from pytensor_ml.params import trainable
from pytensor_ml.state import Initializer, XavierNormalInitializer


class EmbeddingLayer(UnaryLayerOp):
    __props__ = ("n_embeddings", "n_features")

    def build_inner_graph(self, ids, W):
        return [W[ids]]


class Embedding(Layer):
    r"""
    Lookup-table embedding.

    Map each integer index to a learned row of the ``(n_embeddings, n_features)`` table,
    appending a trailing feature axis of size ``n_features`` while preserving the shape of the
    index input.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "Embedding" when None.
    n_embeddings : int
        Number of rows in the table -- the number of distinct indices it can map.
    n_features : int
        Size of each embedding row.
    weight_initializer : Initializer, optional
        How the table is drawn, at construction and on every redraw. Xavier normal when omitted, which puts
        the vocabulary size in the denominator -- correct Xavier, and much tighter than the
        ``NormalInitializer(0.0, 0.02)`` that reference implementations of GPT-2 use, so this is the keyword
        to reach for when matching one.

    Examples
    --------
    Look up a learned vector per integer token, which is how a vocabulary enters a network. The input
    carries token ids, so it must be an integer tensor:

    .. code-block:: python

        from pytensor_ml.layers import Embedding, Input

        tokens = Input("tokens", shape=(None, 128), dtype="int64")
        embedded = Embedding("embed", n_embeddings=50_000, n_features=256)(tokens)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_embeddings: int,
        n_features: int,
        weight_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "n_embeddings")
        self.n_embeddings = n_embeddings
        self.n_features = n_features

        # Drawn here, as in Linear, and for the same reason.
        W_initializer = (
            XavierNormalInitializer() if weight_initializer is None else weight_initializer
        )
        self.W = trainable(
            W_initializer.initial_value((n_embeddings, n_features)),
            f"{self.name}_W",
            initializer=W_initializer,
            layer_name=self.name,
        )

    def __call__(self, ids: pt.TensorLike) -> pt.TensorVariable:
        ids = pt.as_tensor(ids)

        out = EmbeddingLayer(
            name=self.name,
            n_embeddings=self.n_embeddings,
            n_features=self.n_features,
        )(ids, self.W)
        out.name = f"{self.name}_output"

        return out
