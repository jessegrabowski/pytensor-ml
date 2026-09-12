Layers
======

.. currentmodule:: pytensor_ml.layers

Base
----

.. autosummary::
    :toctree: generated/

    Layer
    pytensor_ml.base.VariadicLayer

Combinators
-----------

.. autosummary::
    :toctree: generated/

    Input
    Sequential
    Concatenate
    Flatten
    Squeeze

Dense
-----

.. autosummary::
    :toctree: generated/

    Linear
    Embedding

Convolution and pooling
-----------------------

.. autosummary::
    :toctree: generated/

    Conv1D
    Conv2D
    ConvTranspose1D
    ConvTranspose2D
    MaxPool1D
    MaxPool2D
    AvgPool1D
    AvgPool2D
    Upsample1D
    Upsample2D

Padding
-------

.. autosummary::
    :toctree: generated/

    ZeroPad1D
    ZeroPad2D
    ConstantPad1D
    ConstantPad2D
    ReflectionPad1D
    ReflectionPad2D
    ReplicationPad1D
    ReplicationPad2D

Normalization and regularization
--------------------------------

.. autosummary::
    :toctree: generated/

    BatchNorm
    LayerNorm
    RMSNorm
    GroupNorm
    Dropout

Recurrent
---------

.. autosummary::
    :toctree: generated/

    RNN
    LSTM
    GRU
    Bidirectional
    Recurrent
    RecurrentCell
    ElmanCell
    LSTMCell
    GRUCell

Attention and transformers
--------------------------

.. autosummary::
    :toctree: generated/

    MultiheadAttention
    CausalSelfAttention
    FeedForward
    TransformerBlock
    scaled_dot_product_attention
    RotaryEmbedding
    rotary_embedding
