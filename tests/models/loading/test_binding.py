from pytensor_ml.layers import Linear
from pytensor_ml.models import KeyMap, bind_linear


def test_bind_linear_binds_no_bias_for_a_bias_free_layer():
    """CLIP's text projection has no bias, and binding one would ask the checkpoint for a tensor it
    does not hold."""
    keys = KeyMap()
    bind_linear(keys, Linear("text_projection", n_in=4, n_out=2, bias=False), "text_projection")

    assert keys.keys() == {"text_projection.weight"}
