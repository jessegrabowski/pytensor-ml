# Imported for its side effect: registering pytensor_ml's JAX funcify dispatches. One submodule per
# marker op that gets a kernel, mirroring the layout under pytensor_ml/layers.
import pytensor_ml.dispatch.jax.attention
import pytensor_ml.dispatch.jax.conv
import pytensor_ml.dispatch.jax.pooling
