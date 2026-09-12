import logging
import sys

import pytensor_ml.dispatch  # installs the lazy backend-dispatch registration hook
import pytensor_ml.rewriting.conv  # registers the convolution rewrites into pytensor's databases
import pytensor_ml.rewriting.layers  # registers the degenerate-dropout rewrite into pytensor's databases

from pytensor_ml._version import __version__
from pytensor_ml.checkpoint import load_state, save_state
from pytensor_ml.pretrained import from_pretrained, load_network, save_network, save_pretrained
from pytensor_ml.pytensorf import function

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler(sys.stderr)
        _log.addHandler(handler)


__all__ = [
    "__version__",
    "from_pretrained",
    "function",
    "load_network",
    "load_state",
    "save_network",
    "save_pretrained",
    "save_state",
]
