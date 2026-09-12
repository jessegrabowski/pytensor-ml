import pytest

import pytensor_ml.optim as optim


def test_every_exported_name_resolves():
    """``__all__`` is written by hand, and a mistake in it is invisible: a missing comma silently
    concatenates two entries into one name that resolves to nothing, and ``import *`` is the only thing
    that notices."""
    missing = [name for name in optim.__all__ if not hasattr(optim, name)]

    assert not missing


def test_exported_names_are_unique_and_sorted():
    assert optim.__all__ == sorted(set(optim.__all__))


@pytest.mark.parametrize(
    "name", ["to_updates", "steps_of", "state_for", "reuses_state", "counter", "chain"]
)
def test_the_transform_authoring_api_stays_public(name):
    """The ``Transform`` docstring tells authors to write a transform with these. Dropping one from
    ``__all__`` would leave the documented path running through ``pytensor_ml.optim.base``."""
    assert name in optim.__all__
