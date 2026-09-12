from collections.abc import Callable

import pytensor.tensor as pt

from pytensor.raise_op import CheckAndRaise
from pytensor.tensor import TensorVariable
from pytensor.tensor.exceptions import NotScalarConstantError


def checked_scalar(
    value: TensorVariable,
    *,
    name: str,
    complaint: str | None = None,
    condition_fn: Callable[[TensorVariable], TensorVariable] | None = None,
) -> TensorVariable:
    """
    Return a hyperparameter, refusing an invalid one wherever that can be established.

    A hyperparameter is often written as arithmetic on a shape, such as ``10 * X.shape[0]``, which
    has no value until the function runs. Fold the comparison where it folds and refuse a
    bad value outright; otherwise attach the check to the graph. That check rides on the value, so it
    runs wherever the value does -- not under jax, which drops assertions, and not where a rewrite
    has eliminated the value's last consumer.

    Parameters
    ----------
    value : TensorVariable
        The value to check. Must be a scalar.
    name : str
        The parameter's name, used to say which value is at fault.
    complaint : str, optional
        What to say when the value is invalid, following the name and without the value itself.
        Required whenever ``condition_fn`` is given.
    condition_fn : callable, optional
        Maps the value to the condition it must satisfy, such as ``lambda count: count >= 1``. Omit it to
        check only that the value is a scalar, which is what a pair of values needs established before
        either can be compared against the other.

    Returns
    -------
    value : TensorVariable
        The value unchanged when the condition holds at build time, and carrying a runtime check when
        the condition cannot be settled until the function runs.
    """
    if value.ndim != 0:
        raise ValueError(
            f"{name} must be a single number, but got one with {value.ndim} dimensions. Reduce it "
            "first: `X.shape[0]` rather than `X.shape`."
        )

    if condition_fn is None:
        return value

    condition = condition_fn(value)
    try:
        holds = pt.get_underlying_scalar_constant_value(condition)
    except NotScalarConstantError:
        return pt.as_tensor_variable(
            CheckAndRaise(ValueError, f"{name} {complaint}.")(value, condition)
        )

    if not holds:
        offender = pt.get_underlying_scalar_constant_value(value)
        raise ValueError(f"{name} {complaint}, got {offender}.")
    return value
