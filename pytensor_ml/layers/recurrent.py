from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Variable
from pytensor.scan import scan
from pytensor.tensor.variable import TensorVariable

from pytensor_ml.activations import Activation, Sigmoid, Tanh
from pytensor_ml.base import Layer, _resolve_layer_name
from pytensor_ml.params import trainable_parameter
from pytensor_ml.state import (
    Initializer,
    OrthogonalInitializer,
    XavierNormalInitializer,
    ZeroInitializer,
)


class RecurrentCell(ABC):
    """
    One step of a recurrence, for :class:`Recurrent` to scan over a sequence.

    A cell owns the parameters its step uses and knows the shape of the state it carries, which is all
    the loop needs from it. Subclasses implement :meth:`step` and :meth:`initial_state`.

    Examples
    --------
    Subclass it to define one timestep and let :class:`Recurrent` scan it over the sequence. A cell owns its
    parameters, says how to build its initial state, and maps ``(input, *state)`` to the next state:

    .. code-block:: python

        import numpy as np
        import pytensor.tensor as pt

        from pytensor_ml.layers import Input, Recurrent
        from pytensor_ml.layers.recurrent import RecurrentCell
        from pytensor_ml.params import trainable
        from pytensor_ml.state import XavierUniformInitializer


        class LeakyCell(RecurrentCell):
            def __init__(self, name, n_in, n_hidden):
                self.n_hidden = n_hidden
                self.W = trainable(
                    np.zeros((n_in + n_hidden, n_hidden)),
                    f"{name}_W",
                    initializer=XavierUniformInitializer(),
                )

            def step(self, x_t, h):
                candidate = pt.tanh(pt.concatenate([x_t, h], axis=-1) @ self.W)
                return (0.5 * h + 0.5 * candidate,)

            def initial_state(self, X):
                return (pt.zeros((X.shape[0], self.n_hidden)),)


        X = Input("X", shape=(None, 50, 16))
        hidden_states = Recurrent(LeakyCell("leaky", n_in=16, n_hidden=32))(X)
    """

    @abstractmethod
    def step(self, x_t: TensorVariable, *state: TensorVariable) -> tuple[TensorVariable, ...]:
        """
        Advance the state by one timestep.

        Parameters
        ----------
        x_t : TensorVariable
            The sequence at this step, shape ``(..., n_in)``.
        *state : TensorVariable
            The state carried out of the previous step, as :meth:`initial_state` laid it out.

        Returns
        -------
        state : tuple of TensorVariable
            The new state, in the same order. Its first element is the cell's output, which is what
            :class:`Recurrent` stacks over time -- an LSTM carrying :math:`(h, c)` returns :math:`h` first.
        """

    @abstractmethod
    def initial_state(self, X: TensorVariable) -> tuple[TensorVariable, ...]:
        """
        Build the state the recurrence starts from, for a sequence ``X`` of shape ``(..., time, n_in)``.

        Carries one value per batch element, so the state's batch axes are ``X``'s and its dtype is the
        one :meth:`step` produces -- a float64 sequence through a float32 cell makes a float64 state.

        Returns
        -------
        state : tuple of TensorVariable
            Zero-filled state, in the order :meth:`step` takes and returns it.
        """


class Recurrent(Layer):
    """
    Scan a :class:`RecurrentCell` over the time axis of a sequence.

    Time is the second-to-last axis and everything before it is a batch axis, so the input is
    ``(..., time, n_in)`` and the output ``(..., time, n_out)``, one cell output per step. Slice the last
    step off the result -- ``out[..., -1, :]`` -- for the sequence-classification case; pytensor's
    ``scan_save_mem`` rewrite sees that the earlier steps are unused and stops storing them.

    Parameters
    ----------
    cell : RecurrentCell
        The step to run at each timestep.
    name : str or None
        Name for the layer's output and its scan. Defaults to "Recurrent" when None.
    reverse : bool, optional
        Run the sequence from its last step to its first. The output stays aligned with the input, so
        ``out[..., t, :]`` is the step that read ``X[..., t, :]`` either way, and a backward layer's
        output concatenates elementwise with a forward one's. Default is False.

    Examples
    --------
    Wrap any cell to scan it over the time axis. :class:`RNN`, :class:`LSTM` and :class:`GRU` are this
    layer around their matching cell, and ``reverse=True`` is what the backward half of a
    :class:`Bidirectional` uses:

    .. code-block:: python

        from pytensor_ml.layers import GRUCell, Input, Recurrent

        X = Input("X", shape=(None, 200, 16))
        hidden_states = Recurrent(GRUCell("cell", n_in=16, n_hidden=32), reverse=True)(X)
    """

    def __init__(self, cell: RecurrentCell, *, name: str | None = None, reverse: bool = False):
        self.cell = cell
        self.name = _resolve_layer_name(name, type(self).__name__)
        self.reverse = reverse

    def __call__(
        self,
        X: pt.TensorLike,
        initial_state: pt.TensorLike | Sequence[TensorVariable] | None = None,
        *,
        reverse: bool | None = None,
        mask: pt.TensorLike | None = None,
    ) -> TensorVariable:
        """
        Run the cell over ``X`` and return its output at every step.

        Parameters
        ----------
        X : TensorVariable
            Input sequence, shape ``(..., time, n_in)``.
        initial_state : TensorVariable or sequence of TensorVariable, optional
            The state the recurrence starts from, matching what the cell carries. The cell's zero state
            when omitted.
        reverse : bool, optional
            Direction for this call, in place of the layer's own. The layer's when omitted.
        mask : TensorVariable, optional
            Which steps are real, shape ``(..., time)``, matching ``X``'s batch axes. A false step
            leaves every carried state as the step before it left it, so padding a batch of ragged
            sequences out to a rectangle does not disturb the recurrence over the real steps of it.
            Every step counts when omitted.

        Returns
        -------
        outputs : TensorVariable
            The cell's output at each step, shape ``(..., time, n_out)``.
        """
        reverse = self.reverse if reverse is None else reverse

        X = pt.as_tensor(X)
        if X.ndim < 2:
            raise ValueError(
                f"{self.name} takes a sequence of shape (..., time, n_in), but got a "
                f"{X.ndim}-dimensional input, which has no time axis to recur over."
            )

        zero_state = self.cell.initial_state(X)
        if initial_state is None:
            state = zero_state
        else:
            if isinstance(initial_state, list | tuple):
                state = tuple(pt.as_tensor(tensor) for tensor in initial_state)
            else:
                state = (pt.as_tensor(initial_state),)
            self._check_state_against(state, zero_state, X)

        # Scan iterates the leading axis, so time moves to the front and back again on the way out.
        sequences: list[Variable] = [pt.moveaxis(X, -2, 0)]
        # A masked step reads one more sequence than a plain one, so the two do not share a signature.
        step: Callable[..., tuple[TensorVariable, ...]] = self.cell.step
        if mask is not None:
            mask = pt.as_tensor(mask)
            self._check_mask_against(mask, X)
            steps = pt.moveaxis(mask, -1, 0)
            # Scan takes its step count from the shortest sequence it is given, so a mask a step short
            # would quietly run the whole recurrence a step short.
            sequences.append(pt.specify_shape(steps, (X.shape[-2], *steps.shape[1:])))
            step = self._masked_step

        # Not strict: the step closes over the cell's parameters and scan lifts them in. A generator
        # captured that way has no update, which `collect_default_updates` refuses.
        state_sequence = scan(
            step,
            sequences=sequences,
            outputs_info=list(state),
            name=f"{self.name}_recurrence",
            go_backwards=reverse,
            return_updates=False,
        )

        # Scan hands back a bare variable for a single carried state and a list for several. The cell's
        # output is the first one either way.
        output = state_sequence[0] if isinstance(state_sequence, list) else state_sequence

        # Scan stacks in the order it iterated, so a backward pass comes out last step first.
        if reverse:
            output = output[::-1]

        # Back to where it came from: time sits directly after the input's batch axes, which is the
        # second-to-last axis only for a state carrying a single feature axis.
        out = pt.moveaxis(output, 0, X.ndim - 2)
        out.name = f"{self.name}_output"
        return out

    def _masked_step(
        self, x_t: TensorVariable, mask_t: TensorVariable, *state: TensorVariable
    ) -> tuple[TensorVariable, ...]:
        """Take the step, then keep it only where ``mask_t`` says this step is real."""
        # The step runs either way -- scan's carried states are fixed-shape buffers, so there is no
        # skipping a subset of the batch, only discarding what it computed for them.
        stepped = self.cell.step(x_t, *state)
        # One trailing axis per feature axis of the state, so the mask lines its batch axes up against
        # the state's however many feature axes the cell chose to carry.
        return tuple(
            pt.switch(pt.shape_padright(mask_t, held.ndim - mask_t.ndim), new, held)
            for new, held in zip(stepped, state)
        )

    def _check_mask_against(self, mask: TensorVariable, X: TensorVariable) -> None:
        """Reject a mask that does not name one step per batch element, before scan reports it."""
        if mask.ndim != X.ndim - 1:
            raise ValueError(
                f"{self.name} takes a mask of shape (..., time), matching its input without the "
                f"feature axis, so a {X.ndim}-dimensional input needs a {X.ndim - 1}-dimensional "
                f"mask; got a {mask.ndim}-dimensional one."
            )

    def _check_state_against(
        self,
        given: Sequence[TensorVariable],
        expected: Sequence[TensorVariable],
        X: TensorVariable,
    ) -> None:
        """Reject a starting state the cell would not have built, before scan reports it from inside."""
        if len(given) != len(expected):
            raise ValueError(
                f"{self.name}'s cell carries {len(expected)} state tensor(s), but got {len(given)}."
            )
        for position, (state, zero) in enumerate(zip(given, expected)):
            if state.ndim != zero.ndim:
                raise ValueError(
                    f"{self.name} starts from a state carrying the same batch axes as its input, so a "
                    f"{X.ndim}-dimensional input needs a {zero.ndim}-dimensional state at position "
                    f"{position}; got a {state.ndim}-dimensional one."
                )


class ElmanCell(RecurrentCell):
    r"""
    The step of an Elman recurrence, which updates its state from the step's input and the previous state:

    .. math::

        h_t = \phi\left(x_t W_{ih} + b + h_{t-1} W_{hh}\right),

    where :math:`\phi` is the activation.

    Parameters
    ----------
    name : str or None
        Name prefix for the cell's parameters. Defaults to "ElmanCell" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state.
    activation : Activation, optional
        Applied to each step's pre-activation. Default is :class:`~pytensor_ml.activations.Tanh`, which
        bounds the state and so keeps the recurrence from running away over a long sequence.
    bias : bool, optional
        Add the learned shift :math:`b`. One bias covers the step, rather than torch's separate
        ``b_ih`` and ``b_hh``, whose sum is the only thing the step can distinguish. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn. It meets the state once per step, so its singular values compound
        over the sequence: at one the state's norm survives any length, and spread around one the
        gradient explodes along some directions while vanishing along others. Orthogonal when omitted,
        as in keras and flax.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.

    Examples
    --------
    One timestep of a plain recurrent layer, for handing to :class:`Recurrent` when you want the scan
    configured yourself rather than through :class:`RNN`:

    .. code-block:: python

        from pytensor_ml.activations import Tanh
        from pytensor_ml.layers import ElmanCell, Input, Recurrent

        X = Input("X", shape=(None, 50, 16))
        cell = ElmanCell("cell", n_in=16, n_hidden=32, activation=Tanh())

        hidden_states = Recurrent(cell)(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "n_in")
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation = activation if activation is not None else Tanh()
        self.bias = bias

        # Held directly rather than as a nested Linear: the projection runs inside the recurrence, and a
        # layer op there would bury its matmul in an inner graph where the scan rewrites cannot see it.
        self.W_ih = trainable_parameter(
            f"{self.name}_W_ih",
            (n_in, n_hidden),
            weight_initializer,
            XavierNormalInitializer(),
            layer_name=self.name,
        )
        if bias:
            self.b = trainable_parameter(
                f"{self.name}_b",
                (n_hidden,),
                bias_initializer,
                ZeroInitializer(),
                layer_name=self.name,
            )
        self.W_hh = trainable_parameter(
            f"{self.name}_W_hh",
            (n_hidden, n_hidden),
            recurrent_initializer,
            OrthogonalInitializer(),
            layer_name=self.name,
        )

    def step(self, x_t: TensorVariable, *state: TensorVariable) -> tuple[TensorVariable, ...]:
        (h_prev,) = state
        pre_activation = x_t @ self.W_ih + h_prev @ self.W_hh
        if self.bias:
            pre_activation = pre_activation + self.b
        return (self.activation(pre_activation),)

    def initial_state(self, X: TensorVariable) -> tuple[TensorVariable, ...]:
        return (_zero_state(X, self.n_hidden, self.W_ih, self.W_hh),)


class RNN(Recurrent):
    r"""
    Elman recurrent layer over a sequence: a :class:`Recurrent` scanning an :class:`ElmanCell`.

    Takes the cell's arguments directly, for the common case where a network wants a plain recurrence
    and no cell of its own. The parameters live on the cell, as ``rnn.cell.W_ih``. See
    :class:`ElmanCell` for the recurrence itself and :class:`Recurrent` for the axes.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "RNN" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state.
    activation : Activation, optional
        Applied to each step's pre-activation. Default is :class:`~pytensor_ml.activations.Tanh`.
    bias : bool, optional
        Add the learned shift :math:`b`. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn. Orthogonal when omitted; see :class:`ElmanCell` for why that is the
        draw an RNN is most sensitive to.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.
    reverse : bool, optional
        Run the sequence backward, with the output still aligned to the input. Default is False.

    Examples
    --------
    The plain recurrent layer: one hidden state carried along the time axis, returning its value at every
    step. Input is ``(batch, time, features)``:

    .. code-block:: python

        from pytensor_ml.layers import RNN, Input

        X = Input("X", shape=(None, 50, 16))
        hidden_states = RNN("rnn", n_in=16, n_hidden=32)(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
        reverse: bool = False,
    ):
        name = name if name else "RNN"
        super().__init__(
            ElmanCell(
                name,
                n_in=n_in,
                n_hidden=n_hidden,
                activation=activation,
                bias=bias,
                weight_initializer=weight_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
            ),
            name=name,
            reverse=reverse,
        )


class GRUCell(RecurrentCell):
    r"""
    The step of a gated recurrent unit, which interpolates between the previous state and a candidate:

    .. math::

        r_t &= \sigma\left(x_t W_{ir} + b_r + h_{t-1} W_{hr}\right) \\
        z_t &= \sigma\left(x_t W_{iz} + b_z + h_{t-1} W_{hz}\right) \\
        n_t &= \phi\left(x_t W_{in} + b_n + r_t \odot \left(h_{t-1} W_{hn} + c\right)\right) \\
        h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1},

    where :math:`\phi` is the activation and :math:`\sigma` the gate activation. The update gate
    :math:`z` decides how much of the previous state survives the step, so a unit holding :math:`z` near
    one carries its value across the whole sequence and the gradient reaches the start of it; the reset
    gate :math:`r` decides how much of that state the candidate is allowed to see.

    The three gates share one projection of the input and one of the state, so a step is two matmuls
    rather than six. :math:`r` multiplies the state's projection after that matmul rather than before
    it -- as torch, flax and keras' ``reset_after`` all do -- which is what allows the fused form.

    Parameters
    ----------
    name : str or None
        Name prefix for the cell's parameters. Defaults to "GRUCell" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state.
    activation : Activation, optional
        Applied to the candidate state. Default is :class:`~pytensor_ml.activations.Tanh`.
    gate_activation : Activation, optional
        Applied to the reset and update gates. Only a squashing function makes :math:`h_t` an
        interpolation; a gate ranging outside :math:`[0, 1]` extrapolates past both the previous state
        and the candidate. Default is :class:`~pytensor_ml.activations.Sigmoid`.
    bias : bool, optional
        Add the learned shifts :math:`b` and :math:`c`. One bias per gate covers the input and state
        projections together, rather than torch's separate pair, whose sum is the only thing a gate can
        distinguish. The candidate's state projection is the exception and carries its own :math:`c`:
        :math:`r_t` scales it, so it moves independently of :math:`b_n`. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn, across all three gates at once, as in keras. Orthogonal when
        omitted; see :class:`ElmanCell` for why the state's own weight is the sensitive draw.
    bias_initializer : Initializer, optional
        How :math:`b` and :math:`c` are drawn. Zeros when omitted.

    Examples
    --------
    One timestep of a GRU, carrying a single hidden state. Use it where the scan is built by hand rather
    than through :class:`GRU`:

    .. code-block:: python

        from pytensor_ml.layers import GRUCell, Input, Recurrent

        X = Input("X", shape=(None, 200, 16))
        hidden_states = Recurrent(GRUCell("cell", n_in=16, n_hidden=32))(X)
    """

    _n_gates = 3

    def __init__(
        self,
        name: str | None = None,
        *,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        gate_activation: Activation | None = None,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "n_in")
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation = activation if activation is not None else Tanh()
        self.gate_activation = gate_activation if gate_activation is not None else Sigmoid()
        self.bias = bias

        self.W_ih = trainable_parameter(
            f"{self.name}_W_ih",
            (n_in, self._n_gates * n_hidden),
            weight_initializer,
            XavierNormalInitializer(),
            layer_name=self.name,
        )
        self.W_hh = trainable_parameter(
            f"{self.name}_W_hh",
            (n_hidden, self._n_gates * n_hidden),
            recurrent_initializer,
            OrthogonalInitializer(),
            layer_name=self.name,
        )
        if bias:
            self.b = trainable_parameter(
                f"{self.name}_b",
                (self._n_gates * n_hidden,),
                bias_initializer,
                ZeroInitializer(),
                layer_name=self.name,
            )
            self.c = trainable_parameter(
                f"{self.name}_c",
                (n_hidden,),
                bias_initializer,
                ZeroInitializer(),
                layer_name=self.name,
            )

    def step(self, x_t: TensorVariable, *state: TensorVariable) -> tuple[TensorVariable, ...]:
        (h_prev,) = state

        from_input = x_t @ self.W_ih
        if self.bias:
            from_input = from_input + self.b
        from_state = h_prev @ self.W_hh

        input_r, input_z, input_n = _split_gates(from_input, self.n_hidden, self._n_gates)
        state_r, state_z, state_n = _split_gates(from_state, self.n_hidden, self._n_gates)
        if self.bias:
            state_n = state_n + self.c

        reset = self.gate_activation(input_r + state_r)
        update = self.gate_activation(input_z + state_z)
        candidate = self.activation(input_n + reset * state_n)

        return ((1 - update) * candidate + update * h_prev,)

    def initial_state(self, X: TensorVariable) -> tuple[TensorVariable, ...]:
        return (_zero_state(X, self.n_hidden, self.W_ih, self.W_hh),)


class GRU(Recurrent):
    r"""
    Gated recurrent layer over a sequence: a :class:`Recurrent` scanning a :class:`GRUCell`.

    Takes the cell's arguments directly, for the common case where a network wants a plain recurrence
    and no cell of its own. The parameters live on the cell, as ``gru.cell.W_ih``. See :class:`GRUCell`
    for the recurrence itself and :class:`Recurrent` for the axes.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "GRU" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state.
    activation : Activation, optional
        Applied to the candidate state. Default is :class:`~pytensor_ml.activations.Tanh`.
    gate_activation : Activation, optional
        Applied to the reset and update gates. Default is
        :class:`~pytensor_ml.activations.Sigmoid`; see :class:`GRUCell` for what a gate outside
        :math:`[0, 1]` does to the step.
    bias : bool, optional
        Add the learned shifts. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn. Orthogonal when omitted.
    bias_initializer : Initializer, optional
        How the biases are drawn. Zeros when omitted.
    reverse : bool, optional
        Run the sequence backward, with the output still aligned to the input. Default is False.

    Examples
    --------
    Gated like an :class:`LSTM` but with one state instead of two, so it trains faster and holds fewer
    parameters at similar quality on many sequences:

    .. code-block:: python

        from pytensor_ml.layers import GRU, Input

        X = Input("X", shape=(None, 200, 16))
        hidden_states = GRU("gru", n_in=16, n_hidden=32)(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        gate_activation: Activation | None = None,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
        reverse: bool = False,
    ):
        name = name if name else "GRU"
        super().__init__(
            GRUCell(
                name,
                n_in=n_in,
                n_hidden=n_hidden,
                activation=activation,
                bias=bias,
                gate_activation=gate_activation,
                weight_initializer=weight_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
            ),
            name=name,
            reverse=reverse,
        )


class LSTMCell(RecurrentCell):
    r"""
    The step of a long short-term memory cell, which carries a memory alongside its output:

    .. math::

        i_t &= \sigma\left(x_t W_{ii} + b_i + h_{t-1} W_{hi}\right) \\
        f_t &= \sigma\left(x_t W_{if} + b_f + h_{t-1} W_{hf}\right) \\
        g_t &= \phi\left(x_t W_{ig} + b_g + h_{t-1} W_{hg}\right) \\
        o_t &= \sigma\left(x_t W_{io} + b_o + h_{t-1} W_{ho}\right) \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
        h_t &= o_t \odot \phi(c_t),

    where :math:`\phi` is the activation and :math:`\sigma` the gate activation. The memory
    :math:`c` runs through the sequence touched only by two elementwise gates, so the gradient reaches
    the start of it without passing through a weight; the forget gate :math:`f` decides what the memory
    keeps, the input gate :math:`i` what the candidate :math:`g` adds to it, and the output gate
    :math:`o` how much of it the step exposes as :math:`h`.

    The four gates share one projection of the input and one of the state, so a step is two matmuls
    rather than eight. Every gate sees the two projections only as a sum, so a single bias covers both,
    which is the layout flax uses and half of torch's.

    Parameters
    ----------
    name : str or None
        Name prefix for the cell's parameters. Defaults to "LSTMCell" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state, and of the memory it carries alongside.
    activation : Activation, optional
        Applied to the candidate and again to the memory on the way out, as in torch, flax and keras.
        Default is :class:`~pytensor_ml.activations.Tanh`, which bounds the memory the output gate
        reads however far the unbounded :math:`c` has drifted.
    gate_activation : Activation, optional
        Applied to the input, forget and output gates. Only a squashing function makes them gates; the
        forget gate ranging outside :math:`[0, 1]` grows or flips the memory it is meant to decay.
        Default is :class:`~pytensor_ml.activations.Sigmoid`.
    bias : bool, optional
        Add the learned shift :math:`b`, one slice per gate. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn, across all four gates at once. Orthogonal when omitted; see
        :class:`ElmanCell` for why the state's own weight is the sensitive draw.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted, as in torch and flax. Drawing the forget slice at
        one instead starts the memory holding rather than decaying, which is keras' default.

    Examples
    --------
    One timestep of an LSTM, carrying ``(hidden, cell)`` as its state. Reach for it when the scan needs
    configuring directly rather than through :class:`LSTM`:

    .. code-block:: python

        from pytensor_ml.layers import Input, LSTMCell, Recurrent

        X = Input("X", shape=(None, 200, 16))
        hidden_states = Recurrent(LSTMCell("cell", n_in=16, n_hidden=32), reverse=True)(X)
    """

    _n_gates = 4

    def __init__(
        self,
        name: str | None = None,
        *,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        gate_activation: Activation | None = None,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
    ):
        self.name = _resolve_layer_name(name, type(self).__name__, "n_in")
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.activation = activation if activation is not None else Tanh()
        self.gate_activation = gate_activation if gate_activation is not None else Sigmoid()
        self.bias = bias

        self.W_ih = trainable_parameter(
            f"{self.name}_W_ih",
            (n_in, self._n_gates * n_hidden),
            weight_initializer,
            XavierNormalInitializer(),
            layer_name=self.name,
        )
        self.W_hh = trainable_parameter(
            f"{self.name}_W_hh",
            (n_hidden, self._n_gates * n_hidden),
            recurrent_initializer,
            OrthogonalInitializer(),
            layer_name=self.name,
        )
        if bias:
            self.b = trainable_parameter(
                f"{self.name}_b",
                (self._n_gates * n_hidden,),
                bias_initializer,
                ZeroInitializer(),
                layer_name=self.name,
            )

    def step(self, x_t: TensorVariable, *state: TensorVariable) -> tuple[TensorVariable, ...]:
        h_prev, c_prev = state

        projected = x_t @ self.W_ih + h_prev @ self.W_hh
        if self.bias:
            projected = projected + self.b
        pre_in, pre_forget, pre_candidate, pre_out = _split_gates(
            projected, self.n_hidden, self._n_gates
        )

        input_gate = self.gate_activation(pre_in)
        forget_gate = self.gate_activation(pre_forget)
        output_gate = self.gate_activation(pre_out)
        candidate = self.activation(pre_candidate)

        memory = forget_gate * c_prev + input_gate * candidate
        return (output_gate * self.activation(memory), memory)

    def initial_state(self, X: TensorVariable) -> tuple[TensorVariable, ...]:
        # One variable in both slots: scan gives every carried state its own inner input regardless.
        zeros = _zero_state(X, self.n_hidden, self.W_ih, self.W_hh)
        return (zeros, zeros)


class LSTM(Recurrent):
    r"""
    Long short-term memory layer over a sequence: a :class:`Recurrent` scanning an :class:`LSTMCell`.

    Takes the cell's arguments directly, for the common case where a network wants a plain recurrence
    and no cell of its own. The parameters live on the cell, as ``lstm.cell.W_ih``. See
    :class:`LSTMCell` for the recurrence itself and :class:`Recurrent` for the axes. The layer returns
    :math:`h` at every step; the memory :math:`c` stays inside the loop.

    Parameters
    ----------
    name : str or None
        Name prefix for the layer's parameters. Defaults to "LSTM" when None.
    n_in : int
        Size of the input feature axis.
    n_hidden : int
        Size of the hidden state, and of the memory it carries alongside.
    activation : Activation, optional
        Applied to the candidate and to the memory on the way out. Default is
        :class:`~pytensor_ml.activations.Tanh`.
    gate_activation : Activation, optional
        Applied to the input, forget and output gates. Default is
        :class:`~pytensor_ml.activations.Sigmoid`.
    bias : bool, optional
        Add the learned shift. Default is True.
    weight_initializer : Initializer, optional
        How :math:`W_{ih}` is drawn. Xavier normal when omitted.
    recurrent_initializer : Initializer, optional
        How :math:`W_{hh}` is drawn. Orthogonal when omitted.
    bias_initializer : Initializer, optional
        How :math:`b` is drawn. Zeros when omitted.
    reverse : bool, optional
        Run the sequence backward, with the output still aligned to the input. Default is False.

    Examples
    --------
    Carries a cell state alongside the hidden state, with gates deciding what to keep. That extra path is
    what lets it hold information over far longer sequences than :class:`RNN`:

    .. code-block:: python

        from pytensor_ml.layers import LSTM, Input

        X = Input("X", shape=(None, 200, 16))
        hidden_states = LSTM("lstm", n_in=16, n_hidden=32)(X)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        n_in: int,
        n_hidden: int,
        activation: Activation | None = None,
        bias: bool = True,
        gate_activation: Activation | None = None,
        weight_initializer: Initializer | None = None,
        recurrent_initializer: Initializer | None = None,
        bias_initializer: Initializer | None = None,
        reverse: bool = False,
    ):
        name = name if name else "LSTM"
        super().__init__(
            LSTMCell(
                name,
                n_in=n_in,
                n_hidden=n_hidden,
                activation=activation,
                bias=bias,
                gate_activation=gate_activation,
                weight_initializer=weight_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
            ),
            name=name,
            reverse=reverse,
        )


class Bidirectional(Layer):
    """
    Read a sequence in both directions and concatenate what each pass saw.

    A forward layer's output at step ``t`` summarizes everything up to ``t``, and a backward layer's
    summarizes everything from ``t`` on, so the two together give each step the whole sequence. Both
    outputs stay aligned to the input's time axis, so the concatenation joins the two views of the same
    step; the result is ``(..., time, n_forward + n_backward)``.

    The two layers are separate objects with separate parameters, which is what lets each direction
    learn its own recurrence. Their direction is this wrapper's to choose: whatever ``reverse`` they
    carry is ignored here, and neither layer is changed by being wrapped.

    Parameters
    ----------
    forward : Recurrent
        Run over the sequence as given.
    backward : Recurrent
        Run over the sequence from its last step to its first.
    name : str or None
        Name for the layer's output. Defaults to "Bidirectional" when None.

    Examples
    --------
    Run one layer forwards and another backwards, concatenating their outputs, so every step sees the whole
    sequence. Give the two directions separate layers -- sharing one would tie their weights:

    .. code-block:: python

        from pytensor_ml.layers import GRU, Bidirectional, Input

        X = Input("X", shape=(None, 200, 16))
        forward = GRU("forward", n_in=16, n_hidden=32)
        backward = GRU("backward", n_in=16, n_hidden=32)

        hidden_states = Bidirectional(forward, backward)(X)
    """

    def __init__(self, forward: Recurrent, backward: Recurrent, *, name: str | None = None):
        if forward is backward:
            raise ValueError(
                "Bidirectional needs two layers so each direction has its own parameters to learn its "
                "own recurrence. Build a second one, with its own name."
            )
        self.forward = forward
        self.backward = backward
        self.name = _resolve_layer_name(name, type(self).__name__)

    def __call__(self, X: pt.TensorLike, *, mask: pt.TensorLike | None = None) -> TensorVariable:
        """
        Run both directions over ``X`` and concatenate them on the feature axis.

        Parameters
        ----------
        X : TensorVariable
            Input sequence, shape ``(..., time, n_in)``.
        mask : TensorVariable, optional
            Which steps are real, shape ``(..., time)``. Both directions read it, which is what keeps
            the backward pass from starting on the padding. Every step counts when omitted.

        Returns
        -------
        outputs : TensorVariable
            Both directions' outputs, shape ``(..., time, n_forward + n_backward)``.
        """
        out = pt.concatenate(
            [
                self.forward(X, reverse=False, mask=mask),
                self.backward(X, reverse=True, mask=mask),
            ],
            axis=-1,
        )
        out.name = f"{self.name}_output"
        return out


def _zero_state(X: TensorVariable, n_hidden: int, *parameters: TensorVariable) -> TensorVariable:
    """One zero state carrying ``X``'s batch axes, at the dtype ``X`` and ``parameters`` promote to."""
    state_dtype = np.result_type(X.dtype, *(parameter.dtype for parameter in parameters))
    return pt.zeros((*X.shape[:-2], n_hidden), dtype=state_dtype)


def _split_gates(projection: TensorVariable, n_hidden: int, n_gates: int) -> list[TensorVariable]:
    """Cut a stacked projection into one part per gate, in torch's gate order."""
    # Split rather than a slice per gate: its gradient is one Join, where the slices would each
    # accumulate into a zero buffer.
    return pt.split(projection, [n_hidden] * n_gates, n_splits=n_gates, axis=-1)


__all__ = [
    "GRU",
    "LSTM",
    "RNN",
    "Bidirectional",
    "ElmanCell",
    "GRUCell",
    "LSTMCell",
    "Recurrent",
    "RecurrentCell",
]
