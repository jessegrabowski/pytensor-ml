Installation
============

pytensor_ml runs on Python 3.12 and newer, and depends on PyTensor, NumPy and
safetensors. Nothing else is needed to train a network and save it.

From PyPI
---------

.. code-block:: bash

    pip install pytensor-ml

The distribution is named ``pytensor-ml``; the package you import is
``pytensor_ml``.

From source
-----------

.. code-block:: bash

    git clone https://github.com/pymc-devs/pytensor-ml.git
    cd pytensor-ml
    pip install -e .

Optional extras
---------------

.. code-block:: bash

    pip install "pytensor-ml[examples]"   # matplotlib and tqdm, for the example notebooks
    pip install "pytensor-ml[dev]"        # test, lint and type-checking tools

Backends
--------

A network is a PyTensor graph, so it compiles to whichever backend PyTensor is
pointed at. Numba is the default and arrives with PyTensor itself, so a plain
install already compiles the whole graph rather than stepping through it op by
op. The rest are ordinary packages you install yourself:

.. list-table::
    :header-rows: 1
    :widths: 18 26 56

    * - Mode
      - Install
      - Notes
    * - ``"NUMBA"``
      - included
      - The default, and a dependency of PyTensor, so it is already there.
    * - ``"C"``
      - included
      - Compiles each op to C and calls it from Python. Needs a working C
        compiler, which most systems already have.
    * - ``"JAX"``
      - ``pip install jax``
      - CPU by default; GPU and TPU need the matching hardware-specific
        wheel from the JAX project.
    * - ``"PYTORCH"``
      - ``pip install torch``
      - Runs on whichever devices the installed torch build supports.
    * - ``"MLX"``
      - ``pip install mlx``
      - Apple silicon only, and unavailable on every other platform.

pytensor_ml registers its own kernels for convolution, pooling and attention on
Numba, JAX, PyTorch and MLX. Registration is lazy: nothing imports a backend
until a graph is actually compiled against it, so an installed-but-unused
backend costs nothing at import time, and a missing one is only a problem if you
ask for it.

Development install
-------------------

.. code-block:: bash

    git clone https://github.com/pymc-devs/pytensor-ml.git
    cd pytensor-ml
    pip install -e ".[dev]"
    pre-commit install

The conda environments under ``conda_envs/`` are the reference setup CI runs
against, pinned more tightly than the package metadata. Install one of those
when a failure does not reproduce anywhere else.

See :doc:`/dev/contributing` for the rest of the contributor setup.
