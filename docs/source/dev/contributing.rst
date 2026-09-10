Contributing
============

Contributions are welcome, from a typo fix to a new layer family.

**Every pull request needs an issue first.** Open one describing what you mean
to change, and wait for a reply before writing the code. The design may already
be settled, the behaviour may be deliberate, or someone may be halfway through
the same work. Link the issue from the pull request.

Setting up
----------

Environments are managed with `pixi <https://pixi.sh>`_. Install it once, and
every environment CI uses builds itself on first run:

.. code-block:: bash

    git clone https://github.com/pymc-devs/pytensor-ml.git
    cd pytensor-ml
    pixi run lint-install   # install the pre-commit hooks

Each environment is defined in ``pyproject.toml``:

.. list-table::
    :header-rows: 1
    :widths: 16 84

    * - Environment
      - Contents
    * - ``default``
      - Runtime stack, test tools and lint tools. What a bare ``pixi shell``
        gives you.
    * - ``test``
      - Runtime stack and test tools, and deliberately no backend.
    * - ``lint``
      - ``mypy`` and ``pre-commit``.
    * - ``jax``, ``torch``, ``mlx``
      - ``test`` plus that one backend, for the dispatch suites.
    * - ``docs``
      - The Sphinx stack.
    * - ``notebook``
      - JupyterLab and the plotting stack, for the notebooks under ``examples/``.

The ``test``, ``lint`` and ``notebook`` features are declared once, as
``[project.optional-dependencies]`` in ``pyproject.toml``. Pixi reads each
extra as a feature of the same name, so ``pip install ".[test]"`` and
``pixi run -e test`` install the same set, and a conda table for the same
feature adds to it rather than replacing it -- which is how the compiled
packages get pinned to conda-forge builds.

``pixi.lock`` is checked in, and it has to stay that way. It pins the exact
environment every job resolves to, so a run that works for you works for
everyone else, for CI and for Read the Docs. If a change of yours updates the
lock file, commit it in the same PR as the change that caused it. A lock file
that has drifted out of sync with ``pyproject.toml`` is worse than no lock file,
because it fails somewhere far from the change that broke it.

Running the tests
-----------------

.. code-block:: bash

    pixi run test                            # everything a core job runs
    pixi run test tests/test_layers.py       # one file, while iterating
    pixi run mypy                            # the type check CI runs
    pixi run lint                            # ruff and the rest, through pre-commit

The backend dispatch tests under ``tests/dispatch/`` need that backend
installed, and the ``test`` environment deliberately installs none of them. One
CI job per backend covers those, and running the core suite with no backend
present proves the library does not quietly depend on one. Run one of the
backend environments to cover them locally:

.. code-block:: bash

    pixi run -e jax pytest tests/dispatch/jax/
    pixi run -e torch pytest tests/dispatch/pytorch/
    pixi run -e mlx pytest tests/dispatch/mlx/     # Apple silicon only

Docs are built from their own environment:

.. code-block:: bash

    pixi run docs-build     # build once into docs/build/html
    pixi run docs-serve     # rebuild on save, served at http://localhost:8000

The example notebooks get their own environment:

.. code-block:: bash

    pixi run notebook       # JupyterLab, rooted at examples/

**A new test file has to be added to a CI group.** ``tests/test_workflow_groups.py``
reads ``.github/workflows/run_tests.yml`` and fails when a test file is in no
group, in two groups, or named by a group but missing from disk. Add the file to
one of the ``test-subset`` entries in the same PR.

Style and typing
----------------

``ruff`` formats and lints, ``mypy`` type-checks ``pytensor_ml/``, and both run
in CI, so a clean ``pre-commit run --all-files`` is the cheapest way to keep a
PR green. There is no allowlist of expected type failures: the codebase is
mypy-clean, and it stays that way. For the conventions no tool enforces, see
the :doc:`style guide <style_guide>`.

Docstrings are numpydoc. Every public entrypoint carries an ``Examples``
section whose code is complete and runnable -- imports included, no ``>>>``
prompts, so a reader can paste it straight into a script. Run a new example
before you submit it; nothing checks that for you yet.

Working with LLMs
-----------------

LLM-assisted contributions are welcome, on two conditions.

**Disclose it.** Say in the pull request that a model was involved, and roughly
how much. A tab-completion here and there is not the same as a generated module,
and a reviewer reads the two differently.

**You are responsible for every line you submit, not the model.** You understand
what the code does and why, you can defend each decision in review, and you have
run it. Review comments come to you, and "the model wrote it that way" answers
nothing. Code that nobody understands costs more to maintain than no
contribution at all, and a pull request whose author cannot explain it will be
closed.
