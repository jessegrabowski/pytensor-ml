Contributing
============

Contributions are welcome, from a typo fix to a new layer family.

**Every pull request needs an issue first.** Open one describing what you mean
to change, and wait for a reply before writing the code. The design may already
be settled, the behaviour may be deliberate, or someone may be halfway through
the same work. Link the issue from the pull request.

Setting up
----------

.. code-block:: bash

    git clone https://github.com/pymc-devs/pytensor-ml.git
    cd pytensor-ml
    pip install -e ".[dev]"
    pre-commit install

The conda environments under ``conda_envs/`` are what CI installs, pinned
harder than the package metadata. Reach for one when a failure reproduces in CI
but not locally:

.. code-block:: bash

    conda env create -f conda_envs/pytensor_ml.yml

Docs are built with pixi, which owns its own environment:

.. code-block:: bash

    pixi run docs-build     # build once into docs/build/html
    pixi run docs-serve     # rebuild on save, served at http://localhost:8000

``pixi.lock`` is checked in, and it has to stay that way. It pins the exact
environment a docs build resolves to, so a build that works for you works for
everyone else and for Read the Docs. If a change of yours updates the lock file,
commit it in the same PR as the change that caused it. A lock file that has
drifted out of sync with ``pyproject.toml`` is worse than no lock file, because
it fails somewhere far from the change that broke it.

Running the tests
-----------------

.. code-block:: bash

    pytest                              # everything a core job runs
    pytest tests/test_layers.py         # one file, while iterating
    python scripts/run_mypy.py          # the type check CI runs

The backend dispatch tests under ``tests/dispatch/`` need that backend
installed, and the core environment deliberately installs none of them. One CI
job per backend covers those, and running the core suite with no backend present
proves the library does not quietly depend on one. Install the backend you are
working on to run its tests locally.

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
