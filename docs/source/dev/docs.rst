Working on the docs
===================

This page covers how the documentation system is wired and the workflows
needed to extend it — adding an API page, a notebook example, or a citation.

Building locally
----------------

The docs build from ``docs/`` using the ``pytensor_ml-docs`` conda
environment:

.. code-block:: bash

    conda env update -f conda_envs/environment-docs.yml
    cd docs
    make show         # build + open the rendered HTML in the default browser
    make livehtml     # auto-rebuild + auto-refresh on every save (sphinx-autobuild)
    make clean        # wipe build/ and all generated source (gallery, thumbnails, autosummary stubs)

Read the Docs builds the same way; ``.readthedocs.yaml`` points at the same
conda env and ``docs/source/conf.py``.

Layout
------

Source content lives under ``docs/source/``:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Path
      - Purpose
    * - ``index.rst``
      - Landing page + top-level toctree.
    * - ``api.rst`` + ``api/*.rst``
      - Autosummary entry points; one file per public submodule.
    * - ``install.rst``
      - Installation guide. Hand-written narrative.
    * - ``examples/*.rst``
      - One page per notebook category, in the order set by ``GALLERY_PAGES`` in
        ``generate_gallery.py``. A category with no notebooks writes no page, so add its entry to the
        toctree in ``index.rst`` along with the first notebook. **Generated** at build time.
    * - ``examples/<page>/**/*.ipynb``
      - Notebook copies staged from ``examples/``, under the same folders. **Generated**.
    * - ``dev/``
      - This page, the contributing guide and the style guide.
    * - ``references.bib``
      - BibTeX entries; cited via ``{cite:t}`` or ``{cite:p}``.
    * - ``_templates/autosummary/``
      - Sphinx autosummary class template (per-method subpages).

Build-time-generated paths are gitignored via ``docs/.gitignore``; never
commit anything under ``source/_thumbnails/``, ``source/examples/<category>/``,
``source/api/**/generated/``, or ``source/examples/gallery.rst``.

The custom Sphinx extension lives at ``docs/sphinxext/generate_gallery.py``.
It discovers notebooks under ``examples/``, copies them into
``docs/source/examples/<category>/``, extracts thumbnails, and emits
``examples/gallery.rst``.

Adding an API page
------------------

Public objects are documented through ``autosummary`` stubs, so a new class or
function only needs an entry in the relevant ``docs/source/api/*.rst`` file
under the right section heading; the stub page is generated on the next build.
A whole new module gets its own ``api/<module>.rst`` plus a line in the
``api.rst`` toctree.

Adding a notebook example
-------------------------

Drop the ``.ipynb`` under ``examples/``. The ``generate_gallery`` extension
auto-discovers it on the next build, extracts the last image output as a
thumbnail, and emits a grid card. Ship the notebook with its outputs already
rendered — ``nb_execution_mode`` is ``"off"``, so nothing is re-run at build
time.

To group notebooks into named categories, create subdirectories under
``examples/`` (e.g. ``examples/introductory/foo.ipynb``). The subdir name
becomes the category id; pretty titles are looked up in ``CATEGORY_TITLES``
inside ``generate_gallery.py`` and fall back to title-casing the folder name.

Notebooks must be tracked by git to appear in the gallery, so untracked
work-in-progress notebooks under ``examples/`` don't pollute the build.
