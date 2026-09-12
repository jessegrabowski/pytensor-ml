import importlib
import importlib.abc
import importlib.util
import sys

# Pytensor imports its own backend dispatches from the linker at compile time and offers no plugin hook, so
# watch for a backend's dispatch package loading and register ours right after. This keeps jax/mlx off the
# main import path -- they load only when that backend actually compiles a graph.
_REGISTRATIONS = {
    "pytensor.link.jax.dispatch": "pytensor_ml.dispatch.jax",
    "pytensor.link.mlx.dispatch": "pytensor_ml.dispatch.mlx",
    "pytensor.link.numba.dispatch": "pytensor_ml.dispatch.numba",
    "pytensor.link.pytorch.dispatch": "pytensor_ml.dispatch.pytorch",
}


class _RegisterAfterImport(importlib.abc.MetaPathFinder):
    """Import a registration module right after its target backend-dispatch package finishes loading."""

    def find_spec(self, fullname, path=None, target=None):
        registration = _REGISTRATIONS.get(fullname)
        if registration is None:
            return None

        # Resolve the real spec with ourselves out of the way to avoid infinite recursion.
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        load = spec.loader.exec_module

        def exec_module(module):
            load(module)
            _REGISTRATIONS.pop(fullname, None)
            if not _REGISTRATIONS and self in sys.meta_path:
                sys.meta_path.remove(self)
            importlib.import_module(registration)

        spec.loader.exec_module = exec_module
        return spec


# A backend dispatch already loaded before us (e.g. a prior compile in the same process) won't trip the
# finder, so register against it immediately.
for _dispatch_module, _registration in list(_REGISTRATIONS.items()):
    if _dispatch_module in sys.modules:
        _REGISTRATIONS.pop(_dispatch_module)
        importlib.import_module(_registration)

if _REGISTRATIONS:
    sys.meta_path.insert(0, _RegisterAfterImport())
