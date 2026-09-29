"""Test-only failures at the Max API boundary; production has no failure switches."""

from contextlib import contextmanager
from unittest.mock import patch


class _Proxy:
    def __init__(self, wrapped, **overrides):
        self._wrapped = wrapped
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


@contextmanager
def inject_rotation_failure(adapter, runtime, write_number, node_contains=""):
    count = 0

    def set_transform(node, kind, value, key):
        nonlocal count
        result = runtime.biped.setTransform(node, kind, value, key)
        if str(kind).lower() == "rotation" and node_contains in str(node.name):
            count += 1
            if count == write_number:
                raise RuntimeError("Injected test rotation failure")
        return result

    biped = _Proxy(runtime.biped, setTransform=set_transform)
    with patch.object(adapter, "runtime", return_value=_Proxy(runtime, biped=biped)):
        yield
