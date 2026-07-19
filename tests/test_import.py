from importlib import import_module


def test_package_can_be_imported() -> None:
    assert import_module("ptbxl") is not None
