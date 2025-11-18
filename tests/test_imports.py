"""
Simple pytest to assert core modules import correctly. This test ensures the
project skeleton is importable and that Python module paths are resolved.

Note: This does not execute any runtime logic from the modules — it only
imports them to surface syntax/typo/import errors early.
"""

import importlib


def test_core_modules_import():
    modules = [
        "src.core_config",
        "src.ingestion",
        "src.graph_nodes",
        "src.workflow_builder",
        "main",
    ]

    for mod in modules:
        importlib.import_module(mod)
