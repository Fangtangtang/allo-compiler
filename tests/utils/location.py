# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast


def _check_locations(func_node: ast.FunctionDef, label: str = "function"):
    """
    Walk the processed AST and verify every stmt/expr node
    has valid lineno and col_offset attributes.
    """
    missing = []
    for node in ast.walk(func_node):
        if isinstance(node, (ast.stmt, ast.expr)):
            lineno = getattr(node, "lineno", None)
            col = getattr(node, "col_offset", None)
            if lineno is None or col is None:
                missing.append(
                    f"  {type(node).__name__}: lineno={lineno}, col_offset={col}"
                )
            else:
                assert (
                    lineno > 0
                ), f"{label}: {type(node).__name__} has non-positive lineno={lineno}"
                assert (
                    col >= 0
                ), f"{label}: {type(node).__name__} has negative col_offset={col}"
    assert (
        len(missing) == 0
    ), f"{label}: {len(missing)} node(s) missing location info:\n" + "\n".join(missing)
