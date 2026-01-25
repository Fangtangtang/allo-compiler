# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union
from collections.abc import Callable
import ast
from .utils import parse_ast, SymbolTable


class ASTProcessor(ast.NodeTransformer):
    def __init__(self, symbol_table: SymbolTable, global_symbols: dict):
        super().__init__()
        self.symbol_table: SymbolTable = symbol_table
        self.global_symbols: dict = global_symbols

    def process(self, fn: Union[Callable, str], instantiate: list = None):
        """
        Process the input function.

        Args:
            fn: The function to process.
            instantiate: The arguments to instantiate the function. default to None.
        """
        ast_module: ast.Module = parse_ast(fn)
        if instantiate is not None:
            # if instantiate is not None, we need to use the args to instantiate the unique function
            assert len(ast_module.body) == 1 and isinstance(
                ast_module.body[0], ast.FunctionDef
            )
            raise NotImplementedError
        else:
            for stmt in ast_module.body:
                self.visit(stmt)

    def visit_Name(self, node: ast.Name):
        raise NotImplementedError

    def visit_Constant(self, node: ast.Constant):
        raise NotImplementedError

    def visit_Tuple(self, node: ast.Tuple):
        raise NotImplementedError

    def visit_Dict(self, node: ast.Dict):
        raise NotImplementedError

    def visit_Index(self, node: ast.Index):
        raise NotImplementedError

    def visit_Attribute(self, node: ast.Attribute):
        raise NotImplementedError

    def visit_Subscript(self, node: ast.Subscript):
        raise NotImplementedError

    def visit_ExtSlice(self, node: ast.ExtSlice):
        raise NotImplementedError

    def visit_Slice(self, node: ast.Slice):
        raise NotImplementedError

    def visit_UnaryOp(self, node: ast.UnaryOp):
        raise NotImplementedError

    def visit_BinOp(self, node: ast.BinOp):
        raise NotImplementedError

    def visit_BoolOp(self, node: ast.BoolOp):
        raise NotImplementedError

    def visit_Compare(self, node: ast.Compare):
        raise NotImplementedError

    def visit_Assign(self, node: ast.Assign):
        raise NotImplementedError

    def visit_AugAssign(self, node: ast.AugAssign):
        raise NotImplementedError

    def visit_AnnAssign(self, node: ast.AnnAssign):
        raise NotImplementedError

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, {ast.Call, ast.Constant}):
            return self.visit(node.value)
        raise RuntimeError(f"Unsupported expression: {node.value}")

    def visit_For(self, node: ast.For):
        raise NotImplementedError

    def visit_While(self, node: ast.While):
        raise NotImplementedError

    def visit_If(self, node: ast.If):
        raise NotImplementedError

    def visit_IfExp(self, node: ast.IfExp):
        raise NotImplementedError

    def visit_Call(self, node: ast.Call):
        raise NotImplementedError

    def visit_Return(self, node: ast.Return):
        raise NotImplementedError

    def visit_With(self, node: ast.With):
        raise NotImplementedError

    def visit_FunctionDef(self, node: ast.FunctionDef):
        raise NotImplementedError

    # ----- invalid syntax -----

    def visit_Break(self, node: ast.Break):
        raise RuntimeError("Break statement is not supported")

    def visit_Continue(self, node: ast.Continue):
        raise RuntimeError("Continue statement is not supported")
