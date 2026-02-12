# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
from allo._mlir.dialects import (
    arith as arith_d,
    func as func_d,
)
from allo._mlir.ir import FlatSymbolRefAttr, StringAttr


@register_builtin_handler("get_wid")
class WidHandler(BuiltinHandler):

    def build(self, node: ast.Call, *args):
        targets = args[0]
        num = len(targets)
        callee = self.builder.current_func.name.value
        grid_name = self.builder.symbol_table.mangle_grid_name(callee)
        builtin_func = f"{grid_name}_get_wid"
        # insert function declaration in global
        results = [arith_d.IndexType.get()] * num
        func_type = func_d.FunctionType.get([], results)
        kernel = func_d.FuncOp(builtin_func, func_type, ip=self.builder.get_global_ip())
        kernel.attributes["sym_visibility"] = StringAttr.get("private")
        # call function in work
        op = func_d.CallOp(
            results, FlatSymbolRefAttr.get(builtin_func), [], ip=self.builder.get_ip()
        )
        for i, target in enumerate(targets):
            assert isinstance(target, ast.Name)
            self.builder.reserved_bindings[target.id] = op.results[i]
