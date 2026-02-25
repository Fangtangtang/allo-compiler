# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
import allo._mlir.extras.types as mlir_types
from allo._mlir.dialects import func as func_d, memref as memref_d
from allo._mlir.ir import FlatSymbolRefAttr, UnitAttr
import allo._mlir.extras.dialects.func as func


@register_builtin_handler("get_wid")
class WidHandler(BuiltinHandler):

    def build(self, node: ast.Call, *args):
        targets = args[0]
        num = len(targets)
        callee = self.builder.current_func.name.value
        grid_name = self.builder.symbol_table.mangle_grid_name(callee)
        builtin_func = f"{grid_name}.get_wid"
        # insert function declaration in global
        results = [mlir_types.index()] * num
        with self.builder.get_global_ip():
            func.function(builtin_func, [], results, is_private=True)
        # call function in work
        op = func_d.CallOp(
            results, FlatSymbolRefAttr.get(builtin_func), [], ip=self.builder.get_ip()
        )
        for i, target in enumerate(targets):
            assert isinstance(target, ast.Name)
            self.builder.reserved_bindings[target.id] = op.results[i]


@register_builtin_handler("get_mem")
class MemoryGetHandler(BuiltinHandler):
    def build(self, node, *args):
        targets = args[0]
        assert len(targets) == 1
        symbol_name = node.args[0].id
        dtype, hint = self.builder.build_type(node.args[1], True)
        op = memref_d.GetGlobalOp(dtype, symbol_name, ip=self.builder.get_ip())
        op.attributes[hint] = UnitAttr.get()
        self.builder.reserved_bindings[targets[0].id] = op.result
