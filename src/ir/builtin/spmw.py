# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
from allo._mlir.dialects import (
    arith as arith_d,
    shard as shard_d,
)
from allo._mlir.ir import DenseI16ArrayAttr


@register_builtin_handler("get_wid")
class WidHandler(BuiltinHandler):

    def build(self, node: ast.Call, *args):
        targets = args[0]
        num = len(targets)
        grid_name = self.builder.symbol_table.mangle_grid_name(
            self.builder.current_func.name.value
        )
        op = shard_d.ProcessMultiIndexOp(
            [arith_d.IndexType.get()] * num,
            grid_name,
            axes=DenseI16ArrayAttr.get(range(num)),
            ip=self.builder.get_ip(),
        )
        for i, target in enumerate(targets):
            assert isinstance(target, ast.Name)
            self.builder.reserved_bindings[target.id] = op.results[i]
