# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
from allo._mlir.dialects import arith as arith_d, memref as memref_d, linalg as linalg_d


@register_builtin_handler("constant")
class ConstantHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args = node.args
        assert isinstance(args[0], ast.Constant) and isinstance(args[1], ast.Subscript)
        dtype = self.builder.build_type(args[1])
        const_op = arith_d.ConstantOp(dtype, args[0].value, ip=self.builder.get_ip())
        return const_op


@register_builtin_handler("cast")
class CastHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("broadcast")
class BroadcastHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args = node.args
        origianl = self.builder.get_op_result(self.builder.visit(args[0]))
        assert isinstance(args[1], ast.Tuple) and isinstance(args[2], ast.Subscript)
        dims = [v.value for v in args[1].elts]
        dtype = self.builder.build_type(args[2])
        alloc_op = memref_d.AllocOp(dtype, [], [], ip=self.builder.get_ip())
        with self.builder.get_ip():
            if len(getattr(origianl.type, "shape", [])) == 0:
                linalg_d.fill(origianl, outs=[alloc_op.result])
            else:
                linalg_d.broadcast(
                    input=origianl, outs=[alloc_op.result], dimensions=dims
                )
        return alloc_op
