# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
from allo._mlir.dialects import arith as arith_d


@register_builtin_handler("constant")
class ConstantHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
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
        raise NotImplementedError
