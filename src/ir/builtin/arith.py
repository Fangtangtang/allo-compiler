# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
from allo._mlir.dialects import (
    allo as allo_d,
    arith as arith_d,
    memref as memref_d,
    linalg as linalg_d,
)
from allo._mlir.ir import IntegerType, BF16Type, F16Type, F32Type, F64Type


@register_builtin_handler("Add")
class AddHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                return arith_d.AddIOp(left, right, ip=self.builder.get_ip())
            if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                return arith_d.AddFOp(left, right, ip=self.builder.get_ip())
            return allo_d.AddFixedOp(left, right, ip=self.builder.get_ip())
        else:
            # TODO
            # with self.builder.get_ip():
            #     linalg_d.add(left, right)
            raise NotImplementedError


@register_builtin_handler("Sub")
class SubHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("Mult")
class MultHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("Div")
class DivHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("FloorDiv")
class FloorDivHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("Mod")
class ModHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("Eq")
class EqHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("NotEq")
class NotEqHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("Lt")
class LtHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("LtE")
class LtEHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("Gt")
class GtHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError


@register_builtin_handler("GtE")
class GtEHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        raise NotImplementedError
