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
from allo._mlir.ir import IntegerType, BF16Type, F16Type, F32Type, F64Type, UnitAttr


@register_builtin_handler("Add")
class AddHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, is_unsigned = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                op = arith_d.AddIOp(left, right, ip=self.builder.get_ip())
            elif isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                op = arith_d.AddFOp(left, right, ip=self.builder.get_ip())
            else:
                op = allo_d.AddFixedOp(left, right, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        else:
            assert result_type == left.type, "hard constraint of linalg_d.add failed"
            alloc_op = self.builder.build_buffer(left.type, is_unsigned)
            with self.builder.get_ip():
                linalg_d.add(left, right, outs=[alloc_op])
            return alloc_op


@register_builtin_handler("Sub")
class SubHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, is_unsigned = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                op = arith_d.SubIOp(left, right, ip=self.builder.get_ip())
            elif isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                op = arith_d.SubFOp(left, right, ip=self.builder.get_ip())
            else:
                op = allo_d.SubFixedOp(left, right, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        else:
            assert result_type == left.type, "hard constraint of linalg_d.sub failed"
            alloc_op = self.builder.build_buffer(left.type, is_unsigned)
            with self.builder.get_ip():
                linalg_d.sub(left, right, outs=[alloc_op])
            return alloc_op


@register_builtin_handler("Mult")
class MultHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, is_unsigned = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                op = arith_d.MulIOp(left, right, ip=self.builder.get_ip())
            elif isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                op = arith_d.MulFOp(left, right, ip=self.builder.get_ip())
            else:
                op = allo_d.MulFixedOp(left, right, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        else:
            assert result_type == left.type, "hard constraint of linalg_d.mul failed"
            alloc_op = self.builder.build_buffer(left.type, is_unsigned)
            with self.builder.get_ip():
                linalg_d.mul(left, right, outs=[alloc_op])
            return alloc_op


@register_builtin_handler("Div")
class DivHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, is_unsigned = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                if is_unsigned:
                    op = arith_d.DivUIOp(left, right, ip=self.builder.get_ip())
                    op.attributes["unsigned"] = UnitAttr.get()
                    return op
                return arith_d.DivSIOp(left, right, ip=self.builder.get_ip())
            elif isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                return arith_d.DivFOp(left, right, ip=self.builder.get_ip())
            else:
                op = allo_d.DivFixedOp(left, right, ip=self.builder.get_ip())
                if is_unsigned:
                    op.attributes["unsigned"] = UnitAttr.get()
                return op
        else:
            assert result_type == left.type, "hard constraint of linalg_d.div failed"
            alloc_op = self.builder.build_buffer(left.type, is_unsigned)
            with self.builder.get_ip():
                linalg_d.div(left, right, outs=[alloc_op])
            return alloc_op


@register_builtin_handler("FloorDiv")
class FloorDivHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, is_unsigned = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType) and not is_unsigned:
                return arith_d.FloorDivSIOp(left, right, ip=self.builder.get_ip())
        raise RuntimeError("not supported")


@register_builtin_handler("Mod")
class ModHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        result_type, is_unsigned = self.builder.build_type(args_[2])
        if len(getattr(left.type, "shape", [])) == 0:
            # scalar
            if isinstance(left.type, IntegerType):
                if is_unsigned:
                    op = arith_d.RemUIOp(left, right, ip=self.builder.get_ip())
                    op.attributes["unsigned"] = UnitAttr.get()
                    return op
                return arith_d.RemSIOp(left, right, ip=self.builder.get_ip())
            if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
                return arith_d.RemFOp(left, right, ip=self.builder.get_ip())
        raise RuntimeError("not supported")


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
