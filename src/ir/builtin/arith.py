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


def type_compatible(types):
    """
    Helper function to check if all types are compatible. (to match linalg_d op's type constraint)
    """
    if len(types) <= 1:
        return True
    ref = types[0]
    for t in types[1:]:
        if not type(t) is type(ref):
            return False
        if t.element_type != ref.element_type or t.shape != ref.shape:
            return False
    return True


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
            if not type_compatible([left.type, right.type, result_type]):
                raise ValueError("hard constraint of linalg_d.add failed")
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
            if not type_compatible([left.type, right.type, result_type]):
                raise ValueError("hard constraint of linalg_d.sub failed")
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
            if not type_compatible([left.type, right.type, result_type]):
                raise ValueError("hard constraint of linalg_d.mul failed")
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
            if not type_compatible([left.type, right.type, result_type]):
                raise ValueError("hard constraint of linalg_d.div failed")
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
    # - equal (mnemonic: `"eq"`; integer value: `0`)
    # - float equal (`"oeq"`; integer value: `1`)
    # - fixed equal (integer value: `0`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        _, is_unsigned = self.builder.build_type(args_[2]) # FIXME: should use operand type
        if isinstance(left.type, IntegerType):
            op = arith_d.CmpIOp(0, left, right, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(1, left, right, ip=self.builder.get_ip())
        # fixed
        op = allo_d.CmpFixedOp(0, left, right, ip=self.builder.get_ip())
        if is_unsigned:
            op.attributes["unsigned"] = UnitAttr.get()
        return op


@register_builtin_handler("NotEq")
class NotEqHandler(BuiltinHandler):
    # - not equal (mnemonic: `"ne"`; integer value: `1`)
    # - float not equal (`"one"` ：integer value: `6`)
    # - fixed not equal (integer value: `1`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        _, is_unsigned = self.builder.build_type(args_[2])
        if isinstance(left.type, IntegerType):
            op = arith_d.CmpIOp(1, left, right, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(6, left, right, ip=self.builder.get_ip())
        # fixed
        op = allo_d.CmpFixedOp(1, left, right, ip=self.builder.get_ip())
        if is_unsigned:
            op.attributes["unsigned"] = UnitAttr.get()
        return op


# Less than
@register_builtin_handler("Lt")
class LtHandler(BuiltinHandler):
    # - signed less than (mnemonic: `"slt"`; integer value: `2`)
    # - unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    # - float less than (integer value: `4`)
    # - fixed less than (integer value: `2`)
    # - unsigned fixed less than (integer value: `6`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        _, is_unsigned = self.builder.build_type(args_[2])
        if isinstance(left.type, IntegerType):
            if is_unsigned:
                op = arith_d.CmpIOp(6, left, right, ip=self.builder.get_ip())
                op.attributes["unsigned"] = UnitAttr.get()
                return op
            return arith_d.CmpIOp(2, left, right, ip=self.builder.get_ip())
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(4, left, right, ip=self.builder.get_ip())
        # fixed
        if is_unsigned:
            op = allo_d.allo_d.CmpFixedOp(6, left, right, ip=self.builder.get_ip())
            op.attributes["unsigned"] = UnitAttr.get()
            return op
        return allo_d.allo_d.CmpFixedOp(2, left, right, ip=self.builder.get_ip())


# Less than or equal
@register_builtin_handler("LtE")
class LtEHandler(BuiltinHandler):
    # - signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
    # - unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    # - float less than or equal (integer value: `5`)
    # - fixed less than or equal (integer value: `3`)
    # - unsigned fixed less than or equal (integer value: `7`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        _, is_unsigned = self.builder.build_type(args_[2])
        if isinstance(left.type, IntegerType):
            if is_unsigned:
                op = arith_d.CmpIOp(7, left, right, ip=self.builder.get_ip())
                op.attributes["unsigned"] = UnitAttr.get()
                return op
            return arith_d.CmpIOp(3, left, right, ip=self.builder.get_ip())
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(5, left, right, ip=self.builder.get_ip())
        # fixed
        if is_unsigned:
            op = allo_d.allo_d.CmpFixedOp(7, left, right, ip=self.builder.get_ip())
            op.attributes["unsigned"] = UnitAttr.get()
            return op
        return allo_d.allo_d.CmpFixedOp(3, left, right, ip=self.builder.get_ip())


# Greater than
@register_builtin_handler("Gt")
class GtHandler(BuiltinHandler):
    # - signed greater than (mnemonic: `"sgt"`; integer value: `4`)
    # - unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    # - float greater than (integer value: `2`)
    # - fixed greater than (integer value: `4`)
    # - unsigned fixed greater than (integer value: `8`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        _, is_unsigned = self.builder.build_type(args_[2])
        if isinstance(left.type, IntegerType):
            if is_unsigned:
                op = arith_d.CmpIOp(8, left, right, ip=self.builder.get_ip())
                op.attributes["unsigned"] = UnitAttr.get()
                return op
            return arith_d.CmpIOp(4, left, right, ip=self.builder.get_ip())
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(2, left, right, ip=self.builder.get_ip())
        # fixed
        if is_unsigned:
            op = allo_d.allo_d.CmpFixedOp(8, left, right, ip=self.builder.get_ip())
            op.attributes["unsigned"] = UnitAttr.get()
            return op
        return allo_d.allo_d.CmpFixedOp(4, left, right, ip=self.builder.get_ip())


# Greater than or equal
@register_builtin_handler("GtE")
class GtEHandler(BuiltinHandler):
    # - signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
    # - unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)
    # - float greater than or equal (integer value: `3`)
    # - fixed greater than or equal (integer value: `5`)
    # - unsigned fixed greater than or equal (integer value: `9`)
    def build(self, node: ast.Call, *args):
        args_ = node.args
        left = self.builder.get_op_result(self.builder.visit(args_[0]))
        right = self.builder.get_op_result(self.builder.visit(args_[1]))
        _, is_unsigned = self.builder.build_type(args_[2])
        if isinstance(left.type, IntegerType):
            if is_unsigned:
                op = arith_d.CmpIOp(9, left, right, ip=self.builder.get_ip())
                op.attributes["unsigned"] = UnitAttr.get()
                return op
            return arith_d.CmpIOp(5, left, right, ip=self.builder.get_ip())
        if isinstance(left.type, (BF16Type, F16Type, F32Type, F64Type)):
            return arith_d.CmpFOp(3, left, right, ip=self.builder.get_ip())
        # fixed
        if is_unsigned:
            op = allo_d.allo_d.CmpFixedOp(9, left, right, ip=self.builder.get_ip())
            op.attributes["unsigned"] = UnitAttr.get()
            return op
        return allo_d.allo_d.CmpFixedOp(5, left, right, ip=self.builder.get_ip())
