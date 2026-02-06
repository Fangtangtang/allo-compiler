# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
from allo._mlir.dialects import arith as arith_d, allo as allo_d, linalg as linalg_d
from allo._mlir.ir import IntegerType, UnitAttr
from allo.ir.types import (
    AlloType,
    Index,
    Float,
    Int,
    UInt,
    Fixed,
    UFixed,
    float16,
    bfloat16,
)


@register_builtin_handler("constant")
class ConstantHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        assert isinstance(args_[0], ast.Constant)
        assert isinstance(args_[1], ast.Subscript)
        dtype, _ = self.builder.build_type(args_[1])
        const_op = arith_d.ConstantOp(dtype, args_[0].value, ip=self.builder.get_ip())
        return const_op


@register_builtin_handler("cast")
class CastHandler(BuiltinHandler):
    @staticmethod
    def infer(*args):
        cast_map = {
            # Index <-> UInt/Int
            (Int, Index): "cast_index",
            (UInt, Index): "cast_index",
            (Index, Int): "cast_index",
            (Index, UInt): "cast_index",
            # UInt/Int <-> Float
            (Int, Float): "cast_si_to_fp",
            (UInt, Float): "cast_ui_to_fp",
            (Float, Int): "cast_fp_to_si",
            (Float, UInt): "cast_fp_to_ui",
            # Float <-> Fixed/UFixed
            (Float, Fixed): "cast_float_to_fixed",
            (Float, UFixed): "cast_float_to_fixed",
            (Fixed, Float): "cast_fixed_to_float",
            (UFixed, Float): "cast_fixed_to_float",
            # Int/UInt <-> Fixed/UFixed
            (Fixed, Int): "cast_fixed_to_int",
            (Fixed, UInt): "cast_fixed_to_int",
            (UFixed, Int): "cast_fixed_to_int",
            (UFixed, UInt): "cast_fixed_to_int",
            (Int, Fixed): "cast_int_to_fixed",
            (Int, UFixed): "cast_int_to_fixed",
            (UInt, Fixed): "cast_int_to_fixed",
            (UInt, UFixed): "cast_int_to_fixed",
            # Fixed/UFixed <-> Fixed/UFixed
            (Fixed, Fixed): "cast_fixed_to_fixed",
            (Fixed, UFixed): "cast_fixed_to_fixed",
            (UFixed, Fixed): "cast_fixed_to_fixed",
            (UFixed, UFixed): "cast_fixed_to_fixed",
            # UInt/Int -> UInt/Int
            (Int, Int): "cast_int",
            (UInt, UInt): "cast_int",
            (Int, UInt): "cast_int",
            (UInt, Int): "cast_int",
            # Float -> Float
            (Float, Float): "cast_float",
            # Float -> Index
            (Float, Index): "cast_float_to_index",
            # Index -> Float
            (Index, Float): "cast_index_to_float",
            # Index -> Fixed/UFixed
            (Index, Fixed): "cast_index_to_fixed",
            (Index, UFixed): "cast_index_to_fixed",
            # Fixed/UFixed -> Index
            (Fixed, Index): "cast_fixed_to_index",
            (UFixed, Index): "cast_fixed_to_index",
        }
        src_type, res_type = args[0], args[1]
        # [NOTE]: float16 <-> bfloat16 not supported
        assert not (
            src_type == float16 and res_type == bfloat16
        ), "f16 -> bf16 not supported"
        assert not (
            src_type == bfloat16 and res_type == float16
        ), "bf16 -> f16 not supported"
        if (type(src_type), type(res_type)) in cast_map:
            handler_name = cast_map[(type(src_type), type(res_type))]
        else:
            raise TypeError(f"Invalid casting. src: {src_type}, dst: {res_type}")

        return res_type, src_type, handler_name

    def get_operand(self, node: ast.Call):
        return self.builder.get_op_result(self.builder.visit(node.args[0]))

    def get_result_type(self, node: ast.Call):
        return self.builder.build_type(node.args[1])


@register_builtin_handler("cast_index")
class IndexCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_unsigned = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            # scalar
            op = arith_d.IndexCastOp(mlir_type, val, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        raise NotImplementedError


@register_builtin_handler("cast_si_to_fp")
class SIToFPCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, _ = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            # scalar
            return arith_d.SIToFPOp(mlir_type, val, ip=self.builder.get_ip())
        raise NotImplementedError


@register_builtin_handler("cast_ui_to_fp")
class UIToFPCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, _ = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            # scalar
            return arith_d.UIToFPOp(mlir_type, val, ip=self.builder.get_ip())
        raise NotImplementedError


@register_builtin_handler("cast_fp_to_si")
class FPToSICastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, _ = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            # scalar
            return arith_d.FPToSIOp(mlir_type, val, ip=self.builder.get_ip())
        raise NotImplementedError


@register_builtin_handler("cast_fp_to_ui")
class FPToUICastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, _ = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            # scalar
            return arith_d.FPToUIOp(mlir_type, val, ip=self.builder.get_ip())
        raise NotImplementedError


@register_builtin_handler("cast_float_to_fixed")
class FloatToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_unsigned = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            # scalar
            op = allo_d.FloatToFixedOp(mlir_type, val, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        raise NotImplementedError


@register_builtin_handler("cast_fixed_to_float")
class FixedToFloatCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, _ = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            # scalar
            return allo_d.FixedToFloatOp(mlir_type, val, ip=self.builder.get_ip())
        raise NotImplementedError


@register_builtin_handler("cast_fixed_to_int")
class FixedToIntCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_unsigned = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            # scalar
            op = allo_d.FixedToIntOp(mlir_type, val, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        raise NotImplementedError


@register_builtin_handler("cast_int_to_fixed")
class IntToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_unsigned = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            # scalar
            op = allo_d.IntToFixedOp(mlir_type, val, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        raise NotImplementedError


@register_builtin_handler("cast_fixed_to_fixed")
class FixedToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_unsigned = self.get_result_type(node)
        if len(getattr(mlir_type, "shape", [])) == 0:
            # scalar
            op = allo_d.FixedToFixedOp(mlir_type, val, ip=self.builder.get_ip())
            if is_unsigned:
                op.attributes["unsigned"] = UnitAttr.get()
            return op
        raise NotImplementedError


@register_builtin_handler("cast_int")
class IntCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        dst_type, is_unsigned = self.get_result_type(node)
        if len(getattr(dst_type, "shape", [])) == 0:
            # scalar
            src_width = val.type.width
            dst_width = dst_type.width
            if src_width > dst_width:
                op = arith_d.TruncIOp(dst_type, val, ip=self.builder.get_ip())
                if is_unsigned:
                    op.attributes["unsigned"] = UnitAttr.get()
                return op
            elif src_width < dst_width:
                if is_unsigned:
                    op = arith_d.ExtUIOp(dst_type, val, ip=self.builder.get_ip())
                    op.attributes["unsigned"] = UnitAttr.get()
                    return op
                else:
                    return arith_d.ExtSIOp(dst_type, val, ip=self.builder.get_ip())
            else:
                return val
        else:
            raise NotImplementedError


@register_builtin_handler("cast_float")
class FloatCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        dst_type, _ = self.get_result_type(node)
        if len(getattr(dst_type, "shape", [])) == 0:
            # scalar
            src_width = val.type.width
            dst_width = dst_type.width
            if src_width > dst_width:
                return arith_d.TruncFOp(dst_type, val, ip=self.builder.get_ip())
            elif src_width < dst_width:
                return arith_d.ExtFOp(dst_type, val, ip=self.builder.get_ip())
            else:
                return val
        else:
            raise NotImplementedError


@register_builtin_handler("cast_float_to_index")
class FloatToIndexCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        # FP -> UI -> Index
        op = arith_d.FPToUIOp(
            IntegerType.get_signless(32), val, ip=self.builder.get_ip()
        )
        mlir_type, _ = self.get_result_type(node)
        return arith_d.IndexCastOp(mlir_type, op.result, ip=self.builder.get_ip())


@register_builtin_handler("cast_index_to_float")
class IndexToFloatCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        # Index -> SI -> FP
        op = arith_d.IndexCastOp(
            IntegerType.get_signless(32), val, ip=self.builder.get_ip()
        )
        mlir_type, _ = self.get_result_type(node)
        return arith_d.SIToFPOp(mlir_type, op.result, ip=self.builder.get_ip())


@register_builtin_handler("cast_index_to_fixed")
class IndexToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        op = arith_d.IndexCastOp(
            IntegerType.get_signless(32), val, ip=self.builder.get_ip()
        )
        mlir_type, is_unsigned = self.get_result_type(node)
        op = allo_d.IntToFixedOp(mlir_type, op.result, ip=self.builder.get_ip())
        if is_unsigned:
            op.attributes["unsigned"] = UnitAttr.get()
        return op


@register_builtin_handler("cast_fixed_to_index")
class FixedToIndexCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        op = allo_d.FixedToIntOp(
            IntegerType.get_signless(32), val, ip=self.builder.get_ip()
        )
        mlir_type, _ = self.get_result_type(node)
        return arith_d.IndexCastOp(mlir_type, op.result, ip=self.builder.get_ip())


@register_builtin_handler("broadcast")
class BroadcastHandler(BuiltinHandler):
    def build(self, node: ast.Call, *args):
        args_ = node.args
        origianl = self.builder.get_op_result(self.builder.visit(args_[0]))
        assert isinstance(args_[1], ast.Tuple) and isinstance(args_[2], ast.Subscript)
        dims = [v.value for v in args_[1].elts]
        alloc_op = self.builder.build_buffer(*self.builder.build_type(args_[2]))
        with self.builder.get_ip():
            if len(getattr(origianl.type, "shape", [])) == 0:
                linalg_d.fill(origianl, outs=[alloc_op.result])
            else:
                linalg_d.broadcast(
                    input=origianl, outs=[alloc_op.result], dimensions=dims
                )
        return alloc_op
