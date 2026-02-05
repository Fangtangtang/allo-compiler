# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
from allo._mlir.dialects import arith as arith_d, allo as allo_d, linalg as linalg_d
from allo._mlir.ir import IntegerType
from allo.ir.types import (
    AlloType,
    Index,
    Float,
    Int,
    UInt,
    Fixed,
    UFixed,
    int32,
    bool as allo_bool,
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
        handler_name = None
        if (type(src_type), type(res_type)) in cast_map:
            handler_name = cast_map[(type(src_type), type(res_type))]
        elif isinstance(src_type, (Int, UInt)) and isinstance(res_type, (Int, UInt)):
            if src_type.bits > res_type.bits:
                handler_name = "cast_int_trunc"
            elif src_type.bits < res_type.bits:
                if isinstance(res_type, UInt):
                    handler_name = "cast_int_zext"
                else:
                    handler_name = "cast_int_sext"
            else:  # same bits, signed <-> unsigned
                handler_name = "cast_int_signedness"
        elif isinstance(src_type, Float) and isinstance(res_type, Float):
            if src_type.bits > res_type.bits:
                handler_name = "cast_float_trunc"
            elif src_type.bits < res_type.bits:
                handler_name = "cast_float_ext"
            else:  # same bits, same float type
                handler_name = "cast_float_identity"
        else:
            raise NotImplementedError
        if handler_name is None:
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
        mlir_type, is_signed = self.get_result_type(node)
        return arith_d.IndexCastOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_si_to_fp")
class SIToFPCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return arith_d.SIToFPOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_ui_to_fp")
class UIToFPCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return arith_d.UIToFPOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_fp_to_si")
class FPToSICastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return arith_d.FPToSIOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_fp_to_ui")
class FPToUICastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return arith_d.FPToUIOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_float_to_fixed")
class FloatToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return allo_d.FloatToFixedOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_fixed_to_float")
class FixedToFloatCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return allo_d.FixedToFloatOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_fixed_to_int")
class FixedToIntCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return allo_d.FixedToIntOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_int_to_fixed")
class IntToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return allo_d.IntToFixedOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_fixed_to_fixed")
class FixedToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return allo_d.FixedToFixedOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_int")
class IntCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        # Note: redundant logic here if we rely on infer, but keeping it for safety if CastHandler.infer falls back to "cast_int"
        src_type = val.type
        dst_type, is_signed = self.get_result_type(node)
        src_width = src_type.width
        dst_width = dst_type.width

        if src_width > dst_width:
            return arith_d.TruncIOp(dst_type, val, ip=self.builder.get_ip())
        elif src_width < dst_width:
            # Default to Signed Extension if unknown
            return arith_d.ExtSIOp(dst_type, val, ip=self.builder.get_ip())
        else:
            return val


@register_builtin_handler("cast_int_sext")
class IntSExtCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return arith_d.ExtSIOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_int_zext")
class IntZExtCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return arith_d.ExtUIOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_int_trunc")
class IntTruncCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        mlir_type, is_signed = self.get_result_type(node)
        return arith_d.TruncIOp(mlir_type, val, ip=self.builder.get_ip())


@register_builtin_handler("cast_float")
class FloatCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        dst_type, is_signed = self.get_result_type(node)
        src_width = val.type.width
        dst_width = dst_type.width
        if src_width > dst_width:
            return arith_d.TruncFOp(dst_type, val, ip=self.builder.get_ip())
        elif src_width < dst_width:
            return arith_d.ExtFOp(dst_type, val, ip=self.builder.get_ip())
        return val


@register_builtin_handler("cast_float_to_index")
class FloatToIndexCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        # FP -> UI -> Index
        inter_type = IntegerType.get_signless(32)
        op = arith_d.FPToUIOp(inter_type, val, ip=self.builder.get_ip())
        mlir_type, is_signed = self.get_result_type(node)
        return arith_d.IndexCastOp(mlir_type, op.result, ip=self.builder.get_ip())


@register_builtin_handler("cast_index_to_float")
class IndexToFloatCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        # Index -> SI -> FP
        inter_type = IntegerType.get_signless(32)
        op = arith_d.IndexCastOp(inter_type, val, ip=self.builder.get_ip())
        mlir_type, is_signed = self.get_result_type(node)
        return arith_d.SIToFPOp(mlir_type, op.result, ip=self.builder.get_ip())


@register_builtin_handler("cast_index_to_fixed")
class IndexToFixedCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        inter_type = IntegerType.get_signless(32)
        op = arith_d.IndexCastOp(inter_type, val, ip=self.builder.get_ip())
        mlir_type, is_signed = self.get_result_type(node)
        return allo_d.IntToFixedOp(mlir_type, op.result, ip=self.builder.get_ip())


@register_builtin_handler("cast_fixed_to_index")
class FixedToIndexCastHandler(CastHandler):
    def build(self, node: ast.Call, *args):
        val = self.get_operand(node)
        inter_type = IntegerType.get_signless(32)
        op = allo_d.FixedToIntOp(inter_type, val, ip=self.builder.get_ip())
        mlir_type, is_signed = self.get_result_type(node)
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
