# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
from allo._mlir.dialects import (
    allo as allo_d,
)
from allo._mlir.ir import AffineMap, AffineMapAttr


@register_builtin_handler("constrcut_stream")
class StreamHandler(BuiltinHandler):

    def build(self, node, *args):
        assert isinstance(node.args[0], ast.Name)
        name = node.args[0].id
        dtype, shape, _, is_unsign = self.builder.parse_type_ann(node.args[1])
        stream_type = allo_d.StreamType.get(dtype.build(), depth=dtype.depth)
        return allo_d.stream_global(
            name, stream_type, shape, ip=self.builder.get_global_ip()
        )  # FIXME: unsigned


@register_builtin_handler("put")
class StreamPutHandler(BuiltinHandler):

    def build(self, node, *args):
        assert isinstance(node.args[0], ast.Name)
        assert isinstance(node.args[1], ast.Tuple)
        name = node.args[0].id
        indices, ivs, symbols = [], [], []
        for elt in node.args[1].elts:
            aff = self.builder.get_affine_expr(elt, ivs, symbols)
            assert aff is not None
            indices.append(aff)
        affine_map = AffineMap.get(
            dim_count=len(ivs), symbol_count=len(symbols), exprs=indices
        )
        value = self.builder.get_op_result(self.builder.visit(node.args[2]))
        allo_d.put_stream_global(
            name,
            ivs + symbols,
            value,
            AffineMapAttr.get(affine_map),
            ip=self.builder.get_ip(),
        )


@register_builtin_handler("get")
class StreamGetHandler(BuiltinHandler):

    def build(self, node, *args):
        assert isinstance(node.args[0], ast.Name)
        assert isinstance(node.args[1], ast.Tuple)
        name = node.args[0].id
        indices, ivs, symbols = [], [], []
        for elt in node.args[1].elts:
            aff = self.builder.get_affine_expr(elt, ivs, symbols)
            assert aff is not None
            indices.append(aff)
        affine_map = AffineMap.get(
            dim_count=len(ivs), symbol_count=len(symbols), exprs=indices
        )
        result, is_unsign = self.builder.build_type(node.args[2])  # FIXME: unsigned
        return allo_d.get_stream_global(
            result,
            name,
            ivs + symbols,
            AffineMapAttr.get(affine_map),
            ip=self.builder.get_ip(),
        )
