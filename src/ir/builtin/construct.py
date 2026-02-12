# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from .handler import BuiltinHandler, register_builtin_handler
from allo._mlir.dialects import (
    allo as allo_d,
    func as func_d,
)
from allo._mlir.ir import FlatSymbolRefAttr, StringAttr


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
        raise NotImplementedError


@register_builtin_handler("get")
class StreamGetHandler(BuiltinHandler):

    def build(self, node, *args):
        raise NotImplementedError
