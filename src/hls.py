# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union
from collections.abc import Callable
from .ir.utils import SymbolTable, get_global_vars
from .ir.ast_preprocessor import ASTPreProcessor
from .ir.ir_builder import IRBuilder
from .passes.instantiate import instantiate_for_hls, instantiate_for_hierarchical_hls
from allo.backend.hls import HLSModule


def to_hls(fn: Union[Callable, str], instantiate: list = None, project=None):
    symbol_table = SymbolTable()
    ast_processor = ASTPreProcessor(symbol_table, global_symbols=get_global_vars(fn))
    # process the top function
    node, top_name = ast_processor.process(fn, instantiate=instantiate)
    builder = IRBuilder(symbol_table)
    module = builder.build()
    parsed = instantiate_for_hierarchical_hls(module, top_name)
    mod = HLSModule(
        parsed,
        top_func_name=top_name,
        platform="vitis_hls",
        mode="sw_emu",
        project=project if project else "top.prj",
        ext_libs=[],
        func_args={},
        wrap_io=True,
    )
    return mod
