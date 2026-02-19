# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from typing import Union
from collections.abc import Callable
from .ir.config import ir_builder_config_context
from .ir.utils import SymbolTable, get_global_vars
from .ir.ast_processor import ASTProcessor
from .ir.ir_builder import IRBuilder
from allo.backend.llvm import LLVMModule
from allo.backend.hls import HLSModule
from allo.backend.simulator import LLVMOMPModule
from allo._mlir.dialects import allo as allo_d, func as func_d
from allo._mlir.passmanager import PassManager as mlir_pass_manager


def build(fn: Union[Callable, str], instantiate: list = None, typing: str = None):
    typing = "hls" if typing is None else typing
    with ir_builder_config_context(typing):
        symbol_table = SymbolTable()
        ast_processor = ASTProcessor(symbol_table, global_symbols=get_global_vars(fn))
        # process the top function
        node, top_name = ast_processor.process(fn, instantiate=instantiate)
        for name, constant in symbol_table.constants.items():
            print(name, "=", constant.value)
        for op in symbol_table.global_ops:
            print(ast.unparse(op))
        for node in symbol_table.functions.values():
            print(ast.unparse(node), "\n")
        print()
        builder = IRBuilder(symbol_table)
        module = builder.build()
        return module, top_name


def process(fn: Union[Callable, str], instantiate: list = None, typing: str = None):
    """
    Compile the input function.
    """
    module, top_name = build(fn, instantiate, typing)
    print(module)
    return LLVMModule(module, top_name)


def process_spmw(fn: Union[Callable, str], instantiate: list = None):
    """
    Compile the input function in SPMW model.
    """
    module, top_name = build(fn, instantiate)
    print(module)


def to_hls(fn: Union[Callable, str], instantiate: list = None, project=None):
    module, top_name = build(fn, instantiate, "hls")
    print(module)

    for func in module.body.operations:
        if isinstance(func, func_d.FuncOp):
            allo_d.copy_on_write_on_function(func)

    pipeline = "builtin.module(canonicalize)"
    with module.context:
        mlir_pass_manager.parse(pipeline).run(module.operation)

    print(module)
    func_args = {}
    mod = HLSModule(
        module,
        top_func_name=top_name,
        platform="vitis_hls",
        mode="sw_emu",
        project=project if project else "top.prj",
        ext_libs=[],
        func_args=func_args,
        wrap_io=False,
    )

    return mod
