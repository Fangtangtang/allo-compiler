# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from typing import Union
from collections.abc import Callable
from .ir.utils import SymbolTable, get_global_vars
from .ir.ast_processor import ASTProcessor
from .ir.ir_builder import IRBuilder
from allo.backend.llvm import LLVMModule


def process(fn: Union[Callable, str], instantiate: list = None):
    """
    Compile the input function.
    """
    symbol_table = SymbolTable()
    ast_processor = ASTProcessor(symbol_table, global_symbols=get_global_vars(fn))
    # process the top function
    node = ast_processor.process(fn, instantiate=instantiate)
    print(ast.unparse(node), "\n")
    builder = IRBuilder(symbol_table)
    module = builder.build(node)
    print(module)
    return LLVMModule(module, fn.__name__)
