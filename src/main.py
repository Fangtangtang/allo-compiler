# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union
from collections.abc import Callable
from .ir.utils import SymbolTable, get_global_vars
from .ir.ast_processor import ASTProcessor


def process(fn: Union[Callable, str], instantiate: list = None):
    """
    Compile the input function.
    """
    symbol_table = SymbolTable()
    ast_processor = ASTProcessor(symbol_table, global_symbols=get_global_vars(fn))
    # process the top function
    ast_processor.process(fn, instantiate=instantiate)
