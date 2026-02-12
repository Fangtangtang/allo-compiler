# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
import hashlib
import inspect
import numpy as np
from collections.abc import Callable
from types import FunctionType as PyFunctionType
from allo.ir.types import AlloType
from allo.memory import Memory


def get_ast(src) -> ast.FunctionDef:
    assert hasattr(src, "_ast") and hasattr(src, "_type"), "Invalid function"
    node = copy.deepcopy(src._ast)  # get a new copy to avoid overwriting
    node._type = src._type
    return node


class SymbolTable:
    def __init__(self):
        # function name -> function instance (instantiated from templates) node
        self.functions = {}
        # global: constant name -> constant value
        self.constants = {}
        # global: variable name -> variable node
        self.variables = {}

        self.types = {}  # str(dtype) -> AlloType / refinement
        self.global_symbols = {}  # str -> python object

        self.global_ops = []
        # ----- tools -----
        self.tmpl_instantiations = {}  # template name -> instance args -> instance name

    def mangle_template_name(self, name: str, args: list) -> str:
        """
        Name mangling for instantiated functions.

        Args:
            name: The name of the function.
            args: The arguments to instantiate the function. If the last argument is a string, it is used as the user defined suffix of the instantiated function.

        Returns:
            The name of the instantiated function.
        """
        if isinstance(args[-1], str):
            suffix = args.pop()
            return "_" + name + "_" + suffix
        key = tuple(args)
        func_dict = self.tmpl_instantiations.setdefault(name, {})
        if key not in func_dict:
            func_dict[key] = (
                "_" + name + "_" + "_".join(map(str, args)) + "_" + str(len(func_dict))
            )
        return func_dict[key]

    def mangle_with_namespace(self, name: str, namespace: str) -> str:
        return f"__{namespace}__{name}"

    def mangle_grid_name(self, work_name) -> str:
        return f"{work_name}_mesh"

    @staticmethod
    def get_hash(arr):
        assert isinstance(arr, np.ndarray), "only support np.ndarray"
        return hashlib.sha256(
            arr.tobytes() + str((arr.shape, arr.dtype)).encode()
        ).hexdigest()[:16]


def get_global_vars(func):
    def _get_global_vars(_func, skip={"get_global_vars", "process"}, stop={"<module>"}):
        if isinstance(_func, Callable):
            # Discussions: https://github.com/taichi-dev/taichi/issues/282
            global_vars = _func.__globals__.copy()
        else:
            global_vars = {}

        # Get back to outer scopes
        # Mainly used to get the annotation definitions (shape and type),
        # which are probably not defined in __globals__
        frame = inspect.currentframe().f_back
        while frame:
            if frame.f_code.co_name in skip:
                frame = frame.f_back
                continue
            # collect allowed types
            for name, var in frame.f_locals.items():
                # FIXME: find a better way to collect required symbols
                if isinstance(
                    var, (int, float, AlloType, Memory, list)
                ) or inspect.isfunction(var):
                    global_vars[name] = var
            # boundary
            if frame.f_code.co_name in stop:
                break
            frame = frame.f_back

        if isinstance(_func, Callable):
            freevar_names = _func.__code__.co_freevars
            closure = _func.__closure__
            if closure:
                freevar_values = [x.cell_contents for x in closure]
                for name, value in zip(freevar_names, freevar_values):
                    global_vars[name] = value
        return global_vars

    all_globals = {}
    worklist = [func]
    visited_funcs = set()

    while worklist:
        f = worklist.pop()
        if f in visited_funcs:
            continue
        visited_funcs.add(f)

        gv = _get_global_vars(f)
        for name, val in gv.items():
            if name not in all_globals:
                all_globals[name] = val
                # import functions from other files
                if isinstance(val, PyFunctionType):
                    worklist.append(val)

    return all_globals


class Scope:
    def __init__(self):
        self.consts = {}
        self.vars = {}


class BlockScopeGuard:
    def __init__(self, scopes: list[Scope]):
        self.scopes: list[Scope] = scopes

    def __enter__(self):
        self.scopes.append(Scope())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()


class ErrorValue:
    def __init__(self, name, msg):
        self.name = name
        self.msg = msg

    def __getattr__(self, attr):
        raise RuntimeError(f"Use of invalid symbol '{self.name}': {self.msg}")
