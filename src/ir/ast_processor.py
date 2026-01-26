# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Union
from collections.abc import Callable
import ast
from .utils import parse_ast, SymbolTable, Scope
from allo.ir.types import AlloType, Struct, Stream, Stateful, ConstExpr
from allo.memory import DTensor, Layout


class BlockScopeGuard:
    def __init__(self, scopes: list[Scope]):
        self.scopes: list[Scope] = scopes

    def __enter__(self):
        self.scopes.append(Scope())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()


class ASTProcessor(ast.NodeTransformer):
    def __init__(self, symbol_table: SymbolTable, global_symbols: dict):
        super().__init__()
        self.symbol_table: SymbolTable = symbol_table
        self.global_symbols: dict = global_symbols
        self.scopes: list[Scope] = []
        self.current_func: list[str] = []

    def block_scope_guard(self):
        return BlockScopeGuard(self.scopes)

    def put_var(self, name, val):
        assert (
            name not in self.symbol_table.functions
            and name not in self.symbol_table.variables
        )
        self.scopes[-1].vars[name] = val

    def put_const(self, name, const):
        assert (
            name not in self.symbol_table.functions
            and name not in self.symbol_table.variables
        )
        self.scopes[-1].consts[name] = const

    def get_symbol(self, name, allow_missing=False):
        """
        Get the value of a symbol from the current scope chain.

        Args:
            - name (str): The variable name to look up.
            - allow_missing (bool): If True, return None when the symbol
                does not exist. Otherwise, raise an error.
        """
        for scope in reversed(self.scopes):
            if name in scope.vars:
                return scope.vars[name]
            if name in scope.consts:
                return scope.consts[name]
        if allow_missing:
            return None
        raise ValueError(f"Variable {name} not defined in current scope.")

    def get_consts(self):
        consts = {}
        for scope in self.scopes:
            for k, v in scope.consts.items():
                consts[k] = v
        return consts

    def process(self, fn: Union[Callable, str], instantiate: list = None):
        """
        Process the input function.

        Args:
            fn: The function to process.
            instantiate: The arguments to instantiate the function. default to None.
        """
        ast_module: ast.Module = parse_ast(fn)
        if instantiate is not None:
            # if instantiate is not None, we need to use the args to instantiate the unique function
            assert len(ast_module.body) == 1 and isinstance(
                ast_module.body[0], ast.FunctionDef
            )
            return self.visit_FunctionDef(ast_module.body[0], instantiate)
        else:
            if len(ast_module.body) == 1:
                return self.visit(ast_module.body[0])
            for stmt in ast_module.body:
                self.visit(stmt)
            return ast_module

    def eval_constant(self, node, consts=None):
        """
        Evaluate the constant expression.

        Args:
            node: The node to evaluate.
            consts: The constants to use. default to None. Used to avoid reanalyzing the constants.
        """
        if consts is None:
            consts = self.get_consts()
            consts.update(self.global_symbols)
        return eval(compile(ast.Expression(node), "", "eval"), consts)

    def resolve_node(self, node: ast.AST):
        if isinstance(node, ast.Name):
            return self.global_symbols[node.id]  # limited to single-level symbol lookup
        if isinstance(node, ast.Call):
            ty_cls = self.global_symbols[node.func.id]
            consts = self.get_consts()
            consts.update(self.global_symbols)
            args = [self.eval_constant(arg, consts=consts) for arg in node.args]
            kwargs = {
                kw.arg: self.eval_constant(kw.value, consts=consts)
                for kw in node.keywords
            }
            return ty_cls(*args, **kwargs)

    def visit_type_annotation(self, annotation: ast.AST):
        """
        Visit the type annotation.

        Returns:
            dtype: data type.
            shape: The shape of the type.
            refinement: type refinement. Layout, Stateful, Memory, etc.
        """
        if isinstance(annotation, ast.Name):
            # e.g., A: int32
            return self.resolve_node(annotation), tuple(), None
        if isinstance(annotation, ast.Call):
            # e.g., A: Int(32)
            return self.resolve_node(annotation), tuple(), None
        if isinstance(annotation, ast.Subscript):
            # e.g., a: int32[32], a: Int(32)[32], pipe: Stream[Ty, 4][4]
            base_type, base_shape, _ = self.visit_type_annotation(annotation.value)
            assert len(base_shape) == 0
            if base_type is Stream:
                # e.g., Stream[Ty, 4]
                assert (
                    isinstance(annotation.slice, ast.Tuple)
                    and len(annotation.slice.elts) == 2
                ), "Only support `ele_type` and `depth` for now"
                ele_type, ele_shape, _ = self.visit_type_annotation(
                    annotation.slice.elts[0]
                )
                depth = self.eval_constant(annotation.slice.elts[1])
                return (
                    Stream(dtype=ele_type, shape=ele_shape, depth=depth),
                    tuple(),
                    None,
                )
            if base_type is ConstExpr:
                # e.g., a: ConstExpr[int32]
                ele_type, ele_shape, _ = self.visit_type_annotation(annotation.slice)
                assert len(ele_shape) == 0, "ConstExpr only supports scalar types"
                const_dtype = copy.deepcopy(base_type)
                const_dtype.constexpr = True
                return const_dtype, tuple(), None
            size = (
                annotation.slice.value
                if isinstance(annotation.slice, ast.Index)
                else annotation.slice
            )
            elts = size.elts if isinstance(size, ast.Tuple) else [size]
            return (
                base_type,
                tuple(self.eval_constant(x) for x in elts),
                Layout([Layout.Replicate] * len(elts)),  # default layout
            )
        if isinstance(annotation, ast.BinOp):
            # e.g., B: Int(32) @ Stateful = 0, a: int32[32] @ Memory(resource="URAM")
            assert isinstance(annotation.op, ast.MatMult)
            dtype, shape, spec = self.visit_type_annotation(annotation.left)
            if isinstance(annotation.right, ast.Name):
                # 1.    B: Int(32) @ Stateful = 0
                # 2.    mm = Memory(resource="URAM") # defined in 'global' scope
                #       a: int32[32] @ mm
                refinement_type = self.global_symbols[annotation.right.id]
            elif isinstance(annotation.right, ast.Call):
                # a: int32[32] @ Memory(resource="URAM")
                refinement_type = self.resolve_node(annotation.right)
            elif isinstance(annotation.right, ast.List):
                # a: int32[32] @ [S(0)]
                refinement_type = [self.resolve_node(v) for v in annotation.right.elts]
            else:
                raise NotImplementedError
            if refinement_type is Stateful:
                stateful_dtype = copy.deepcopy(dtype)
                stateful_dtype.stateful = True
                return stateful_dtype, shape, spec
            if isinstance(refinement_type, list):
                refinement_type = Layout(refinement_type)
            return dtype, shape, refinement_type
        raise NotImplementedError

    def visit_broadcast(
        self, node: ast.AST, dtype: AlloType, target_shape: list[int]
    ) -> ast.AST:
        """
        Broadcast an expression to a specific shape. Return the broadcasted expression if broadcast is needed, otherwise return the original expression.
        """
        if not hasattr(node, "dtype"):
            node.dtype = dtype
        assert hasattr(node, "shape")
        if node.shape == target_shape:
            return node
        raise NotImplementedError

    def visit_Name(self, node: ast.Name):
        var = self.get_symbol(node.id, allow_missing=True)
        if var is not None:
            node.dtype, node.shape = var.dtype, var.shape
            return node
        const_node = ast.Constant(self.eval_constant(node))
        const_node.shape = tuple()
        return const_node

    def visit_Constant(self, node: ast.Constant):
        # e.g., 1, 1.0, True, False
        node.const_value = self.eval_constant(node)
        node.shape = tuple()  # dtype unknown
        return node

    def visit_Tuple(self, node: ast.Tuple):
        # e.g., A[i, j] -> Indexing
        raise NotImplementedError

    def visit_Dict(self, node: ast.Dict):
        raise NotImplementedError

    def visit_Index(self, node: ast.Index):
        raise NotImplementedError

    def visit_Attribute(self, node: ast.Attribute):
        raise NotImplementedError

    def visit_Subscript(self, node: ast.Subscript):
        # e.g., A[i], A[i, j]
        # slice: A[0:10], A[::-1]
        raise NotImplementedError

    def visit_ExtSlice(self, node: ast.ExtSlice):
        raise NotImplementedError

    def visit_Slice(self, node: ast.Slice):
        # e.g., A[0:10], A[::-1]
        raise NotImplementedError

    def visit_UnaryOp(self, node: ast.UnaryOp):
        # e.g., -x, ~x, not x
        raise NotImplementedError

    def visit_BinOp(self, node: ast.BinOp):
        # e.g., x + y, x - y, x * y, x / y, x // y, x % y, x ** y
        #       x << y, x >> y, x | y, x ^ y, x & y
        raise NotImplementedError

    def visit_BoolOp(self, node: ast.BoolOp):
        # e.g., x and y, x or y
        raise NotImplementedError

    def visit_Compare(self, node: ast.Compare):
        # e.g., x < y, x <= y, x > y, x >= y, x == y, x != y
        raise NotImplementedError

    def visit_Assign(self, node: ast.Assign):
        # e.g., A[i] = 1
        #       v = 1
        #       i, j = 1, 2
        raise NotImplementedError

    def visit_AugAssign(self, node: ast.AugAssign):
        # e.g., A[i] += 1
        raise NotImplementedError

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # e.g., C: float32[32, 32] = 0.0
        #       B: int32 = 0
        #       acc: Int(4) @ Stateful = 0
        dtype, shape, spec = self.visit_type_annotation(node.annotation)
        assert isinstance(
            node.target, ast.Name
        ), "target of AnnAssign must be Name, other type not supported."
        target_ = self.get_symbol(node.target.id, allow_missing=True)
        if target_ is not None:
            assert (
                node.value is not None
            ), "Unsupported annotated assignment without a value."
            # assignment
            assert (
                target_.dtype == dtype and target_.shape == shape
            ), f"Invalid assignment to {node.target.id}, type mismatch."
            assert not getattr(
                target_.dtype, "constexpr", False
            ), "Cannot reassign constants."
        if getattr(dtype, "constexpr", False):
            val = self.eval_constant(node.value)
            self.put_const(node.target.id, val)
        else:
            node.value = self.visit_broadcast(self.visit(node.value), dtype, shape)
            if target_ is None:
                self.put_var(node.target.id, node.value)
        node.target.dtype = node.dtype = dtype
        node.target.shape = node.shape = shape
        node.target.spec = node.spec = spec
        node.annotation = ast.Subscript(
            value=ast.Name(id="__allo__", ctx=ast.Load()),
            slice=ast.Tuple(
                elts=[
                    ast.Name(id=str(dtype), ctx=ast.Load()),
                    ast.Tuple(elts=shape, ctx=ast.Load()),
                    ast.Name(id=str(spec), ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        )
        return node

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, {ast.Call, ast.Constant}):
            return self.visit(node.value)
        raise RuntimeError(f"Unsupported expression: {node.value}")

    def visit_For(self, node: ast.For):
        # e.g., for i in range(10):
        #       for i in range(0, 10, 2):
        #       for i in allo.grid(10, 10):
        #       for i, j in allo.grid(10, 10):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in Allo kernels")
        raise NotImplementedError

    def visit_While(self, node: ast.While):
        # e.g., while i < 10:
        if node.orelse:
            raise RuntimeError(
                "'else' clause for 'while' not supported in Allo kernels"
            )
        raise NotImplementedError

    def visit_If(self, node: ast.If):
        # e.g., if i < 10: ... else: ...
        raise NotImplementedError

    def visit_IfExp(self, node: ast.IfExp):
        # e.g., x if cond else y
        raise NotImplementedError

    def visit_Return(self, node: ast.Return):
        res = self.visit(node.value)
        func_node = self.symbol_table.functions[self.current_func[-1]]
        # TODO: support casting and broadcasting
        if hasattr(res, "dtype"):
            assert res.dtype == func_node.dtype
        else:
            res.dtype = func_node.dtype
        assert res.shape == func_node.shape
        return node

    def visit_With(self, node: ast.With):
        # e.g., with allo.meta_if(cond):
        raise NotImplementedError

    def visit_Call(self, node: ast.Call):
        raise NotImplementedError

    def visit_FunctionDef(self, node: ast.FunctionDef, instantiate: list = None):
        with self.block_scope_guard():
            # instantiate an instance from template
            if instantiate is not None:
                func_name = self.symbol_table.name_mangling(node.name, instantiate)
                assert hasattr(node, "type_params") and len(node.type_params) == len(
                    instantiate
                )
                for type_var, call_val in zip(node.type_params, instantiate):
                    name = type_var.name
                raise NotImplementedError
            else:
                func_name = node.name
            # arguments
            for arg in node.args.args:
                arg.dtype, arg.shape, arg.spec = self.visit_type_annotation(
                    arg.annotation
                )
                assert not getattr(
                    arg.dtype, "stateful", False
                ), f"Function parameter '{arg.arg}' cannot be Stateful."
                # TODO: Dtentor
                assert self.get_symbol(name=arg.arg, allow_missing=True) is None, (
                    f"Argument name '{arg.arg}' conflicts with an existing symbol. "
                    f"Please choose a different name to avoid the conflict."
                )
                self.put_var(name=arg.arg, val=arg)
            # return type
            if not (
                (isinstance(node.returns, ast.Constant) and node.returns.value is None)
                or node.returns is None
            ):
                if isinstance(node.returns, ast.Tuple):
                    # Multiple return values
                    node.returns.shape = []
                    node.returns.dtype = []
                    node.returns.spec = []
                    for elt in node.returns.elts:
                        elt.dtype, elt.shape, elt.spec = self.visit_type_annotation(elt)
                        node.returns.dtype.append(elt.dtype)
                        node.returns.shape.append(elt.shape)
                        node.returns.spec.append(elt.spec)
                else:
                    # Single return value
                    node.returns.dtype, node.returns.shape, node.returns.spec = (
                        self.visit_type_annotation(node.returns)
                    )
                node.dtype, node.shape = node.returns.dtype, node.returns.shape
            # function body
            self.symbol_table.functions[func_name] = node
            self.current_func.append(func_name)
            for stmt in node.body:
                self.visit(stmt)
            self.current_func.pop()
        return node

    # ----- invalid syntax -----
    def visit_Break(self, node: ast.Break):
        raise RuntimeError("Break statement is not supported")

    def visit_Continue(self, node: ast.Continue):
        raise RuntimeError("Continue statement is not supported")
