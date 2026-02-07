# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Union
from collections import deque
from collections.abc import Callable
import ast
from .utils import parse_ast, SymbolTable, BlockScopeGuard, Scope
from .typing_rule import cpp_style_registry
from allo.ir.types import (
    AlloType,
    Float,
    Stream,
    Stateful,
    ConstExpr,
    Index,
    bool as allo_bool,
)
from allo.memory import Layout
from .builtin import BUILTIN_HANDLERS


class ASTProcessor(ast.NodeTransformer):
    def __init__(
        self,
        symbol_table: SymbolTable,
        global_symbols: dict,
        typing_rule: str = "default",
    ):
        super().__init__()
        self.symbol_table: SymbolTable = symbol_table
        self.global_symbols: dict = global_symbols
        self.typing_rule = typing_rule
        self.scopes: list[Scope] = []

        self.worklist: deque[ast.FunctionDef] = deque([])
        self.current_func: str = None

    def visit(self, node):
        """
        Visit a node.

        [NOTE]: avoid missing any case
        """
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        assert visitor is not None
        return visitor(node)

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
                consts[k] = v.value
        return consts

    def process(self, fn: Union[Callable, str], instantiate: list = None):
        """
        Process the input function.

        Args:
            fn: The function to process.
            instantiate: The arguments to instantiate the function. default to None.
        """
        module: ast.Module = parse_ast(fn)
        if instantiate is not None:
            # if instantiate is not None, we need to use the args to instantiate the unique function
            assert len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef)
            node, top_name = self.visit_function_signature(module.body[0], instantiate)
            self.worklist.append(top_name)
        else:
            if len(module.body) == 1:
                node, top_name = self.visit_function_signature(module.body[0])
                self.worklist.append(top_name)
            else:
                # FIXME: tentative
                top_name = None
                for stmt in module.body:
                    self.visit(stmt)
        while self.worklist:
            self.visit_function_body(
                self.symbol_table.functions[self.worklist.popleft()]
            )
        return module, top_name

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
        return eval(
            compile(ast.fix_missing_locations(ast.Expression(node)), "", "eval"), consts
        )

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
                const_dtype = copy.deepcopy(ele_type)
                const_dtype.constexpr = True
                return const_dtype, tuple(), None
            size = annotation.slice
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

    def get_ast_annotaiton(
        self, dtype: AlloType, shape: tuple[int], spec
    ) -> ast.Subscript:
        # TODO: may collect spec in the same way
        dtype_name = str(dtype)
        if dtype_name not in self.symbol_table.types:
            self.symbol_table.types[dtype_name] = dtype
        return ast.Subscript(
            value=ast.Name(id="__allo__", ctx=ast.Load()),
            slice=ast.Tuple(
                elts=[
                    ast.Name(id=dtype_name, ctx=ast.Load()),
                    ast.Tuple(
                        elts=[ast.Constant(value=d) for d in shape],
                        ctx=ast.Load(),
                    ),
                    ast.Name(id=str(spec), ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        )

    def visit_broadcast(
        self, node: ast.AST, dtype: AlloType, target_shape: tuple[int]
    ) -> ast.AST:
        """
        Broadcast an expression to a specific shape. Return the broadcasted expression if broadcast is needed, otherwise return the original expression.
        """
        shape = getattr(node, "shape", None)
        assert shape is not None and len(shape) <= len(target_shape)
        if shape == target_shape:
            return node
        # FIXME: tentative, using -1 as placeholder to distinguish from a dim with size is 1
        padded_shape = [-1] * (len(target_shape) - len(shape)) + list(shape)
        dims = []
        for idx, (s, t) in enumerate(zip(padded_shape, target_shape)):
            if s != t:
                if s != 1 and s != -1:
                    raise ValueError(f"shape mismatch: {shape} vs {target_shape}")
                dims.append(idx)
        # FIXME: currently use linalg.broadcast for lowering, can only 'insert' dim
        assert len(target_shape) - len(shape) == len(dims), "not a semantic constraint"
        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="__allo__", ctx=ast.Load()),
                attr="broadcast",
                ctx=ast.Load(),
            ),
            args=[
                node,  # original node
                ast.Tuple(
                    elts=[ast.Constant(value=d) for d in dims],
                    ctx=ast.Load(),
                ),  # dims
                self.get_ast_annotaiton(dtype, target_shape, None),  # target type
            ],
            keywords=[],
        )
        call_node.dtype, call_node.shape = dtype, target_shape
        return call_node

    def visit_cast(self, node: ast.AST, target_dtype: AlloType) -> ast.AST:
        if isinstance(node, ast.Constant):
            # constant should be explicitly 'typed', replace the node with builtin constant construction function call
            shape = node.shape
            if isinstance(target_dtype, Float):
                node.value = float(node.value)
            else:
                node.value = int(node.value)
            node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="__allo__", ctx=ast.Load()),
                    attr="constant",
                    ctx=ast.Load(),
                ),
                args=[
                    node,
                    self.get_ast_annotaiton(
                        target_dtype, shape, getattr(node, "spec", None)
                    ),
                ],
                keywords=[],
            )
            node.dtype, node.shape = target_dtype, shape
        if node.dtype == target_dtype:
            return node
        # infer specific handler using CastHandler (abstract class for all cast handlers)
        try:
            # [NOTE] the first two return value is not useful here, we keep them to make `infer`'s interface consistent
            _, _, handler = BUILTIN_HANDLERS["cast"].infer(node.dtype, target_dtype)
        except TypeError as e:
            raise TypeError(f"Cast inference failed: {e}")

        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="__allo__", ctx=ast.Load()),
                attr=handler,  # Dispatch to specific handler
                ctx=ast.Load(),
            ),
            args=[
                node,
                self.get_ast_annotaiton(
                    target_dtype, node.shape, getattr(node, "spec", None)
                ),
            ],
            keywords=[],
        )
        call_node.dtype = target_dtype
        call_node.shape = node.shape
        return call_node

    def visit_Name(self, node: ast.Name):
        var = self.get_symbol(node.id, allow_missing=True)
        if var is not None:
            if isinstance(var, ast.Constant):
                return var
            node.dtype, node.shape = var.dtype, var.shape
            return node
        const_node = ast.Constant(self.eval_constant(node))
        const_node.shape = tuple()
        return const_node

    def visit_Constant(self, node: ast.Constant):
        # e.g., 1, 1.0, True, False
        node.shape = tuple()  # dtype unknown
        return node

    def visit_Tuple(self, node: ast.Tuple):
        # e.g., return A, B
        raise NotImplementedError

    def visit_Dict(self, node: ast.Dict):
        raise NotImplementedError

    def visit_Attribute(self, node: ast.Attribute):
        raise NotImplementedError

    def visit_Subscript(self, node: ast.Subscript):
        # e.g., A[i], A[i, j]
        # slice: A[0:10], A[::-1]
        value = self.visit(node.value)
        if len(value.shape) > 0:
            # tensor subscript
            elts = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            new_elts = []
            assert len(elts) <= len(value.shape)
            shape = []
            for idx, elt in enumerate(elts):
                elt_ = self.visit(elt)
                if isinstance(elt_, ast.Slice):
                    if elt_.upper is None:
                        elt_.upper = ast.Constant(value.shape[idx])
                    assert isinstance(elt_.lower, ast.Constant)
                    assert isinstance(elt_.upper, ast.Constant)
                    assert isinstance(elt_.step, ast.Constant)
                    size = (elt_.upper.value - elt_.lower.value) // elt_.step.value
                    if size > 0:
                        shape.append(size)
                elif not isinstance(
                    elt_, ast.Constant
                ):  # let constant be a special case (FIXME: merge)
                    elt_ = self.visit_cast(elt_, Index())
                new_elts.append(elt_)
            shape.extend(value.shape[len(elts) :])
            node.dtype, node.shape = value.dtype, tuple(shape)
            if isinstance(node.slice, ast.Tuple):
                node.slice.elts = new_elts
            else:
                node.slice = new_elts[0]
            return node
        raise NotImplementedError

    def visit_Slice(self, node: ast.Slice):
        # e.g., A[0:10], A[::-1]
        if node.lower is not None:
            node.lower = self.visit(node.lower)
        else:
            node.lower = ast.Constant(value=0)
        if node.upper is not None:
            node.upper = self.visit(node.upper)
        if node.step is not None:
            node.step = self.visit(node.step)
        else:
            node.step = ast.Constant(value=1)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        # e.g., +x, -x, not x
        node.operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            # +x -> x
            return node.operand
        if isinstance(node.op, ast.USub):
            # -x -> 0 - x
            if isinstance(node.operand, ast.Constant):
                assert isinstance(node.operand.value, (int, float))
                node.operand.value = -node.operand.value
                return node.operand
            return self.visit(ast.BinOp(ast.Constant(value=0), ast.Sub(), node.operand))
        if isinstance(node.op, ast.Not):
            # not x
            if isinstance(node.operand, ast.Constant):
                assert isinstance(node.operand.value, bool)
                node.operand.value = not node.operand.value
                return node.operand
            return self.visit(
                ast.Compare(ast.Constant(value=False), [ast.Eq()], [node.operand])
            )
        raise TypeError(f"Unsupported unary operator: {type(node.op).__name__}")

    def resolve_broadcast_shape(self, shape_a, shape_b):
        """
        Compute the compatible shape specifically for broadcasting from shape_a and shape_b.

        See the broadcasting rules in NumPy
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        When operating on two arrays, NumPy compares their shapes element-wise.
        It starts with the trailing (i.e. rightmost) dimension and works its way left.
        Two dimensions are compatible when
        1. they are equal, or
        2. one of them is 1.
        """
        # Align shapes by prefixing with 1s
        ndim_a, ndim_b = len(shape_a), len(shape_b)
        ndim_res = max(ndim_a, ndim_b)
        aligned_a = (1,) * (ndim_res - ndim_a) + tuple(shape_a)
        aligned_b = (1,) * (ndim_res - ndim_b) + tuple(shape_b)

        res_shape = []
        for da, db in zip(aligned_a, aligned_b):
            if da == db:
                res_shape.append(da)
            elif da == 1:
                res_shape.append(db)
            elif db == 1:
                res_shape.append(da)
            else:
                raise ValueError(
                    f"Operands could not be broadcast together with shapes {shape_a} {shape_b}"
                )
        return tuple(res_shape)

    def visit_binary_op_operands(
        self, left: ast.expr, right: ast.expr, op: ast.operator
    ):
        arg1 = getattr(left, "dtype", getattr(left, "value", None))
        arg2 = getattr(right, "dtype", getattr(right, "value", None))
        try:
            result_type, l_type, r_type, *others = BUILTIN_HANDLERS[
                str(type(op).__name__)
            ].infer(arg1, arg2)
        except TypeError as e:
            raise TypeError(f"Type error in binary operation ({op}): {e}")
        left = self.visit_cast(left, l_type)
        right = self.visit_cast(right, r_type)
        # Broadcasting
        lhs_shape = getattr(left, "shape", tuple())
        rhs_shape = getattr(right, "shape", tuple())
        if lhs_shape != rhs_shape:
            try:
                result_shape = self.resolve_broadcast_shape(lhs_shape, rhs_shape)
            except ValueError as e:
                raise ValueError(f"Broadcasting error in binary operation {op}: {e}")
            left = self.visit_broadcast(left, left.dtype, result_shape)
            right = self.visit_broadcast(right, right.dtype, result_shape)
        else:
            result_shape = lhs_shape
        args = [left, right, self.get_ast_annotaiton(result_type, result_shape, None)]
        for extra in others:
            if isinstance(extra, str):
                args.append(ast.Name(id=extra, ctx=ast.Load()))
            else:
                raise NotImplementedError
        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="__allo__", ctx=ast.Load()),
                attr=str(type(op).__name__),
                ctx=ast.Load(),
            ),
            args=args,
            keywords=[],
        )
        call_node.dtype, call_node.shape = result_type, result_shape
        return call_node

    def make_assignment(self, target: ast.AST, value: ast.AST) -> ast.AnnAssign:
        target_dtype = getattr(target, "dtype", None)
        target_shape = getattr(target, "shape", None)
        if target_dtype is not None:
            value = self.visit_cast(value, target_dtype)
            value = self.visit_broadcast(value, target_dtype, target_shape)

        target.dtype, target.shape = value.dtype, value.shape
        annotation = self.get_ast_annotaiton(
            target.dtype, target.shape, getattr(target, "spec", None)
        )
        assign_node = ast.AnnAssign(
            target=target,
            annotation=annotation,
            value=value,
            simple=isinstance(target, ast.Name),
        )
        assign_node.dtype, assign_node.shape = value.dtype, value.shape
        return assign_node

    def visit_BinOp(self, node: ast.BinOp):
        # e.g., x + y, x - y, x * y, x / y, x // y, x % y,
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        # costant folding
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            new_node = ast.Constant(value=self.eval_constant(node))
            new_node.shape = tuple()
            return new_node
        return self.visit_binary_op_operands(node.left, node.right, node.op)

    def visit_BoolOp(self, node: ast.BoolOp):
        # e.g., x and y, x or y
        arg_dtypes = []
        new_value = []
        for value in node.values:
            val = self.visit(value)
            arg_dtype = getattr(val, "dtype", getattr(val, "value", None))
            arg_dtypes.append(arg_dtype)
            new_value.append(val)
        node.values = new_value
        try:
            # FIXME: perhaps define a handler for this infer as well?
            typing_result = cpp_style_registry[type(node.op)](*arg_dtypes)
            res_type = typing_result[0]
            for i, val in enumerate(node.values):
                val.dtype, val.shape = typing_result[i + 1], tuple()
        except TypeError as e:
            raise TypeError(f"Type error in bool operation ({node.op}): {e}")
        node.dtype, node.shape = res_type, tuple()
        return node

    def visit_Compare(self, node: ast.Compare):
        # e.g., x < y, x <= y, x > y, x >= y, x == y, x != y
        assert len(node.comparators) == 1, "Only support one comparator for now"
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        # costant folding
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            new_node = ast.Constant(value=self.eval_constant(node))
            new_node.shape = tuple()
            return new_node
        return self.visit_binary_op_operands(left, right, node.ops[0])

    def visit_Assign(self, node: ast.Assign):
        # e.g., A[i] = 1
        #       v = 1
        #       i, j = 1, 2
        assert len(node.targets) == 1, "chained assignment not supported"
        targets = (
            node.targets[0].elts
            if isinstance(node.targets[0], ast.Tuple)
            else [node.targets[0]]
        )
        if isinstance(node.value, ast.Call):
            raise NotImplementedError
        values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        assert len(targets) == len(values)
        # FIXME: this has the potential issue of serializing simultaneous assignment
        node_list = []
        for target, value in zip(targets, values):
            rhs = self.visit(value)
            if isinstance(target, ast.Name):
                target_ = self.get_symbol(target.id, allow_missing=True)
                if target_ is None:
                    assert getattr(rhs, "dtype", None) is not None
                    assert getattr(rhs, "shape", None) is not None
                    self.put_var(name=target.id, val=target)
                else:
                    assert not getattr(
                        target_.dtype, "constexpr", False
                    ), "Cannot reassign constants."
                    assert not getattr(
                        target_, "immutable", False
                    ), "Cannot reassign scalar arguments"
                    target.dtype, target.shape = target_.dtype, target_.shape
            else:
                # e.g., A[i] = 1
                self.visit(target)
            node_list.append(self.make_assignment(target, rhs))
        # replace with a list of AnnAssign for normalization
        return node_list

    def visit_AugAssign(self, node: ast.AugAssign):
        # e.g., A[i] += 1
        rhs = self.visit(node.value)
        lhs = self.visit(node.target)
        assert not getattr(lhs.dtype, "constexpr", False), "Cannot reassign constants."
        assert not getattr(lhs, "immutable", False), "Cannot reassign scalar arguments"
        left = copy.deepcopy(lhs)
        for n in ast.walk(left):
            if isinstance(n, (ast.Name, ast.Attribute, ast.Subscript)):
                n.ctx = ast.Load()
        value = self.visit_binary_op_operands(left, rhs, node.op)
        return self.make_assignment(lhs, value)

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
            node.value = ast.Constant(self.eval_constant(node.value))
            self.put_const(node.target.id, node.value)
            node.value.dtype, node.value.shape = dtype, shape
            # FIXME: can we delete const expr here?
            return None
        else:
            if node.value is not None:
                value = self.visit(node.value)
                value = self.visit_cast(value, dtype)
                node.value = self.visit_broadcast(value, dtype, shape)
            self.put_var(node.target.id, node.target)
        node.target.dtype = node.dtype = dtype
        node.target.shape = node.shape = shape
        node.target.spec = node.spec = spec
        node.annotation = self.get_ast_annotaiton(dtype, shape, spec)
        return node

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            node.value = self.visit(node.value)
            return node
        if isinstance(node.value, ast.Constant):
            # comments
            return None
        raise RuntimeError(f"Unsupported expression: {node.value}")

    def visit_For(self, node: ast.For):
        # e.g., for i in range(10):
        #       for i in range(0, 10, 2):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in Allo kernels")
        if isinstance(node.iter, ast.Call):
            with self.block_scope_guard():
                iter_ = self.get_symbol(node.target.id, allow_missing=True)
                assert (
                    iter_ is None
                ), "Please choose a different name for the loop iterator."
                node.target.shape = tuple()
                node.target.dtype = Index()
                self.put_var(node.target.id, node.target)
                ivs = node.iter.args
                ivs_ = []
                for iv in ivs:
                    iv_ = self.visit(iv)
                    if not isinstance(
                        iv_, ast.Constant
                    ):  # let constant be a special case (FIXME: merge)
                        iv_ = self.visit_cast(iv_, Index())
                    ivs_.append(iv_)
                if len(ivs_) == 1:
                    ivs_.insert(0, ast.Constant(value=0))
                if len(ivs_) == 2:
                    ivs_.append(ast.Constant(value=1))
                node.iter.args = ivs_
                new_body = []
                for stmt in node.body:
                    res = self.visit(stmt)
                    if isinstance(res, list):
                        new_body.extend(res)
                    elif res is not None:
                        new_body.append(res)
                node.body = new_body
            return node
        raise RuntimeError("Unsupported for loop")

    def visit_While(self, node: ast.While):
        # e.g., while i < 10:
        if node.orelse:
            raise RuntimeError(
                "'else' clause for 'while' not supported in Allo kernels"
            )
        node.test = self.visit_cast(self.visit(node.test), allo_bool)
        assert len(node.test.shape) == 0, "while condition should be a scalar."
        with self.block_scope_guard():
            new_body = []
            for stmt in node.body:
                res = self.visit(stmt)
                if isinstance(res, list):
                    new_body.extend(res)
                elif res is not None:
                    new_body.append(res)
            node.body = new_body
        return node

    def visit_If(self, node: ast.If):
        # e.g., if i < 10: ... else: ...
        node.test = self.visit_cast(self.visit(node.test), allo_bool)
        assert len(node.test.shape) == 0, "if condition should be a scalar."
        with self.block_scope_guard():
            new_body = []
            for stmt in node.body:
                res = self.visit(stmt)
                if isinstance(res, list):
                    new_body.extend(res)
                elif res is not None:
                    new_body.append(res)
            node.body = new_body
        if len(node.orelse) > 0:
            with self.block_scope_guard():
                new_body = []
                for stmt in node.orelse:
                    res = self.visit(stmt)
                    if isinstance(res, list):
                        new_body.extend(res)
                    elif res is not None:
                        new_body.append(res)
                node.orelse = new_body
        return node

    def visit_IfExp(self, node: ast.IfExp):
        # e.g., x if cond else y
        raise NotImplementedError

    def visit_Return(self, node: ast.Return):
        # TODO: return a tuple (multiple return value)
        node.value = self.visit(node.value)
        func_node = self.symbol_table.functions[self.current_func]
        node.value = self.visit_cast(node.value, func_node.dtype)
        node.value = self.visit_broadcast(node.value, func_node.dtype, func_node.shape)
        return node

    def visit_Pass(self, node: ast.Pass):
        return node

    def visit_With(self, node: ast.With):
        # e.g., with allo.meta_if(cond):
        raise NotImplementedError

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            # FIXME: tentative!!!
            func = self.resolve_node(node.func)
            if hasattr(func, "__allo_handler__"):
                name = func.__allo_handler__
                # infer type
                arg_types, new_args = [], []
                for arg in node.args:
                    arg_ = self.visit(arg)
                    # FIXME: pass shape and spec
                    arg_types.append(getattr(arg_, "dtype", None))
                    new_args.append(arg_)
                # FIXME: assuming no kwargs for now
                try:
                    result_type, *other_types = BUILTIN_HANDLERS[name].infer(*arg_types)
                except NotImplementedError:
                    raise RuntimeError(f"Custom handler {name} must implement `infer`")

                # FIXME: should support casting and broadcasting
                call_node = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="__allo__", ctx=ast.Load()),
                        attr=name,
                        ctx=ast.Load(),
                    ),
                    args=new_args,
                    keywords=[],
                )
                # FIXME: result dtype, shape?
                return call_node
            else:  # call another source kernel
                module: ast.Module = parse_ast(func)
                assert len(module.body) == 1
                callee, callee_name = self.visit_function_signature(module.body[0])
                self.worklist.append(callee_name)
                node.func.id = callee_name
                # arguments TODO: support kwargs and others?
                assert len(node.args) == len(
                    callee.args.args
                ), f"Invalid call to {callee_name}, argument number mismatch."
                new_args = []
                for arg, callee_arg in zip(node.args, callee.args.args):
                    # TODO: spec?
                    arg = self.visit_cast(self.visit(arg), callee_arg.dtype)
                    arg = self.visit_broadcast(arg, arg.dtype, callee_arg.shape)
                    new_args.append(arg)
                node.args = new_args
                # return value
                if hasattr(callee, "dtype") and hasattr(callee, "shape"):
                    node.dtype, node.shape = callee.dtype, callee.shape
                return node
        # TODO
        return node

    def visit_function_signature(self, node: ast.FunctionDef, instantiate: list = None):
        # instantiate an instance from template
        if instantiate is not None:  # TODO: shall we copy node?
            func_name = self.symbol_table.name_mangling(node.name, instantiate)
            assert hasattr(node, "type_params") and len(node.type_params) == len(
                instantiate
            )
            for type_var, call_val in zip(node.type_params, instantiate):
                name = type_var.name
            raise NotImplementedError
        else:
            func_name = node.name
        if func_name in self.symbol_table.functions:  # function instance visited
            return self.symbol_table.functions[func_name], func_name
        self.symbol_table.functions[func_name] = node  # record function
        node.name = func_name
        # arguments
        for arg in node.args.args:
            arg.dtype, arg.shape, arg.spec = self.visit_type_annotation(arg.annotation)
            if len(arg.shape) == 0:
                arg.immutable = True  # [NOTE] scalar argument is defined as immutable, we don't allocate buffer for them
            arg.annotation = self.get_ast_annotaiton(arg.dtype, arg.shape, arg.spec)
            assert not getattr(
                arg.dtype, "stateful", False
            ), f"Function parameter '{arg.arg}' cannot be Stateful."
            # FIXME: this assumes functions are under global scope
            assert arg.arg not in self.global_symbols, (
                f"Argument name '{arg.arg}' conflicts with an existing symbol. "
                "Please choose a different name to avoid the conflict."
            )
        # return type
        if node.returns is not None:
            if isinstance(node.returns, ast.Tuple):
                # Multiple return values
                node.returns.shape = []
                node.returns.dtype = []
                node.returns.spec = []
                new_elts = []
                for elt in node.returns.elts:
                    elt.dtype, elt.shape, elt.spec = self.visit_type_annotation(elt)
                    node.returns.dtype.append(elt.dtype)
                    node.returns.shape.append(elt.shape)
                    node.returns.spec.append(elt.spec)
                    new_elts.append(
                        self.get_ast_annotaiton(elt.dtype, elt.shape, elt.spec)
                    )
                node.returns.elts = new_elts
            else:
                # Single return value
                dtype, shape, spec = self.visit_type_annotation(node.returns)
                node.returns = self.get_ast_annotaiton(dtype, shape, spec)
                node.returns.dtype = dtype
                node.returns.shape = shape
                node.returns.spec = spec
            node.dtype, node.shape = node.returns.dtype, node.returns.shape
        return node, func_name

    def visit_function_body(self, node: ast.FunctionDef):
        with self.block_scope_guard():
            # arguments
            for arg in node.args.args:
                self.put_var(name=arg.arg, val=arg)
            # function body
            self.current_func = node.name
            new_body = []
            for stmt in node.body:
                res = self.visit(stmt)
                if isinstance(res, list):
                    new_body.extend(res)
                elif res is not None:
                    new_body.append(res)
            if node.returns is None and not isinstance(new_body[-1], ast.Return):
                new_body.append(ast.Return())
            node.body = new_body
            self.current_func = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef, instantiate: list = None):
        # TODO: when will we use this? df.region?
        node, _ = self.visit_function_signature(node, instantiate=instantiate)
        return self.visit_function_body(node)

    # ----- invalid syntax -----
    def visit_Break(self, node: ast.Break):
        raise RuntimeError("Break statement is not supported")

    def visit_Continue(self, node: ast.Continue):
        raise RuntimeError("Continue statement is not supported")
