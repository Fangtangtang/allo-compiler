# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Union
from collections.abc import Callable
import ast
from .utils import parse_ast, SymbolTable, Scope
from .typing_rule import cpp_style_registry, cpp_style_bool
from allo.ir.types import AlloType, Struct, Stream, Stateful, ConstExpr, Index
from allo.memory import Layout


class BlockScopeGuard:
    def __init__(self, scopes: list[Scope]):
        self.scopes: list[Scope] = scopes

    def __enter__(self):
        self.scopes.append(Scope())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scopes.pop()


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
                consts[k] = v.value
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

    def visit_broadcast(
        self, node: ast.AST, dtype: AlloType, target_shape: tuple[int]
    ) -> ast.AST:
        """
        Broadcast an expression to a specific shape. Return the broadcasted expression if broadcast is needed, otherwise return the original expression.
        """
        if not hasattr(node, "dtype"):
            node.dtype = dtype
        shape = getattr(node, "shape", None)
        assert shape is not None and len(shape) <= len(target_shape)
        if shape == target_shape:
            return node
        padded_shape = [1] * (len(target_shape) - len(shape)) + list(shape)
        dims = []
        for idx, (s, t) in enumerate(zip(padded_shape, target_shape)):
            if s != t:
                if s != 1:
                    raise ValueError(f"shape mismatch: {shape} vs {target_shape}")
                dims.append(idx)
        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="__allo__", ctx=ast.Load()),
                attr="broadcast",
                ctx=ast.Load(),
            ),
            args=[
                node,  # original node
                ast.Tuple(
                    elts=[ast.Constant(value=d) for d in target_shape],
                    ctx=ast.Load(),
                ),  # target shape
                ast.Tuple(
                    elts=[ast.Constant(value=d) for d in dims],
                    ctx=ast.Load(),
                ),  # dims
            ],
            keywords=[],
        )
        call_node.dtype, call_node.shape = dtype, target_shape
        return call_node

    def visit_cast(self, node: ast.AST, target_dtype: AlloType) -> ast.AST:
        if not hasattr(node, "dtype"):
            node.dtype = target_dtype
        if node.dtype == target_dtype:
            return node
        # TODO: add checking here to make sure the cast is valid
        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="__allo__", ctx=ast.Load()),
                attr="cast",
                ctx=ast.Load(),
            ),
            args=[
                node,  # original node
                ast.Name(id=str(target_dtype), ctx=ast.Load()),
            ],
            keywords=[],
        )
        call_node.dtype = target_dtype
        if hasattr(node, "shape"):
            call_node.shape = node.shape
        return call_node

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
            shape.extend(value.shape[len(elts) :])
            node.dtype, node.shape = value.dtype, tuple(shape)
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
            raise NotImplementedError
        if isinstance(node.op, ast.Not):
            # not x
            raise NotImplementedError
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
        arg1 = getattr(left, "dtype", getattr(left, "const_value", None))
        arg2 = getattr(right, "dtype", getattr(right, "const_value", None))
        try:
            result_type, l_type, r_type = cpp_style_registry[type(op)](arg1, arg2)
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

        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="__allo__", ctx=ast.Load()),
                attr=str(type(op).__name__),
                ctx=ast.Load(),
            ),
            args=[left, right],
            keywords=[],
        )
        call_node.dtype, call_node.shape = result_type, result_shape
        return call_node

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
        # TODO: test this!
        print(ast.dump(node))
        for idx, value in enumerate(node.values):
            value = self.visit(value)
            assert (
                value.dtype == cpp_style_bool
                and len(getattr(value, "shape", tuple())) == 0
            )
            node.values[idx] = value
        node.dtype, node.shape = cpp_style_bool, tuple()
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
            target_dtype, target_shape = None, None
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
                    target_dtype, target_shape = target_.dtype, target_.shape
            else:
                # e.g., A[i] = 1
                lhs = self.visit(target)
                target_dtype, target_shape = lhs.dtype, lhs.shape
            if target_dtype is not None:
                if hasattr(rhs, "dtype"):
                    rhs = self.visit_cast(rhs, target_dtype)
                rhs = self.visit_broadcast(rhs, target_dtype, target_shape)
            target.dtype, target.shape = rhs.dtype, rhs.shape
            annotation = ast.Subscript(
                value=ast.Name(id="__allo__", ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[
                        ast.Name(id=str(target.dtype), ctx=ast.Load()),
                        ast.Tuple(
                            elts=[ast.Constant(value=d) for d in target.shape],
                            ctx=ast.Load(),
                        ),
                        ast.Name(id=str(getattr(target, "spec", None)), ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            )
            assign_node = ast.AnnAssign(
                target=target,
                annotation=annotation,
                value=rhs,
                simple=isinstance(target, ast.Name),
            )
            assign_node.dtype, assign_node.shape = rhs.dtype, rhs.shape
            node_list.append(assign_node)
        # replace with a list of AnnAssign for normalization
        return node_list

    def visit_AugAssign(self, node: ast.AugAssign):
        # e.g., A[i] += 1
        rhs = self.visit(node.value)
        lhs = self.visit(node.target)
        assert not getattr(lhs.dtype, "constexpr", False), "Cannot reassign constants."
        # TODO: replace with binary op + AnnAssign
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
            node.value = ast.Constant(self.eval_constant(node.value))
            self.put_const(node.target.id, node.value)
            node.value.dtype, node.value.shape = dtype, shape
        else:
            if node.value is not None:
                value = self.visit(node.value)
                if hasattr(value, "dtype"):
                    value = self.visit_cast(value, dtype)
                node.value = self.visit_broadcast(value, dtype, shape)
            self.put_var(node.target.id, node.target)
        node.target.dtype = node.dtype = dtype
        node.target.shape = node.shape = shape
        node.target.spec = node.spec = spec
        node.annotation = ast.Subscript(
            value=ast.Name(id="__allo__", ctx=ast.Load()),
            slice=ast.Tuple(
                elts=[
                    ast.Name(id=str(dtype), ctx=ast.Load()),
                    ast.Tuple(
                        elts=[ast.Constant(value=d) for d in shape], ctx=ast.Load()
                    ),
                    ast.Name(id=str(spec), ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            ),
            ctx=ast.Load(),
        )
        return node

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            return self.visit(node.value)
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
                ivs_ = [self.visit(iv) for iv in ivs]
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
        raise NotImplementedError

    def visit_If(self, node: ast.If):
        # e.g., if i < 10: ... else: ...
        node.test = self.visit(node.test)
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
            new_body = []
            for stmt in node.body:
                res = self.visit(stmt)
                if isinstance(res, list):
                    new_body.extend(res)
                elif res is not None:
                    new_body.append(res)
            node.body = new_body
            self.current_func.pop()
        return node

    # ----- invalid syntax -----
    def visit_Break(self, node: ast.Break):
        raise RuntimeError("Break statement is not supported")

    def visit_Continue(self, node: ast.Continue):
        raise RuntimeError("Continue statement is not supported")
