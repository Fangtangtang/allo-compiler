# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from allo._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    memref as memref_d,
    tensor as tensor_d,
    affine as affine_d,
    scf as scf_d,
    arith as arith_d,
    math as math_d,
    linalg as linalg_d,
)
from allo._mlir.ir import (
    Context,
    Module,
    Location,
    InsertionPoint,
    FunctionType,
    MemRefType,
    RankedTensorType,
    ShapedType,
    IntegerType,
    F32Type,
    UnitAttr,
    IntegerAttr,
    StringAttr,
    DictAttr,
    AffineExpr,
    AffineConstantExpr,
    AffineMap,
    AffineMapAttr,
    IntegerSet,
    IntegerSetAttr,
    FlatSymbolRefAttr,
    DenseElementsAttr,
    TypeAttr,
    ArrayAttr,
    Attribute,
    OpResultList,
    StridedLayoutAttr,
)
from allo.utils import register_dialect
from allo.ir.types import AlloType
from allo.ir.utils import MockArg, MockScalar, MockConstant, MockBuffer
from .utils import SymbolTable, BlockScopeGuard, Scope


class IRBuilder(ast.NodeVisitor):
    def __init__(self, symbol_table: SymbolTable):
        super().__init__()
        self.symbol_table: SymbolTable = symbol_table
        self.scopes: list[Scope] = []
        self.ctx: Context = Context()
        register_dialect(self.ctx)
        self.module: Module = None

        self.ip_stack = []  # module insert pointes

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

    def set_ip(self, ip):
        if not isinstance(ip, InsertionPoint):
            ip = InsertionPoint(ip)
        self.ip_stack.append(ip)

    def get_ip(self):
        return self.ip_stack[-1]

    def pop_ip(self):
        return self.ip_stack.pop()

    def put_var(self, name, val):
        self.scopes[-1].vars[name] = val

    def build(self, ast_module: ast.FunctionDef):
        with self.ctx, Location.unknown():
            self.module = Module.create()
            self.set_ip(self.module.body)
            self.visit(ast_module)
            self.pop_ip()
            return self.module

    def build_type(self, annotation: ast.Subscript):
        assert (
            isinstance(annotation.slice, ast.Tuple) and len(annotation.slice.elts) == 3
        )  # by construction
        dtype = annotation.slice.elts[0]
        shape = annotation.slice.elts[1]
        spec = annotation.slice.elts[2]
        assert isinstance(dtype, ast.Name) and isinstance(shape, ast.Tuple)
        dtype = self.symbol_table.types[dtype.id]
        shape = [int(size.value) for size in shape.elts]
        if len(shape) == 0:
            return dtype.build()
        return MemRefType.get(shape, dtype.build())

    def visit_Name(self, node: ast.Name):
        raise NotImplementedError

    def visit_Constant(self, node: ast.Constant):
        raise NotImplementedError

    def visit_Tuple(self, node: ast.Tuple):
        raise NotImplementedError

    def visit_Subscript(self, node: ast.Subscript):
        raise NotImplementedError

    def visit_Slice(self, node: ast.Slice):
        raise NotImplementedError

    def visit_BoolOp(self, node: ast.BoolOp):
        raise NotImplementedError

    def visit_AnnAssign(self, node: ast.AnnAssign):
        raise NotImplementedError

    def visit_For(self, node: ast.For):
        raise NotImplementedError

    def visit_While(self, node: ast.While):
        raise NotImplementedError

    def visit_If(self, node: ast.If):
        raise NotImplementedError

    def visit_IfExp(self, node: ast.IfExp):
        raise NotImplementedError

    def visit_Return(self, node: ast.Return):
        raise NotImplementedError

    def visit_With(self, node: ast.With):
        raise NotImplementedError

    def visit_Call(self, node: ast.Call):
        raise NotImplementedError

    def visit_FunctionDef(self, node: ast.FunctionDef):
        input_types = [self.build_type(arg.annotation) for arg in node.args.args]
        if node.returns is None:
            output_types = []
        else:
            rets = (
                node.returns.elts
                if isinstance(node.returns, ast.Tuple)
                else [node.returns]
            )
            output_types = [self.build_type(ret) for ret in rets]
        # Build function
        func_type = FunctionType.get(input_types, output_types)
        func_op = func_d.FuncOp(name=node.name, type=func_type, ip=self.get_ip())
        func_op.add_entry_block()
        with self.block_scope_guard():
            # function arguments
            for i, (ast_arg, arg) in enumerate(zip(node.args.args, func_op.arguments)):
                mock_arg = MockArg(arg, idx=i)
                self.put_var(name=ast_arg.arg, val=mock_arg)
            self.set_ip(func_op.entry_block)
            for stmt in node.body:
                self.visit(stmt)
            self.pop_ip()
