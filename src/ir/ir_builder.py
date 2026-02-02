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
from .utils import SymbolTable


class IRBuilder(ast.NodeVisitor):
    def __init__(self, symbol_table: SymbolTable):
        super().__init__()
        self.symbol_table: SymbolTable = symbol_table
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

    def set_ip(self, ip):
        if not isinstance(ip, InsertionPoint):
            ip = InsertionPoint(ip)
        self.ip_stack.append(ip)

    def get_ip(self):
        return self.ip_stack[-1]

    def pop_ip(self):
        return self.ip_stack.pop()

    def build(self, ast_module: ast.FunctionDef):
        with self.ctx, Location.unknown():
            self.module = Module.create()
        self.visit(ast_module)
        return self.module

    def visit_Name(self, node):
        raise NotImplementedError

    def visit_Constant(self, node):
        raise NotImplementedError

    def visit_Subscript(self, node):
        raise NotImplementedError

    def visit_Slice(self, node):
        raise NotImplementedError

    def visit_BoolOp(self, node):
        raise NotImplementedError

    def visit_AnnAssign(self, node):
        raise NotImplementedError

    def visit_For(self, node):
        raise NotImplementedError

    def visit_While(self, node):
        raise NotImplementedError

    def visit_If(self, node):
        raise NotImplementedError

    def visit_IfExp(self, node):
        raise NotImplementedError

    def visit_Return(self, node):
        raise NotImplementedError

    def visit_With(self, node):
        raise NotImplementedError

    def visit_Call(self, node):
        raise NotImplementedError

    def visit_FunctionDef(self, node):
        raise NotImplementedError
