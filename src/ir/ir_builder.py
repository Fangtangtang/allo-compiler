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
    OpView,
    Value,
    BlockArgument,
    FunctionType,
    MemRefType,
    IndexType,
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
from allo.ir.utils import MockArg, MockScalar, MockBuffer
from .utils import SymbolTable, BlockScopeGuard, Scope, MockCallResultTuple
from .builtin import BUILTIN_HANDLERS


class IRBuilder(ast.NodeVisitor):
    def __init__(self, symbol_table: SymbolTable):
        super().__init__()
        self.symbol_table: SymbolTable = symbol_table
        self.scopes: list[Scope] = []
        self.ctx: Context = Context()
        register_dialect(self.ctx)
        self.module: Module = None

        self.current_func: func_d.FuncOp = None  # the function under construction
        self.ip_stack = []  # module insert pointes
        self.handler_cache = {}

    def get_builtin_handler(self, name):
        if name not in self.handler_cache:
            if name not in BUILTIN_HANDLERS:
                return None
            self.handler_cache[name] = BUILTIN_HANDLERS[name](self)
        return self.handler_cache[name]

    def visit(self, node):
        """
        Visit a node.

        [NOTE]: avoid missing any case
        """
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        assert visitor is not None, f"{method} not found"
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

    def get_symbol(self, name, allow_missing=False):
        for scope in reversed(self.scopes):
            if name in scope.vars:
                return scope.vars[name]
            if name in scope.consts:
                return scope.consts[name]
        if allow_missing:
            return None
        raise RuntimeError("unreachable")

    def get_op_result(self, val):
        if isinstance(val, OpView):
            if isinstance(val.result, OpResultList):
                assert len(val.result) == 1
                return val.result[0]
            return val.result
        if isinstance(val, MockArg):
            return val.result
        if isinstance(val, MockCallResultTuple):
            return val
        assert isinstance(val, Value), f"Fail to resolve op result: {val}"
        return val

    def build(self, ast_module: ast.FunctionDef):
        with self.ctx, Location.unknown():
            self.module = Module.create()
            self.set_ip(self.module.body)
            for func_node in self.symbol_table.functions.values():
                self.visit(func_node)
            self.pop_ip()
            return self.module

    def build_type(self, annotation: ast.Subscript, force_memref: bool = False):
        """
        build type from annotation

        Args:
            annotation
            force_memref: if True, return memref type

        Returns:
            type, is_unsigned # FIXME: find a better way to handle unsigned
        """
        assert (
            isinstance(annotation.slice, ast.Tuple) and len(annotation.slice.elts) == 3
        )  # by construction
        dtype = annotation.slice.elts[0]
        shape = annotation.slice.elts[1]
        spec = annotation.slice.elts[2]
        assert isinstance(dtype, ast.Name) and isinstance(shape, ast.Tuple)
        allo_type = self.symbol_table.types[dtype.id]
        shape = [int(size.value) for size in shape.elts]
        if len(shape) == 0 and not force_memref:
            return allo_type.build(), dtype.id.startswith("u")
        return MemRefType.get(shape, allo_type.build()), dtype.id.startswith("u")

    def build_buffer(self, memref_type: MemRefType, is_unsigned: bool):
        buffer = memref_d.AllocOp(memref_type, [], [], ip=self.get_ip())
        if is_unsigned:
            buffer.attributes["unsigned"] = UnitAttr.get()
        return buffer

    def visit_Name(self, node: ast.Name):
        var = self.get_symbol(node.id)
        if isinstance(node.ctx, ast.Load):
            var = self.get_op_result(var)
            if (
                isinstance(getattr(var, "type", None), MemRefType)
                and len(var.type.shape) == 0
            ):
                # load scalar from memref
                affine_map = AffineMap.get_identity(0)
                affine_attr = AffineMapAttr.get(affine_map)
                var = affine_d.AffineLoadOp(
                    var.type.element_type, var, [], affine_attr, ip=self.get_ip()
                )
            return var
        raise NotImplementedError

    def visit_Constant(self, node: ast.Constant):
        if type(node.value) is int:
            return arith_d.ConstantOp(
                arith_d.IndexType.get(), node.value, ip=self.get_ip()
            )
        if type(node.value) is bool:
            return arith_d.ConstantOp(
                arith_d.IntegerType.get_signless(1), node.value, ip=self.get_ip()
            )
        raise NotImplementedError

    def get_affine_expr(self, node: ast.expr, ivs: list):
        """
        Parse an expression into an affine expression.

        [NOTE]: not suppose to build operations in the function, useless you think having some extra unused values are acceptable.
        """
        if isinstance(node, ast.Constant):
            return AffineConstantExpr.get(node.value)
        if isinstance(node, ast.Name):
            var = self.get_symbol(node.id)
            if isinstance(var, MockArg) and var.is_affine:
                ivs.append(var.result)
                return AffineExpr.get_dim(len(ivs) - 1)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                # builtin
                if node.func.value.id == "__allo__":
                    handler = self.get_builtin_handler(node.func.attr)
                    if handler:
                        return handler.get_affine_expr(node, ivs)
        # TODO: other cases
        return None

    def get_affine_attr(self, node: ast.expr):
        ivs = []
        expr = self.get_affine_expr(node, ivs)
        if expr is None:
            return None, None
        return AffineMap.get(dim_count=len(ivs), symbol_count=0, exprs=[expr]), ivs

    def visit_Subscript(self, node: ast.Subscript, val=None):
        base = self.get_op_result(self.visit(node.value))
        if isinstance(base, MockCallResultTuple):
            # [NOTE] special case handling for function call with multiple return
            return base.results[node.slice.value]
        shape: list[int] = base.type.shape  # tensor shape
        layout = base.type.layout
        elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        offsets, sizes, strides = [], [], []
        indices, ivs = [], []
        # try to parse elts to affine expressions (https://mlir.llvm.org/docs/Dialects/Affine/#affine-expressions)
        use_affine = True
        for elt in elts:
            aff = self.get_affine_expr(elt, ivs)
            if aff is not None:
                indices.append(aff)
                if isinstance(elt, ast.Constant):  # constant value
                    offsets.append(int(elt.value))
                else:
                    offsets.append(ShapedType.get_dynamic_stride_or_offset())
                sizes.append(1)
                strides.append(1)
            elif isinstance(elt, ast.Slice):  # getting (static) slice
                lower, upper, step = elt.lower.value, elt.upper.value, elt.step.value
                offsets.append(lower)
                sizes.append((upper - lower) // step)
                strides.append(step)
            else:
                use_affine = False
                # placeholder, so we can use len(indices) to check if is element access
                indices.append(None)
                offsets.append(ShapedType.get_dynamic_stride_or_offset())
                sizes.append(1)
                strides.append(1)
        if len(indices) == len(shape):  # access element
            if use_affine:  # affine operations
                affine_map = AffineMap.get(
                    dim_count=len(ivs), symbol_count=0, exprs=indices
                )
                affine_attr = AffineMapAttr.get(affine_map)
                if isinstance(node.ctx, ast.Load):
                    op = affine_d.AffineLoadOp(
                        base.type.element_type,
                        base,
                        ivs,
                        affine_attr,
                        ip=self.get_ip(),
                    )
                    return op
                else:  # ast.Store
                    op = affine_d.AffineStoreOp(
                        val,
                        base,
                        ivs,
                        affine_attr,
                        ip=self.get_ip(),
                    )
                    return None
            else:  # memref operaitons
                indices = [self.get_op_result(self.visit(elt)) for elt in elts]
                if isinstance(node.ctx, ast.Load):
                    op = memref_d.LoadOp(base, indices, ip=self.get_ip())
                    return op
                else:  # ast.Store
                    op = memref_d.StoreOp(val, base, indices, ip=self.get_ip())
                    return None
        else:  # access slice
            # TODO: support hybrid slice
            dynamic_offset = []
            for elt, offset_ in zip(elts, offsets):
                if offset_ < 0:
                    dynamic_offset.append(self.get_op_result(self.visit(elt)))
            sizes.extend(shape[len(offsets) :])
            strides.extend([1] * (len(shape) - len(offsets)))
            offsets.extend([0] * (len(shape) - len(offsets)))
            if isinstance(layout, StridedLayoutAttr):
                orig_offset = layout.offset
                orig_strides = layout.strides
            elif isinstance(layout, AffineMapAttr):
                orig_offset = 0
                orig_strides = [1]
                for i in reversed(shape[1:]):
                    orig_strides.insert(0, orig_strides[0] * i)
            else:
                raise RuntimeError(f"Unsupported layout type {type(layout)}")
            result_sizes = []
            stride_attr = []
            for idx_, size in enumerate(sizes):
                if size > 1:
                    result_sizes.append(size)
                    stride_attr.append(strides[idx_] * orig_strides[idx_])
            if len(dynamic_offset) > 0 or orig_offset < 0:
                offset_attr = ShapedType.get_dynamic_stride_or_offset()
            else:
                offset_attr = orig_offset + sum(
                    o * s for o, s in zip(offsets, orig_strides)
                )
            result = MemRefType.get(
                shape=result_sizes,
                element_type=base.type.element_type,
                # relative to the base memref
                layout=StridedLayoutAttr.get(offset=offset_attr, strides=stride_attr),
            )
            subview = memref_d.SubViewOp(
                source=base,
                result=result,
                static_offsets=offsets,
                static_sizes=sizes,
                static_strides=strides,
                offsets=dynamic_offset,
                sizes=[],
                strides=[],
                ip=self.get_ip(),
            )
            if isinstance(node.ctx, ast.Load):
                return subview
            else:
                return memref_d.CopyOp(val, subview.result, ip=self.get_ip())

    def visit_BoolOp(self, node: ast.BoolOp):
        opcls = {
            ast.And: arith_d.AndIOp,
            ast.Or: arith_d.OrIOp,
        }.get(type(node.op))
        result = opcls(
            self.get_op_result(self.visit(node.values[0])),
            self.get_op_result(self.visit(node.values[1])),
            ip=self.get_ip(),
        )
        for i in range(2, len(node.values)):
            result = opcls(
                result.result,
                self.get_op_result(self.visit(node.values[i])),
                ip=self.get_ip(),
            )
        return result

    def visit_Assign(self, node: ast.Assign):
        # [NOTE]: only used for special case (call a function with multiple returns)
        assert len(node.targets) == 1 and isinstance(node.value, ast.Call)
        call_op = self.visit(node.value)
        assert self.get_symbol(name=node.targets[0].id, allow_missing=True) is None
        self.put_var(node.targets[0].id, val=MockCallResultTuple(call_op.results))

    def visit_AnnAssign(self, node: ast.AnnAssign):
        value = (
            None if node.value is None else self.get_op_result(self.visit(node.value))
        )
        if isinstance(node.target, ast.Name):
            target = self.get_symbol(name=node.target.id, allow_missing=True)
            if target is None:
                # declare new variable
                alloc_op = self.build_buffer(
                    *self.build_type(node.annotation, force_memref=True)
                )
                alloc_op.attributes["name"] = StringAttr.get(node.target.id)
                self.put_var(node.target.id, val=alloc_op)
                target = alloc_op
        elif isinstance(node.target, ast.Subscript):
            self.visit_Subscript(node.target, val=value)
            return
        else:
            # FIXME: unreachable?
            target = self.visit(node.target)
        if value is None:
            return
        target = self.get_op_result(target)
        if isinstance(value.type, MemRefType):
            # tensor
            memref_d.CopyOp(value, target, ip=self.get_ip())
        else:
            # scalar
            affine_map = AffineMap.get(dim_count=0, symbol_count=0, exprs=[])
            affine_d.AffineStoreOp(
                value, target, [], AffineMapAttr.get(affine_map), ip=self.get_ip()
            )

    def visit_Expr(self, node: ast.Expr):
        return self.visit(node.value)

    def visit_For(self, node: ast.For):
        # TODO: should use higher-level affine loop if possible
        args = node.iter.args
        lb, lb_bound_ivs = self.get_affine_attr(args[0])
        ub, ub_bound_ivs = self.get_affine_attr(args[1])
        use_affine_loop = (
            lb is not None and ub is not None and isinstance(args[2], ast.Constant)
        )
        if use_affine_loop:
            step = int(args[2].value)
            for_op = affine_d.AffineForOp(
                lower_bound=lb,
                upper_bound=ub,
                step=step,
                iter_args=[],
                lower_bound_operands=lb_bound_ivs,
                upper_bound_operands=ub_bound_ivs,
                ip=self.get_ip(),
            )
            affine_d.AffineYieldOp([], ip=InsertionPoint(for_op.body))
        else:
            lb = self.get_op_result(self.visit(args[0]))
            rb = self.get_op_result(self.visit(args[1]))
            step = self.get_op_result(self.visit(args[2]))
            for_op = scf_d.ForOp(lb, rb, step, ip=self.get_ip())
            scf_d.YieldOp([], ip=InsertionPoint(for_op.body))

        with self.block_scope_guard():
            self.put_var(
                name=node.target.id,
                val=MockArg(for_op.induction_variable, use_affine_loop),
            )
            self.set_ip(for_op.body.operations[0])
            for stmt in node.body:
                self.visit(stmt)
            self.pop_ip()
        return

    def visit_While(self, node: ast.While):
        while_op = scf_d.WhileOp([], [], ip=self.get_ip())
        while_op.before.blocks.append(*[])
        while_op.after.blocks.append(*[])
        self.set_ip(while_op.before.blocks[0])
        cond = self.get_op_result(self.visit(node.test))
        scf_d.ConditionOp(cond, [], ip=self.get_ip())
        self.pop_ip()
        self.set_ip(while_op.after.blocks[0])
        with self.block_scope_guard():
            for stmt in node.body:
                self.visit(stmt)
            scf_d.YieldOp([], ip=self.get_ip())
        self.pop_ip()
        return while_op

    def visit_If(self, node: ast.If):
        # TODO: should use higher-level affine loop if possible
        if isinstance(node.test, ast.Constant):  # simple DCE
            if node.test.value:
                with self.block_scope_guard():
                    for stmt in node.body:
                        self.visit(stmt)
            else:
                with self.block_scope_guard():
                    for stmt in node.orelse:
                        self.visit(stmt)
            return
        if_op = scf_d.IfOp(
            self.get_op_result(self.visit(node.test)),
            ip=self.get_ip(),
            has_else=len(node.orelse),
        )
        self.set_ip(if_op.then_block)
        with self.block_scope_guard():
            for stmt in node.body:
                self.visit(stmt)
            scf_d.YieldOp([], ip=self.get_ip())
        self.pop_ip()
        if len(node.orelse) > 0:
            else_block = if_op.elseRegion.blocks[0]
            self.set_ip(else_block)
            with self.block_scope_guard():
                for stmt in node.orelse:
                    self.visit(stmt)
                scf_d.YieldOp([], ip=self.get_ip())
            self.pop_ip()

    def visit_IfExp(self, node: ast.IfExp):
        raise NotImplementedError

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            func_d.ReturnOp([], ip=self.get_ip())
            return
        values = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        rets = []
        for idx, value in enumerate(values):
            ret = self.get_op_result(self.visit(value))
            if (
                isinstance(ret.type, MemRefType)
                and ret.type != self.current_func.type.results[idx]
            ):  # mlir has strict type checking, `memref<32xi32, strided<[1]>>` != `memref<32xi32>`
                # FIXME: return unsigned?
                alloc_op = self.build_buffer(self.current_func.type.results[idx], False)
                memref_d.CopyOp(ret, alloc_op.result, ip=self.get_ip())
                ret = alloc_op.result
            rets.append(ret)
        func_d.ReturnOp(rets, ip=self.get_ip())

    def visit_Pass(self, node: ast.Pass):
        return None

    def visit_With(self, node: ast.With):
        raise NotImplementedError

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            if node.func.value.id == "__allo__":
                # handling for builtins
                name = node.func.attr
                assert name in BUILTIN_HANDLERS
                handler = self.get_builtin_handler(name)
                return handler.build(node)
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
            callee = self.symbol_table.functions[callee_name]
            rets = (
                callee.returns.elts
                if isinstance(callee.returns, ast.Tuple)
                else [callee.returns] if callee.returns is not None else []
            )
            call_op = func_d.CallOp(
                [self.build_type(ret)[0] for ret in rets],
                FlatSymbolRefAttr.get(callee_name),
                [self.get_op_result(self.visit(arg)) for arg in node.args],
                ip=self.get_ip(),
            )
            return call_op
        raise NotImplementedError

    def visit_FunctionDef(self, node: ast.FunctionDef):
        input_types, input_is_unsigned = [], []
        for arg in node.args.args:
            in_type, is_unsigned = self.build_type(arg.annotation)
            input_types.append(in_type)
            input_is_unsigned.append(is_unsigned)
        if node.returns is None:
            output_types = []
        else:
            rets = (
                node.returns.elts
                if isinstance(node.returns, ast.Tuple)
                else [node.returns]
            )
            output_types, output_is_unsigned = [], []
            for ret in rets:
                out_type, is_unsigned = self.build_type(ret)
                output_types.append(out_type)
                output_is_unsigned.append(is_unsigned)
        # Build function
        func_type = FunctionType.get(input_types, output_types)
        func_op = func_d.FuncOp(name=node.name, type=func_type, ip=self.get_ip())
        func_op.add_entry_block()
        self.current_func = func_op
        with self.block_scope_guard():
            # function arguments
            for i, (ast_arg, arg) in enumerate(zip(node.args.args, func_op.arguments)):
                mock_arg = MockArg(arg, idx=i)
                self.put_var(name=ast_arg.arg, val=mock_arg)
            self.set_ip(func_op.entry_block)
            for stmt in node.body:
                self.visit(stmt)
            self.pop_ip()
