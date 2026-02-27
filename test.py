from allo._mlir.extras.dialects import builtin, func
from allo._mlir.dialects import (
    func as func_d,
    memref as memref_d,
    affine as affine_d,
    scf as scf_d,
    arith as arith_d,
)
from allo.utils import register_dialect

from allo._mlir.ir import (
    Context,
    Module,
    Location,
    InsertionPoint,
    SymbolRefAttr,
    FlatSymbolRefAttr,
)

ctx: Context = Context()
register_dialect(ctx)

with ctx, Location.unknown():
    mod = Module.create()  # top module
    with InsertionPoint(mod.body):
        sub_op = builtin.module("sub1")  # sub module
        with InsertionPoint(sub_op.bodyRegion.blocks[0]):
            f = func.function("foo", [], [])
            func_d.ReturnOp([], ip=InsertionPoint(f.entry_block))
        sub_op_2 = builtin.module("sub2")  # sub module
        with InsertionPoint(sub_op_2.bodyRegion.blocks[0]):
            f = func.function("core", [], [])
            # callee = SymbolRefAttr.get(["sub1","foo"])
            callee = FlatSymbolRefAttr.get("sub1")
            call_op = func_d.CallOp([], callee, [], ip=InsertionPoint(f.entry_block))
            func_d.ReturnOp([], ip=InsertionPoint(f.entry_block))
    print(mod)
