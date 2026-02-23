# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import islpy as isl
from collections import defaultdict
from allo._mlir.ir import (
    StringAttr,
    FlatSymbolRefAttr,
    InsertionPoint,
    Location,
    AffineMap,
    AffineMapAttr,
    AffineConstantExpr,
)
from allo._mlir.dialects import (
    allo as allo_d,
    func as func_d,
)


def replace_stream_arrays(module):
    # Find all global stream arrays (len(shape) > 0)
    stream_arrays = {}
    for op in module.body.operations:
        if isinstance(op, allo_d.StreamGlobalOp) and len(op.shape) > 0:
            stream_arrays[op.sym_name.value] = op

    new_streams = {}

    def replace_recursive(operations, work_name):
        for op in operations:
            if isinstance(op, (allo_d.GlobalStreamGetOp, allo_d.GlobalStreamPutOp)):
                with op.context, Location.unknown():
                    stream_sym = op.attributes["global"].value
                    allo_d.simplify_stream_affine_map(op)
                    aff_map = AffineMapAttr(op.map).value
                    assert aff_map.n_inputs == 0, "fail to resolve for now"
                    indices = [AffineConstantExpr(exp).value for exp in aff_map.results]
                    assert stream_sym in stream_arrays
                    is_put = isinstance(op, allo_d.GlobalStreamPutOp)

                    new_name = f"{stream_sym}_{"_".join(map(str, indices))}"

                    # Create new stream Op if not exists
                    if new_name not in new_streams:
                        stream = stream_arrays[stream_sym]
                        new_op = allo_d.stream_global(
                            new_name, stream.element_type, [], ip=InsertionPoint(stream)
                        )
                        new_streams[new_name] = new_op

                    if is_put:
                        new_streams[new_name].attributes["source"] = StringAttr.get(
                            work_name
                        )
                    else:
                        new_streams[new_name].attributes["dest"] = StringAttr.get(
                            work_name
                        )

                    if is_put:
                        allo_d.put_stream_global(
                            FlatSymbolRefAttr.get(new_name),
                            [],
                            op.operands[-1],
                            AffineMapAttr.get(
                                AffineMap.get(dim_count=0, symbol_count=0, exprs=[])
                            ),
                            ip=InsertionPoint(op),
                        )
                    else:
                        new_get = allo_d.GlobalStreamGetOp(
                            op.result.type,
                            FlatSymbolRefAttr.get(new_name),
                            [],
                            AffineMapAttr.get(
                                AffineMap.get(dim_count=0, symbol_count=0, exprs=[])
                            ),
                            ip=InsertionPoint(op),
                        )
                        op.result.replace_all_uses_with(new_get.result)

                    op.operation.erase()
            else:
                for region in op.regions:
                    for block in region.blocks:
                        replace_recursive(block.operations, work_name)

    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp):
            name = op.sym_name.value
            for block in op.body:
                replace_recursive(block.operations, name)

    for op in stream_arrays.values():
        op.operation.erase()
