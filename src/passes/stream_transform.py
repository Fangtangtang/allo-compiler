# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import defaultdict
from allo._mlir.ir import (
    StringAttr,
    FlatSymbolRefAttr,
    InsertionPoint,
    Location,
    AffineBinaryExpr,
    AffineAddExpr,
)
from allo._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    arith as arith_d,
)


def replace_stream_arrays(module):
    # Find all global stream arrays (len(shape) > 0)
    stream_arrays = {}
    for op in module.body.operations:
        if isinstance(op, allo_d.StreamGlobalOp) and len(op.shape) > 0:
            stream_arrays[op.sym_name.value] = op

    stream_map = defaultdict(dict)
    new_streams = {}
    ops_to_erase = []

    def replace_recursive(operations, work_name):
        for op in operations:
            if isinstance(op, (allo_d.GlobalStreamGetOp, allo_d.GlobalStreamPutOp)):
                stream_sym = op.attributes["global"].value
                allo_d.simplify_stream_affine_map(op)
                if stream_sym in stream_arrays:
                    is_put = isinstance(op, allo_d.GlobalStreamPutOp)
                    num_indices = len(op.operands)
                    if is_put:
                        num_indices -= 1  # Last one is value

                    current_indices = []
                    for i in range(num_indices):
                        idx_op = op.operands[i].owner
                        if isinstance(idx_op, arith_d.ConstantOp):
                            val = idx_op.value.value
                            current_indices.append(str(val))
                        else:
                            raise RuntimeError(
                                f"Stream array index not constant {idx_op}"
                            )

                    new_name = f"{stream_sym}_{'_'.join(current_indices)}"

                    # Create new stream Op if not exists
                    if new_name not in new_streams:
                        with InsertionPoint(module.body), Location.unknown():
                            orig_op = stream_arrays[stream_sym]
                            new_op = orig_op.clone()
                            new_op.sym_name = StringAttr.get(new_name)
                            if "dim" in new_op.attributes:
                                del new_op.attributes["dim"]
                            new_streams[new_name] = new_op

                    if is_put:
                        stream_map[new_name]["source"] = work_name
                    else:
                        stream_map[new_name]["dest"] = work_name

                    with InsertionPoint(op), Location.unknown():
                        if is_put:
                            val = op.operands[-1]
                            new_put = allo_d.PutStreamGlobalOp(
                                FlatSymbolRefAttr.get(new_name),
                                [],
                                val,
                                ip=InsertionPoint(op),
                            )
                            if "map" in op.attributes:
                                new_put.attributes["map"] = op.attributes["map"]
                        else:
                            new_get = allo_d.GetStreamGlobalOp(
                                op.result.type,
                                FlatSymbolRefAttr.get(new_name),
                                [],
                                ip=InsertionPoint(op),
                            )
                            if "map" in op.attributes:
                                new_get.attributes["map"] = op.attributes["map"]
                            op.result.replace_all_uses_with(new_get.result)

                    ops_to_erase.append(op)

    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp):
            name = op.sym_name.value
            for block in op.body:
                replace_recursive(block.operations, name)

    for op in ops_to_erase:
        op.operation.erase()

    for op in stream_arrays.values():
        op.operation.erase()

    with open("stream_map.json", "w") as f:
        json.dump(stream_map, f, indent=4)
