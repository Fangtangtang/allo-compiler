# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from .memory import DTensor
from allo._mlir.ir import (
    Module,
    Location,
    InsertionPoint,
    BlockArgument,
    Operation,
    FlatSymbolRefAttr,
)
from allo._mlir.extras.dialects import allo as allo
from allo._mlir.dialects import (
    allo as allo_d,
    func as func_d,
    memref as memref_d,
)
from allo.memory import Layout


def collect_symbol_refs_in_function(func_op: Operation):
    """
    Get symbol refs used in `func_op`. Symbols includes: global constant, streams, other 'function's
    """
    symbols = defaultdict(list)

    def collect_recursive(operations):
        for op in operations:
            for attr in op.attributes.values():
                if isinstance(attr, FlatSymbolRefAttr):
                    symbols[attr.value].append(op)
            for region in op.regions:
                for block in region.blocks:
                    collect_recursive(block.operations)

    for block in func_op.body:
        collect_recursive(block.operations)
    return symbols


def parse_spmw_module(module, top_name):
    with module.context, Location.unknown():
        mod = Module.create()

    symbol_map = {}
    symbol_uses_in_functions: dict[str, set] = {}
    for op in module.body.operations:
        if "sym_name" in op.attributes:
            name = op.attributes["sym_name"].value
            symbol_map[name] = op
        if isinstance(op, func_d.FuncOp):
            symbol_uses_in_functions[name] = collect_symbol_refs_in_function(op)
        else:
            with InsertionPoint(mod.body), Location.unknown():
                op.clone()

    top_module = symbol_map[top_name]
    work_grids = {}
    dtensors = defaultdict(list)
    for func_block in top_module.body:
        for op in func_block.operations:
            if isinstance(op, allo_d.GridMapOp):
                grid = list(op.grid)
                is_input = allo.GridMap.get_interface(op)
                for block in op.body:
                    for sub_op in block.operations:
                        if isinstance(sub_op, func_d.CallOp):
                            work_grids[sub_op.callee.value] = grid
                            break
                for i, (buf, shard) in enumerate(zip(op.tensors, op.sharding)):
                    sharding = [
                        Layout.Shard(s.value) if s.value >= 0 else Layout.Replicate
                        for s in shard
                    ]
                    arg = BlockArgument(buf)
                    dtensor = DTensor(
                        mapping=grid,
                        shape=tuple(arg.type.shape),
                        dtype=arg.type.element_type,
                        spec_list=[Layout(sharding)],
                        tile_shape=tuple(sub_op.operands_[i].type.shape),
                        is_input=is_input[i],
                        id_=arg.arg_number,
                    )
                    dtensors[sub_op.callee.value].append(dtensor)
                for i in range(len(op.tensors), len(sub_op.operands_)):
                    arg = BlockArgument(sub_op.operands_[i])
                    shape = tuple(arg.type.shape)
                    dtensor = DTensor(
                        mapping=grid,
                        shape=shape,
                        dtype=arg.type.element_type,
                        spec_list=[Layout([Layout.Replicate] * len(shape))],
                        tile_shape=shape,
                        is_input=None,
                        id_=arg.arg_number,
                    )
                    dtensors[sub_op.callee.value].append(dtensor)

    return {
        "grids": work_grids,
        "symbols": symbol_map,
        "symbol uses": symbol_uses_in_functions,
        "module": mod,
        "dtensors": dtensors,
    }


def is_resource(op):
    if isinstance(op, func_d.FuncOp):
        return False
    return True
