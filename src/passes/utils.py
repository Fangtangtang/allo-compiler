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
from allo._mlir.dialects import (
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

    def get_shard(dims):
        shard = []
        for x in dims:
            axes = sdy_d.DimensionShardingAttr(x).axes
            if len(axes) == 0:
                shard.append(Layout.Replicate)
            else:
                assert len(axes) == 1
                shard.append(Layout.Shard(int(sdy_d.AxisRefAttr(axes[0]).name)))
        return shard

    for func_block in top_module.body:
        for op in func_block.operations:
            if isinstance(op, sdy_d.ManualComputationOp):
                in_shardings = sdy_d.TensorShardingPerValueAttr(
                    op.in_shardings
                ).shardings
                tensors = []
                assert len(op.tensors) == len(in_shardings)
                for shard, tensor in zip(in_shardings, op.tensors):
                    shard = sdy_d.TensorShardingAttr(shard)
                    arg = BlockArgument(tensor.owner.buffer)
                    dims = shard.dimension_shardings
                    tensors.append(
                        {
                            "id": arg.arg_number,
                            "type": arg.type,
                            "shard": get_shard(dims),
                            "is_input": True,
                        }
                    )
                out_shardings = sdy_d.TensorShardingPerValueAttr(
                    op.out_shardings
                ).shardings
                assert len(op.results) == len(out_shardings)
                for shard, tensor in zip(out_shardings, op.results):
                    shard = sdy_d.TensorShardingAttr(shard)
                    dims = shard.dimension_shardings
                    for use in tensor.uses:
                        for use in use.owner.result.uses:
                            assert isinstance(use.owner, memref_d.CopyOp)
                            arg = BlockArgument(use.owner.operands[1])
                    tensors.append(
                        {
                            "id": arg.arg_number,
                            "type": arg.type,
                            "shard": get_shard(dims),
                            "is_input": False,
                        }
                    )
                mesh_name = sdy.SPMD.get_mesh(op).value
                mesh_attr = sdy_d.MeshAttr(symbol_map[mesh_name].mesh)
                mesh_map = {}
                for ax in mesh_attr.axes:
                    ax = sdy_d.MeshAxisAttr(ax)
                    mesh_map[ax.name] = ax.size
                grid = [mesh_map[str(i)] for i in range(len(mesh_map))]
                for block in op.body:
                    for sub_op in block.operations:
                        if isinstance(sub_op, func_d.CallOp):
                            work_grids[mesh_name] = {
                                "grid": grid,
                                "work": sub_op.callee.value,
                            }
                            break
                for tensor, operand in zip(tensors, sub_op.operands_):
                    dtensor = DTensor(
                        mapping=grid,
                        shape=tuple(tensor["type"].shape),
                        dtype=tensor["type"].element_type,
                        spec_list=[Layout(tensor["shard"])],
                        tile_shape=tuple(operand.type.shape),
                        is_input=tensor["is_input"],
                        id_=tensor["id"],
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
    if isinstance(op, sdy_d.MeshOp):
        return False
    return True
