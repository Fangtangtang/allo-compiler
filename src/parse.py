# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Union
import numpy as np
from collections.abc import Callable
from collections import defaultdict
from .ir.utils import SymbolTable, get_global_vars
from .ir.ast_preprocessor import ASTPreProcessor
from .ir.ir_builder import IRBuilder
from .passes.memory import DTensor
from allo.utils import register_dialect, construct_kernel_name
import allo._mlir.extras.types as mlir_types
from .passes.stream_transform import replace_stream_arrays
from allo._mlir.ir import (
    Module,
    Context,
    Location,
    InsertionPoint,
    StringAttr,
    BlockArgument,
)
from allo._mlir.dialects import (
    allo as allo_d,
    arith as arith_d,
    func as func_d,
    memref as memref_d,
    sdy as sdy_d,
)
from allo._mlir.passmanager import PassManager as mlir_pass_manager
from allo.memory import Layout
from allo.backend.aie import AIE_MLIRModule
from allo.backend.aie.utils import Argument


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


def parse(fn: Union[Callable, str], instantiate: list = None):
    symbol_table = SymbolTable()
    ast_processor = ASTPreProcessor(symbol_table, global_symbols=get_global_vars(fn))
    # process the top function
    node, top_name = ast_processor.process(fn, instantiate=instantiate)
    builder = IRBuilder(symbol_table)
    module = builder.build()

    with open("module.mlir", "w") as f:
        f.write(str(module))

    with open("module.mlir", "r") as f:
        module_content = f.read()

    context = Context()
    with context as ctx:
        register_dialect(ctx, True)
        new_module = Module.parse(module_content)

    with context as ctx, Location.unknown():
        mod = Module.create()

    symbol_map = {}
    for op in new_module.body.operations:
        if "sym_name" in op.attributes:
            name = str(op.attributes["sym_name"]).strip('"')
            symbol_map[name] = op
        if isinstance(op, allo_d.StreamGlobalOp):
            with InsertionPoint(mod.body), Location.unknown():
                op.clone()

    top_module = symbol_map[top_name]
    work_grids = {}
    dtensors = defaultdict(list)
    for func_block in top_module.body:
        for op in func_block.operations:
            if isinstance(op, sdy_d.ManualComputationOp):
                mesh = None
                in_shardings = sdy_d.TensorShardingPerValueAttr(
                    op.in_shardings
                ).shardings
                tensors = []
                assert len(op.tensors) == len(in_shardings)
                for shard, tensor in zip(in_shardings, op.tensors):
                    shard = sdy_d.TensorShardingAttr(shard)
                    mesh = shard.mesh_or_ref
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
                    mesh = shard.mesh_or_ref
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
                mesh_op = symbol_map[mesh.value]
                mesh_attr = sdy_d.MeshAttr(mesh_op.mesh)
                mesh_map = {}
                for ax in mesh_attr.axes:
                    ax = sdy_d.MeshAxisAttr(ax)
                    mesh_map[ax.name] = ax.size
                grid = [mesh_map[str(i)] for i in range(len(mesh_map))]
                for block in op.body:
                    for sub_op in block.operations:
                        if isinstance(sub_op, func_d.CallOp):
                            work_grids[mesh.value] = {
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

    func_instances = defaultdict(dict)
    core_func_args = defaultdict(dict)
    with InsertionPoint(mod.body), Location.unknown():
        for mesh_name, grid_info in work_grids.items():
            grid_shape = grid_info["grid"]
            orig_func = symbol_map[grid_info["work"]]
            orig_func_name = grid_info["work"]
            for dim in np.ndindex(*grid_shape):
                func = orig_func.clone()
                function_name = construct_kernel_name(grid_info["work"], dim)
                func.sym_name = StringAttr.get(function_name)
                func.attributes["tag"] = func.sym_name  # for aie backend
                func_instances[orig_func_name][dim] = function_name
                for i, dtensor in enumerate(dtensors[orig_func_name]):
                    core_func_args[function_name][i] = (
                        Argument(dtensor, None),
                        dtensor.is_input,
                    )
                for func_block in func.body:
                    for op in func_block.operations:
                        assert isinstance(
                            op, func_d.CallOp
                        ) and op.callee.value.endswith(
                            "get_wid"
                        )  # call get_wid
                        for res, num in zip(op.results, dim):
                            const_op = arith_d.ConstantOp(
                                mlir_types.index(), num, ip=InsertionPoint(op)
                            )
                            res.replace_all_uses_with(const_op.result)
                        op.erase()
                        break
                    break

    with open("unrolled_module.mlir", "w") as f:
        f.write(str(mod))

    pipeline = "builtin.module(canonicalize)"
    with context:
        mlir_pass_manager.parse(pipeline).run(mod.operation)

    # replace_stream_arrays(mod)

    with open("simplified_module.mlir", "w") as f:
        f.write(str(mod))

    return mod, func_instances, core_func_args, dtensors


def to_aie(fn: Union[Callable, str], instantiate: list = None):
    mod, func_instances, core_func_args, dtensors = parse(fn, instantiate)
    aie_mod = AIE_MLIRModule(mod, project_dir="top.prj", func_instances=func_instances)
    global_dtensors = {}
    for dtensor_list in dtensors.values():
        for dt in dtensor_list:
            global_dtensors[dt.global_id] = dt

    aie_mod.init_spmw(core_func_args, global_dtensors)
    aie_mod.build(skip=True)
    return aie_mod
