# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Union
import numpy as np
from collections.abc import Callable
from .ir.utils import SymbolTable, get_global_vars
from .ir.ast_processor import ASTProcessor
from .ir.ir_builder import IRBuilder
from allo.utils import register_dialect, construct_kernel_name
from allo._mlir.ir import Module, Context, Location, InsertionPoint, StringAttr
from allo._mlir.dialects import (
    allo as allo_d,
    arith as arith_d,
    func as func_d,
    sdy as sdy_d,
)
from allo._mlir.passmanager import PassManager as mlir_pass_manager


def parse(fn: Union[Callable, str], instantiate: list = None):
    symbol_table = SymbolTable()
    ast_processor = ASTProcessor(symbol_table, global_symbols=get_global_vars(fn))
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

        symbol_map = {}
        for op in new_module.body.operations:
            if "sym_name" in op.attributes:
                name = str(op.attributes["sym_name"]).strip('"')
                symbol_map[name] = op

        # with open("module.json", "w") as f:
        #     json.dump(symbol_map, f, indent=2)

    top_module = symbol_map[top_name]
    work_grids = {}
    for func_block in top_module.body:
        for op in func_block.operations:
            if isinstance(op, sdy_d.ManualComputationOp):
                mesh = None
                in_shardings = sdy_d.TensorShardingPerValueAttr(
                    op.in_shardings
                ).shardings
                for shard in in_shardings:
                    shard = sdy_d.TensorShardingAttr(shard)
                    mesh = shard.mesh_or_ref
                    dims = shard.dimension_shardings
                    # print(dims) # TODO: parse dim
                out_shardings = sdy_d.TensorShardingPerValueAttr(
                    op.out_shardings
                ).shardings
                for shard in out_shardings:
                    shard = sdy_d.TensorShardingAttr(shard)
                    mesh = shard.mesh_or_ref
                    dims = shard.dimension_shardings
                    # print(dims) # TODO: parse dim
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

    with context as ctx, Location.unknown():
        mod = Module.create()

    with InsertionPoint(mod.body), Location.unknown():
        for mesh_name, grid_info in work_grids.items():
            grid_shape = grid_info["grid"]
            orig_func = symbol_map[grid_info["work"]]
            for dim in np.ndindex(*grid_shape):
                func = orig_func.clone()
                func.sym_name = StringAttr.get(
                    construct_kernel_name(grid_info["work"], dim)
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
                                arith_d.IndexType.get(), num, ip=InsertionPoint(op)
                            )
                            res.replace_all_uses_with(const_op.result)
                        op.erase()
                        break
                    break

    print(mod)

    pipeline = "builtin.module(canonicalize)"
    with context:
        mlir_pass_manager.parse(pipeline).run(mod.operation)

    print(mod)
