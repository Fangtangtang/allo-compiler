# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union
import numpy as np
from collections.abc import Callable
from collections import defaultdict
from .ir.utils import SymbolTable, get_global_vars
from .ir.ast_preprocessor import ASTPreProcessor
from .ir.ir_builder import IRBuilder
from .passes.utils import parse_spmw_module
from .passes.meta_programming import unroll_meta_for
from allo.utils import register_dialect, construct_kernel_name
import allo._mlir.extras.types as mlir_types
from .passes.stream import replace_stream_arrays
from allo._mlir.ir import (
    Module,
    Context,
    Location,
    InsertionPoint,
    StringAttr,
)
from allo._mlir.dialects import arith as arith_d, func as func_d
from allo._mlir.passmanager import PassManager as mlir_pass_manager
from allo.backend.aie import AIE_MLIRModule
from allo.backend.aie.utils import Argument


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

    parsed = parse_spmw_module(new_module, top_name)

    mod = parsed["module"]
    symbol_map = parsed["symbols"]
    work_grids = parsed["grids"]
    dtensors = parsed["dtensors"]

    func_instances = defaultdict(dict)
    core_func_args = defaultdict(dict)
    with InsertionPoint(mod.body), Location.unknown():
        for mesh_name, grid_info in work_grids.items():
            grid_shape = grid_info["grid"]
            orig_func_name = grid_info["work"]
            orig_func = symbol_map[orig_func_name]
            for dim in np.ndindex(*grid_shape):
                func = orig_func.clone()
                function_name = construct_kernel_name(orig_func_name, dim)
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
                        )  # call get_wid FIXME: parsing
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

    unroll_meta_for(mod)
    pipeline = "builtin.module(canonicalize)"
    with context:
        mlir_pass_manager.parse(pipeline).run(mod.operation)

    replace_stream_arrays(mod)

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
