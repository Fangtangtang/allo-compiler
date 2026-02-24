# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from collections import defaultdict
from .utils import parse_spmw_module, is_resource, collect_symbol_refs_in_function
from .meta_programming import unroll_meta_for
from .stream import replace_stream_arrays
import allo._mlir.extras.dialects.func as func
import allo._mlir.extras.attr as attr
from allo._mlir.passmanager import PassManager as mlir_pass_manager
from allo.utils import construct_kernel_name as construct_name
import allo._mlir.extras.types as mlir_types
from allo._mlir.ir import (
    Module,
    Location,
    InsertionPoint,
    StringAttr,
    FlatSymbolRefAttr,
    UnitAttr,
)
from allo._mlir.dialects import (
    allo as allo_d,
    arith as arith_d,
    func as func_d,
    sdy as sdy_d,
)


def instantiate_for_hls(module, top_name):
    """
    [NOTE]: hls backend do not support sharding for now
    """
    parsed = parse_spmw_module(module, top_name)

    mod = parsed["module"]
    symbol_map = parsed["symbols"]
    work_grids = parsed["grids"]
    dtensors = parsed["dtensors"]
    instance_copies = {}

    # construct raw function copies
    # TODO: reuse copies if they have same control flow
    with module.context, Location.unknown():
        copies = Module.create()
        with InsertionPoint(copies.body):
            for op in mod.body.operations:
                if isinstance(op, allo_d.StreamGlobalOp):
                    op.clone()
            for grid_info in work_grids.values():
                grid_shape = grid_info["grid"]
                orig_func_name = grid_info["work"]
                orig_func = symbol_map[orig_func_name]
                for dim in np.ndindex(*grid_shape):
                    func_ = orig_func.clone()
                    function_name = construct_name(orig_func_name, dim)
                    func_.sym_name = StringAttr.get(function_name)
                    instance_copies[function_name] = func_
                    # meta data
                    for func_block in func_.body:
                        for op in func_block.operations:
                            assert isinstance(op, func_d.CallOp)
                            assert op.callee.value.endswith("get_wid")  # FIXME: parsing
                            for res, num in zip(op.results, dim):
                                const_op = arith_d.ConstantOp(
                                    mlir_types.index(), num, ip=InsertionPoint(op)
                                )
                                res.replace_all_uses_with(const_op.result)
                            op.erase()
                            break
                        break

        unroll_meta_for(copies)
        pipeline = "builtin.module(lower-memcopy-ops, canonicalize)"
        mlir_pass_manager.parse(pipeline).run(copies.operation)
        replace_stream_arrays(copies)

    # construct top module
    top_module = symbol_map[top_name]
    resources = {}
    for k, v in symbol_map.items():
        if k.startswith(top_name) and is_resource(v):
            resources[k] = v

    new_resources = {}
    with InsertionPoint(mod.body), Location.unknown():
        top_func = func_d.FuncOp(top_name, top_module.type)
        entry_block = top_func.add_entry_block()
        attr.copy_attr(top_module, top_func, {"itypes", "otypes"}, allow_missing=False)
        entry_ip = InsertionPoint.at_block_begin(entry_block)
        top_func.attributes["df.kernel"] = UnitAttr.get()  # FIXME
        # move resource to local
        with entry_ip:
            for name, op in resources.items():
                if isinstance(op, allo_d.StreamGlobalOp):
                    shape = list(op.shape)
                    if len(shape) == 0:
                        names = [name]
                    else:
                        names = [
                            construct_name(name, dim) for dim in np.ndindex(*shape)
                        ]
                    for n in names:
                        stream = allo_d.StreamConstructOp(op.element_type.value)
                        new_resources[n] = stream
                else:
                    raise NotImplementedError

    with InsertionPoint.at_block_begin(mod.body), Location.unknown():
        # call ops
        for grid_info in work_grids.values():
            grid_shape = grid_info["grid"]
            orig_func_name = grid_info["work"]
            tensors = dtensors[orig_func_name]  # argument passing
            for dim in np.ndindex(*grid_shape):
                function_name = construct_name(orig_func_name, dim)
                instance = instance_copies[function_name]
                # [NOTE]: backend do not support `.` in function names, backend use `startswith` to identify top module
                hls_function_name = f"_{function_name.replace(".", "_")}"
                symbols: dict[str, list] = collect_symbol_refs_in_function(instance)
                symbol_names = sorted(symbols.keys())
                itypes = instance.type.inputs
                base_inputs = len(itypes)
                assert "itypes" in instance.attributes
                itype_hints = instance.attributes["itypes"].value
                args = [top_func.arguments[t.global_id] for t in tensors]
                for symbol in symbol_names:
                    resource_op = new_resources[symbol]
                    args.append(resource_op)
                    itypes.append(resource_op.result.type)
                    itype_hints += "_"  # FIXME
                new_func = func.function(
                    hls_function_name,
                    itypes,
                    [],
                    itype_hints=itype_hints,
                    otype_hints="",
                )
                final_op = func_d.ReturnOp([], ip=InsertionPoint(new_func.entry_block))
                # update arguments
                for i, arg in enumerate(instance.arguments):
                    arg.replace_all_uses_with(new_func.arguments[i])
                for op in instance.entry_block.operations:
                    op.operation.move_before(final_op)
                final_op.erase()
                # update resource usage
                for i, symbol in enumerate(symbol_names):
                    ops = symbols[symbol]
                    for op in ops:
                        if isinstance(op, allo_d.GlobalStreamGetOp):
                            new_op = allo_d.StreamGetOp(
                                op.result.type,
                                new_func.arguments[i + base_inputs],
                                [],
                                ip=InsertionPoint(op),
                            )
                            op.result.replace_all_uses_with(new_op.result)
                            op.erase()
                        elif isinstance(op, allo_d.GlobalStreamPutOp):
                            allo_d.StreamPutOp(
                                new_func.arguments[i + base_inputs],
                                [],
                                op.data,
                                ip=InsertionPoint(op),
                            )
                            op.erase()
                        else:
                            raise NotImplementedError
                # call the function in top module
                func_d.CallOp(
                    [],
                    FlatSymbolRefAttr.get(hls_function_name),
                    args,
                    ip=entry_ip,
                )
        func_d.ReturnOp([], ip=entry_ip)

    for op in mod.body.operations:
        if isinstance(op, allo_d.StreamGlobalOp) or isinstance(op, sdy_d.MeshOp):
            op.erase()
    return mod
