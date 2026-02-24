# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from collections import defaultdict
from .utils import parse_spmw_module, is_resource
import allo._mlir.extras.attr as attr
from allo.utils import construct_kernel_name as construct_name
from allo._mlir.ir import (
    Location,
    InsertionPoint,
    StringAttr,
    BlockArgument,
)
from allo._mlir.dialects import (
    allo as allo_d,
    arith as arith_d,
    func as func_d,
)


def instantiate_for_hls(module, top_name):
    parsed = parse_spmw_module(module, top_name)

    mod = parsed["module"]
    symbol_map = parsed["symbols"]

    top_module = symbol_map[top_name]
    resources = {}
    for k, v in symbol_map.items():
        if k.startswith(top_name) and is_resource(v):
            resources[k] = v

    streams = {}
    with InsertionPoint(mod.body), Location.unknown():
        top_func = func_d.FuncOp(top_name, top_module.type)
        entry_block = top_func.add_entry_block()
        attr.copy_attr(top_module, top_func, {"itypes", "otypes"}, allow_missing=False)

        with InsertionPoint.at_block_begin(entry_block):
            # move resource to local
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
                        streams[n] = stream
                else:
                    raise NotImplementedError

            func_d.ReturnOp([])
        print(top_func)
