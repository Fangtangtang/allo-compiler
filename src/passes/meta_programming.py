# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo._mlir.dialects import allo as allo_d, func as func_d


def unroll_meta_for(module):
    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp):
            allo_d.unroll(op)
