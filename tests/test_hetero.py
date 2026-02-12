# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process_spmw
from allo.ir.types import int32, ConstExpr, index
from allo import spmw
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


def test_get_wid():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(local_A: int32[1024] @ [S(0)], local_B: int32[1024] @ [S(0)]):
            pi: ConstExpr[index] = spmw.get_wid()
            local_B[:] = local_A + 1

    s = process_spmw(top)


if __name__ == "__main__":
    test_get_wid()
