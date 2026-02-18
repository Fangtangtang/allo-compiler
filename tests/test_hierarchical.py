# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.parse import parse, to_aie
import allo
from allo.ir.types import int32, Stream
from allo import spmw
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


def test_h1():
    @spmw.unit()
    def vadd(A: int32[1, 1024], B: int32[1, 1024]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(
            local_A: int32[1, 1024] @ [R, S(0)], local_B: int32[1, 1024] @ [R, S(0)]
        ):
            local_B[0] = local_A[0]

    @spmw.unit()
    def top(A: int32[4, 1024], B: int32[4, 1024]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(
            local_A: int32[4, 1024] @ [S(0), R], local_B: int32[4, 1024] @ [S(0), R]
        ):
            vadd(local_A, local_B)

    s = parse(top)


if __name__ == "__main__":
    test_h1()
