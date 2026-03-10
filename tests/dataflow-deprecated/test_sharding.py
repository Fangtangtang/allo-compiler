# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process_spmw
from allo.ir.types import int32
from allo import spmw
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


def test_shard_1D():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(local_A: int32[1024] @ [S(0)], local_B: int32[1024] @ [S(0)]):
            local_B[:] = local_A + 1

    s = process_spmw(top)


def test_shard_2D():
    LyA = [S(0), R]
    M, N = 64, 64

    @spmw.unit()
    def top(A: int32[M, N], B: int32[M, N]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(local_A: int32[M, N] @ LyA, local_B: int32[M, N] @ LyA):
            local_B[:, :] = local_A + 1

    s = process_spmw(top)

    @spmw.unit()
    def top(A: int32[64, 64], B: int32[64, 64]):
        @spmw.work(mapping=[2, 2], inputs=[A], outputs=[B])
        def core(
            local_A: int32[64, 64] @ [S(0), S(1)], local_B: int32[64, 64] @ [S(0), S(1)]
        ):
            local_B[:, :] = local_A + 1

    s = process_spmw(top)


if __name__ == "__main__":
    test_shard_1D()
    test_shard_2D()
