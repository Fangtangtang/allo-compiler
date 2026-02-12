# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process_spmw
from allo.ir.types import int32, ConstExpr, index
from allo import spmw
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


def test_get_wid_1D():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(local_A: int32[1024] @ [S(0)], local_B: int32[1024] @ [S(0)]):
            pi: ConstExpr[index] = spmw.get_wid()
            local_B[:] = local_A + pi

    s = process_spmw(top)

    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(local_A: int32[1024] @ [S(0)], local_B: int32[1024] @ [S(0)]):
            pi: ConstExpr[index] = spmw.get_wid()
            if pi > 1:
                local_B[:] = local_A + pi
            else:
                local_B[:] = local_A - pi

    s = process_spmw(top)


def test_get_wid_2D():
    @spmw.unit()
    def top(A: int32[64, 64], B: int32[64, 64], C: int32[64, 64], D: int32[64, 64]):
        @spmw.work(mapping=[2, 2], inputs=[A], outputs=[B])
        def core1(
            local_A: int32[64, 64] @ [S(0), S(1)], local_B: int32[64, 64] @ [S(0), S(1)]
        ):
            pi, pj = spmw.get_wid()
            local_B[:, :] = local_A + pi - pj

        @spmw.work(mapping=[2, 2], inputs=[C], outputs=[D])
        def core2(
            local_A: int32[64, 64] @ [S(0), S(1)], local_B: int32[64, 64] @ [S(0), S(1)]
        ):
            pi, pj = spmw.get_wid()
            if pi > pj:
                local_B[:, :] = local_A + pi - pj
            else:
                local_B[:, :] = local_A

    s = process_spmw(top)


if __name__ == "__main__":
    test_get_wid_1D()
    test_get_wid_2D()
