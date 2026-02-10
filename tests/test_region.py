# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process_spmw, process
from allo.ir.types import int32, Int
from allo import spmw


def test_region():
    @spmw.kernel
    def kernel1[Ty, M](A: "Ty[M]") -> "Ty[M]":
        B: Ty[M] = A
        return B

    @spmw.unit()
    def top2[Ty, M](A: "Ty[M]", B: "Ty[M]"):
        @spmw.work(mapping=[1], args=[A, B])
        def core(local_A: "Ty[M]", local_B: "Ty[M]"):
            local_B[:] = kernel1[Ty, M](local_A)

    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[1], args=[A, B])
        def core(local_A: int32[1024], local_B: int32[1024]):
            top2[int32, 1024](local_A, local_B)

    s = process_spmw(top)


if __name__ == "__main__":
    test_region()
