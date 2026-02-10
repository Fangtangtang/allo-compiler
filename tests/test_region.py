# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process_spmw, process
from allo.ir.types import int32, Int
from allo import spmw


def test_region():
    @spmw.kernel
    def k(A: int32[1024]) -> int32[1024]:
        return A + 1

    @spmw.unit()
    def top2(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[1], args=[A, B])
        def core(local_A: int32[1024], local_B: int32[1024]):
            local_B[:] = k(local_A)

    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[1], args=[A, B])
        def core(local_A: int32[1024], local_B: int32[1024]):
            top2(local_A, local_B)

    s = process_spmw(top)


if __name__ == "__main__":
    test_region()
