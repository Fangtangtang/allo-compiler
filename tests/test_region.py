# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
from allo.ir.types import int32, Int
from allo import spmw


def test_region():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[1], args=[A, B])
        def core(local_A: int32[1024], local_B: int32[1024]):
            local_B[:] = local_A + 1

    s = process(top)


if __name__ == "__main__":
    test_region()
