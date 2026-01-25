# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from src.main import process
from allo.ir.types import int32

def test_assign_logic():
    def kernel1(A: int32) -> int32:
        B: int32 = 0
        if A > B:
            B = A
        return B

    s = process(kernel1)
    # print(s.module)
    # mod = s.build()
    # assert mod(2) == kernel1(2)

if __name__ == "__main__":
    test_assign_logic()
