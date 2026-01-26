# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from src.main import process
from allo.ir.types import int32, Int


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

    def kernel2(A: Int(32)) -> Int(32):
        B: Int(32) = 0
        if A > B:
            B = A
        return B

    s = process(kernel2)

    def kernel2(A: Int(bits=32)) -> Int(32):
        B: Int(32) = 0
        if A > B:
            B = A
        return B

    s = process(kernel2)

    def kernel3(a: int32[32]) -> int32[32]:
        b: int32[32]
        for i in range(32):
            b[i] = a[i] + 1
        return b

    s = process(kernel3)

    def kernel4(a: Int(32)[32]) -> Int(32)[32]:
        b: Int(32)[32]
        for i in range(32):
            b[i] = a[i] + 1
        return b

    s = process(kernel4)

    def kernel5(A: Int(32)) -> Int(32):
        B: Int(32) = 0
        c: ConstExpr[int32] = 0
        if A > B:
            B = A
        return B

    s = process(kernel5)


if __name__ == "__main__":
    test_assign_logic()
