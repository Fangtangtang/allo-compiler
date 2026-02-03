# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
from allo.ir.types import int32, Int, ConstExpr, UInt


def test_arith():
    def kernel1() -> int32:
        A: int32 = 0
        B: int32 = 1
        A = 0 - 1
        A = B + B
        return A

    s = process(kernel1)
    assert s() == 2

    def kernel2() -> int32:
        A: int32 = 0
        B: int32 = 0
        A = B - 1
        return A

    s = process(kernel2)
    assert s() == -1

    def kernel3() -> int32:
        A: int32 = 0
        B: int32 = 1
        C: int32 = 0
        A, C = B * 2, 1 + B
        return A

    s = process(kernel3)
    assert s() == 2

    def kernel4() -> int32:
        A: int32 = 0
        B: int32 = 0
        A = B // 2
        return A

    s = process(kernel4)

    def kernel5() -> int32:
        A: int32 = 0
        B: int32 = 0
        A = B % 2
        return A

    s = process(kernel5)

    # TODO: ast.Pow not supported for now
    # def kernel6() -> int32:
    #     A: int32 = 0
    #     B: int32 = 0
    #     A = B**2
    #     return A

    # s = process(kernel6)

    def kernel7() -> int32:
        A: int32 = 0
        B: int32 = 0
        A = 1 + B + B + 2
        return A

    s = process(kernel7)


def test_broadcast():
    def kernel1() -> int32[10]:
        A: int32[10] = 0
        B: int32 = 1
        C: int32[10] = A + B
        return C

    s = process(kernel1)

    def kernel2() -> int32[10]:
        A: int32[10] = 0
        B: int32[1] = 1
        C: int32[10] = A + B
        return C

    s = process(kernel2)

    def kernel3() -> int32[10]:
        A: int32[10] = 0
        C: int32[10] = A + 1
        return C

    s = process(kernel3)


def test_compare():
    def kernel1():
        A: int32 = 0
        B: int32 = 0
        C: UInt(8) = A == B

    s = process(kernel1)

    def kernel2():
        A: int32 = 0
        B: int32 = 0
        C: UInt(8) = A != B

    s = process(kernel2)

    def kernel3():
        A: int32 = 0
        B: int32 = 0
        C: UInt(8) = A > B

    s = process(kernel3)

    def kernel4():
        A: int32 = 0
        B: int32 = 0
        C: UInt(8) = A >= B

    s = process(kernel4)

    def kernel5():
        A: int32 = 0
        B: int32 = 0
        C: UInt(8) = A < B

    s = process(kernel5)

    def kernel6():
        A: int32 = 0
        B: int32 = 0
        C: UInt(8) = A <= B

    s = process(kernel6)

    def kernel7():
        A: int32 = 0
        B: UInt(8) = 0 <= A
        C: UInt(8) = A > 0

    s = process(kernel7)

    def kernel8():
        C: UInt(8) = 1 >= 0

    s = process(kernel8)

    def kernel9():
        C: UInt(8) = 1 < 0

    s = process(kernel9)


if __name__ == "__main__":
    test_arith()
    # test_broadcast()
    # test_compare()
