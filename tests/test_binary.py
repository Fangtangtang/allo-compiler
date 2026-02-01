# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
from allo.ir.types import int32, Int, ConstExpr


def test_arith():
    def kernel1() -> int32:
        A: int32 = 0
        B: int32 = 0
        A = 0 - 1
        A = B + B
        return A

    s = process(kernel1)

    def kernel2() -> int32:
        A: int32 = 0
        B: int32 = 0
        A = B - 1
        return A

    s = process(kernel2)

    def kernel3() -> int32:
        A: int32 = 0
        B: int32 = 0
        C: int32 = 0
        A, C = B * 2, 1 + B
        return A

    s = process(kernel3)

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


if __name__ == "__main__":
    test_arith()
