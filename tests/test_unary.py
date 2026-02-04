# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
from allo.ir.types import int32, Int, ConstExpr, bool


def test_unary():
    def kernel1() -> int32:
        A: int32 = 1
        B: int32 = +A
        return B

    s = process(kernel1)

    def kernel2() -> int32:
        A: int32 = 1
        B: int32 = -A
        return B

    s = process(kernel2)

    def kernel3() -> int32[10]:
        A: int32[10] = 0
        B: int32[10] = -A
        return B

    s = process(kernel3)

    def kernel4() -> int32[10]:
        A: int32[10] = 0
        B: int32[10] = +A
        return B

    s = process(kernel4)

    def kernel5() -> int32:
        A: int32 = +1
        return A

    s = process(kernel5)

    def kernel6() -> int32:
        A: int32 = -1
        return A

    s = process(kernel6)

    def kernel7() -> int32:
        A: int32 = 1
        B: int32 = 1
        C: int32 = 1
        A, B, C = -A, +B, -C
        return A + B + C

    s = process(kernel7)

    def kernel8() -> int32[10]:
        A: int32[10] = +1
        B: int32[10] = -1
        A, B = -A, +B
        return A + B

    s = process(kernel8)


def test_unary_not():
    def kernel1() -> bool:
        A: bool = 1 == 1
        B: bool = not A
        return B

    s = process(kernel1)

    def kernel2() -> bool:
        A: bool = 1 == 1
        B: bool = not A == 1
        return B

    s = process(kernel2)

    def kernel3() -> bool:
        A: bool = 1 == 1
        B: bool = not True
        return B

    s = process(kernel3)

    def kernel4() -> bool:
        A: bool = 1 == 1
        B: bool
        if not A:
            B: bool = not False
        else:
            B: bool = not True
        return B

    s = process(kernel4)


if __name__ == "__main__":
    test_unary()
    test_unary_not()
