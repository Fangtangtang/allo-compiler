# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from src.main import process
from allo.ir.types import int32, UInt, ConstExpr


def test_branch():
    def kernel1() -> int32:
        A: int32 = 0
        B: int32 = 0
        if A > B:
            B = A
        if True:
            B = 1
        return B

    s = process(kernel1)

    def kernel2() -> int32:
        A: int32 = 0
        B: int32 = 0
        if A > B:
            B = A
        else:
            B = B
        return B

    s = process(kernel2)

    def kernel3() -> int32:
        A: int32 = 0
        B: int32 = 0
        if A > B:
            B = A
        elif A < B:
            B = B
        else:
            B = 0
        return B

    s = process(kernel3)


def test_branch_complicate():
    def kernel1() -> int32:
        A: int32 = 0
        B: int32 = 0
        if A > B and False or B == 1:
            B = A
        if True:
            B = 1
        return B

    s = process(kernel1)

    def kernel2() -> int32:
        A: int32 = 0
        B: int32 = 0
        if A > B:
            B = A
        elif A < B or B == 0:
            B = B
        else:
            B = 0
        return B

    s = process(kernel2)

    def kernel3() -> int32:
        A: int32 = 0
        B: int32 = 0
        if True and A > B:
            B = A
        elif A < B or B == 0 and False:
            B = B
        else:
            B = 0
        return B

    s = process(kernel3)


if __name__ == "__main__":
    # test_branch()
    test_branch_complicate()
