# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
from allo.ir.types import int32, Int, ConstExpr, Fixed
import numpy as np


def test_annassign():
    """
    Test the annotated assignment.
    """
    zero = 0
    one = 1 + zero

    def kernel1() -> int32:
        """
        Initialize variables with constants.
        """
        A: int32 = 0
        B: int32 = zero
        C: ConstExpr[int32] = one
        D: ConstExpr[int32] = C + 2
        E: int32 = C + 1
        F: int32 = C
        return E

    s = process(kernel1)
    assert s() == one + 1

    def kernel2(A: int32) -> int32:
        """
        Initialize variables with arguments or varibales.
        """
        B: int32 = A
        C: int32 = B
        return C

    s = process(kernel2)
    assert s(1) == 1
    assert s(2) == 2
    assert s(-1) == -1

    def kernel3(a: int32[8]) -> int32[8]:
        b: int32[8] = a
        return b

    s = process(kernel3)
    np_A = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_A, np_B)

    print("test_annassign passed!")


def test_assign():
    def kernel1() -> int32:
        B: int32 = 0
        A = B
        return A

    s = process(kernel1)

    def kernel2() -> int32:
        A: int32 = 0
        B: int32 = 0
        C: int32 = 0
        A, C = B, B
        return A

    s = process(kernel2)

    def kernel3() -> int32[2]:
        b: int32[2]
        b[0] = 1
        b[1] = 0
        a: int32[2, 2]
        a[0] = b
        a[1, :] = b[:]
        return b

    s = process(kernel3)


def test_augassign():
    def kernel1() -> int32:
        A: int32 = 0
        B: int32 = 0
        A += B
        return A

    s = process(kernel1)

    def kernel2() -> int32:
        A: int32[2] = 0
        B: int32 = 0
        A[0] += B
        return A[0]

    s = process(kernel2)

    def kernel3() -> int32:
        A: int32 = 0
        B: int32 = 0
        A -= B
        return A

    s = process(kernel3)

    def kernel4() -> int32:
        A: int32 = 0
        B: int32 = 0
        A *= B
        return A

    s = process(kernel4)

    def kernel5() -> int32:
        A: int32[2] = 0
        A[0] -= 1
        return A[0]

    s = process(kernel5)

    def kernel6() -> int32:
        A: int32[2, 2] = 0
        A[0] -= 1
        return A[0, 0]

    s = process(kernel6)


def test_broadcast_init():
    def kernel1() -> int32[2]:
        a: int32[2] = 1
        b: int32[32, 32] = 0
        return a

    s = process(kernel1)
    assert np.array_equal(s(), np.array([1, 1], dtype=np.int32))

    def kernel2() -> int32:
        a: int32 = 1
        b: int32[32, 32] = a
        return b[0, 0]

    s = process(kernel2)

    def kernel3() -> int32:
        a: int32[32] = 1
        b: int32[4, 32] = a
        return b[0, 0]

    s = process(kernel3)


if __name__ == "__main__":
    # test_annassign()
    # test_assign()
    # test_augassign()
    # test_broadcast_init()
    def kernel3() -> int32[2]:
        b: int32[2]
        b[0] = 1
        # b[1] = 0
        a: int32[2, 2]
        # a[0] = b
        # a[1, :] = b[:]
        return b

    s = process(kernel3)