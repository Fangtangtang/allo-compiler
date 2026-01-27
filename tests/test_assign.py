# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from src.main import process
import allo
from allo.ir.types import int32, Int, ConstExpr


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
        return B

    """
    Parsed:
    def kernel1() -> int32:
        A: __allo__[i32, (), None] = 0
        B: __allo__[i32, (), None] = 0
        C: __allo__[<class 'allo.ir.types.ConstExpr'>, (), None] = 1
        D: __allo__[<class 'allo.ir.types.ConstExpr'>, (), None] = 3
        return B
    """
    s = process(kernel1)

    def kernel2(A: int32) -> int32:
        """
        Initialize variables with arguments or varibales.
        """
        B: int32 = A
        C: int32 = B
        return C

    s = process(kernel2)

    def kernel3(a: int32[32]) -> int32[32]:
        b: int32[32] = a
        return b

    s = process(kernel3)


def test_assign():
    # def kernel1() -> int32:
    #     B: int32 = 0
    #     A = B
    #     return A

    # s = process(kernel1)

    def kernel2() -> int32:
        A: int32 = 0
        B: int32 = 0
        C: int32 = 0
        A, C = B, B
        return A

    s = process(kernel2)
    """
    Parsed:
    def kernel2() -> int32:
        A: __allo__[i32, (), None] = 0
        B: __allo__[i32, (), None] = 0
        C: __allo__[i32, (), None] = 0
        A: __allo__[i32, (), None] = B
        C: __allo__[i32, (), None] = B
        return A    
    """

    def kernel3() -> int32[2]:
        b: int32[2]
        b[0] = 1
        b[1] = 0
        a: int32[2, 2]
        a[0] = b
        a[1, :] = b[:]
        return b

    s = process(kernel3)
    """
    def kernel3() -> int32[2]:
        b: __allo__[i32, (2,), <allo.memory.Layout object at 0x7b69aeff0d10>]
        b[0]: __allo__[i32, (), None] = 1
        b[1]: __allo__[i32, (), None] = 0
        a: __allo__[i32, (2, 2), <allo.memory.Layout object at 0x7b69aee09bb0>]
        a[0]: __allo__[i32, (2,), None] = b
        a[1, 0:2:1]: __allo__[i32, (2,), None] = b[0:2:1]
        return b    
    """


def test_augassign():
    def kernel1() -> int32:
        A: int32 = 0
        B: int32 = 0
        A += B
        return A

    s = process(kernel1)


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
    # test_annassign()
    test_assign()
    # test_augassign()
    # test_assign_logic()
