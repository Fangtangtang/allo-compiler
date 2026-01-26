# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from src.main import process
from allo.ir.types import int32, Int


def test_type():
    def kernel1(A: Int(32)) -> Int(32):
        B: Int(32) @ Stateful = 0
        if A > B:
            B = A
        return B

    s = process(kernel1)

    def kernel2(a: Int(32)[32]) -> Int(32)[32]:
        b: Int(32)[32] @ Stateful = 0
        for i in range(32):
            b[i] = a[i] + 1
        return b

    s = process(kernel2)


if __name__ == "__main__":
    test_type()
