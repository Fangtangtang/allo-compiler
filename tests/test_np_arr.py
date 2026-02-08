# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
import numpy as np
from allo.ir.types import int32, bool, ConstExpr


def test_np_array():
    a = 1
    arr = np.array([[1, 2], [3, 4]])

    def kernel1() -> int32:
        tmp: int32[2, 2] = [[1, 2], [3, 4]]
        return tmp[0, 0]

    s = process(kernel1)
    assert s() == 1

    def kernel2() -> int32:
        # rhs must be a compile time constant
        tmp: int32[2, 2] = [[a, 2], [3, 4]]
        return tmp[0, 0]

    s = process(kernel2)
    assert s() == 1

    def kernel3() -> int32:
        # only need to construct the same constant tensor once
        tmp: int32[2, 2] = [[1, 2], [3, 4]]
        tmp1: int32[2, 2] = [[a, 2], [3, 4]]
        tmp2: int32[2, 2] = [[1, 2], [3, 4]]
        return tmp[0, 0] + tmp1[0, 0] + tmp2[0, 0]

    s = process(kernel3)
    assert s() == 3


if __name__ == "__main__":
    test_np_array()
