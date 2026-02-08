# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
import numpy as np
from allo.ir.types import int16, int32, float32, bool, ConstExpr


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

    def kernel3() -> int32:
        # only need to construct the same constant tensor once
        tmp: int32[2, 2] = [[1, 2], [3, 4]]
        # cast to target type
        tmp1: float32[2, 2] = [[a, 2], [3, 4]]
        # cast to target type
        tmp2: int16[2, 2] = [[1, 2], [3, 4]]
        return tmp[0, 0] + tmp1[0, 0] + tmp2[0, 0]

    s = process(kernel3)
    assert s() == 3

    def kernel4() -> int32:
        # rhs can be a global constant np.array
        tmp: int32[2, 2] = arr
        return tmp[0, 0]

    s = process(kernel4)
    assert s() == arr[0][0]

    def kernel5() -> int32:
        tmp: int32[1] = [1]
        tmp: int32[1] = [a]
        return tmp[0]

    s = process(kernel5)
    assert s() == a

    def kernel5() -> int32:
        tmp: int32[1] = [1]
        tmp = [a]
        return tmp[0]

    s = process(kernel5)
    assert s() == a

    def kernel6() -> int32[2, 2]:
        return arr

    s = process(kernel6)
    assert np.array_equal(s(), arr)

    def kernel7() -> int32[2, 2]:
        return [[1, 2], [3, 4]]

    s = process(kernel7)
    assert np.array_equal(s(), np.array([[1, 2], [3, 4]]))

    def kernel8() -> int32[2, 2]:
        ones: int32[2, 2] = 1
        return [[1, 2], [3, 4]] + ones

    s = process(kernel8)
    assert np.array_equal(s(), np.array([[2, 3], [4, 5]]))

    def kernel8() -> int32[2, 2]:
        ones: int32[2, 2] = 1
        return arr * ones

    s = process(kernel8)
    assert np.array_equal(s(), arr)

    print("test_np_array passed")


if __name__ == "__main__":
    test_np_array()
