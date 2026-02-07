# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
from allo.ir.types import bool, int8, int32, float32, float16, index


def test_multi_return():
    def kernel(A: int32, B: int32) -> (int32, int32):
        res0: int32 = 0
        res1: int32 = 0
        return res0, res1

    s = process(kernel)
    # [NOTE]: for llvm backend, When returning multiple values, we only support all tensors.

    def kernel(A: int32[8], B: int32[8]) -> (int32[8], int32[8]):
        res0: int32[8] = 0
        res1: int32[8] = 0
        for i in range(8):
            res0[i] = A[i] + 1
            res1[i] = B[i] + 1
        return res0, res1

    s = process(kernel)
    np_A = np.random.randint(0, 10, size=(8,)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(8,)).astype(np.int32)
    np_res0, np_res1 = s(np_A, np_B)
    assert np.array_equal(np_res0, np_A + 1)
    assert np.array_equal(np_res1, np_B + 1)


if __name__ == "__main__":
    test_multi_return()
