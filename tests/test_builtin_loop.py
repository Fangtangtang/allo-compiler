# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
import allo
from allo.ir.types import int32, index, uint1, float32
from allo.dsl import grid
from allo.spmw import kernel


def test_grid_loop():
    @kernel
    def kernel1(A: int32[32], B: int32[32]) -> int32[32]:
        C: int32[32] = 0
        for i, _ in allo.grid(32, 2):
            C[i] = A[i] + B[i]
        return C

    s = process(kernel1)
    np_A = np.random.randint(0, 255, (32,), dtype=np.int32)
    np_B = np.random.randint(0, 255, (32,), dtype=np.int32)
    np_C = s(np_A, np_B)
    assert np.allclose(np_C, np_A + np_B)

    # @kernel
    # def kernel2(A: int32[32], B: int32[32]) -> int32[32]:
    #     C: int32[32] = 0
    #     for i, _ in grid(32, 2):
    #         C[i] = A[i] + B[i]
    #     return C

    # s = process(kernel2)
    # np_A = np.random.randint(0, 255, (32,), dtype=np.int32)
    # np_B = np.random.randint(0, 255, (32,), dtype=np.int32)
    # np_C = s(np_A, np_B)
    # assert np.allclose(np_C, np_A + np_B)

    # @kernel
    # def kernel3(A: int32[32], B: int32[32]) -> int32[32]:
    #     C: int32[32] = 0
    #     for i, _ in allo.grid(32, 2):
    #         C[i] = C[i] + A[i] + B[i]
    #     return C

    # s = process(kernel3)
    # np_A = np.random.randint(0, 255, (32,), dtype=np.int32)
    # np_B = np.random.randint(0, 255, (32,), dtype=np.int32)
    # np_C = s(np_A, np_B)
    # assert np.allclose(np_C, (np_A + np_B) * 2)


if __name__ == "__main__":
    test_grid_loop()
