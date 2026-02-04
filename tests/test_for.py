# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
from allo.ir.types import bool, int8, int32, float32


def test_vadd():
    def vadd(A: float32[32], B: float32[32], C: float32[32]):
        for i in range(32):
            # syntax sugar for lib op "add"
            C[i] = A[i] + B[i]

    s = process(vadd)
    np_A = np.random.rand(32).astype(np.float32)
    np_B = np.random.rand(32).astype(np.float32)
    np_C = np.zeros((32,), dtype=np.float32)
    vadd(np_A, np_B, np_C)
    np_D = np.zeros((32,), dtype=np.float32)
    s(np_A, np_B, np_D)
    np.testing.assert_allclose(np_C, np_D)

    # def madd(A: float32[32, 4], B: float32[32, 4], C: float32[32, 4]):
    #     for i in range(32):
    #         C[i] = A[i] + B[i]

    # s = process(madd)
    # np_A = np.random.rand(32, 4).astype(np.float32)
    # np_B = np.random.rand(32, 4).astype(np.float32)
    # np_C = np.zeros((32, 4), dtype=np.float32)
    # madd(np_A, np_B, np_C)
    # np_D = np.zeros((32, 4), dtype=np.float32)
    # s(np_A, np_B, np_D)
    # np.testing.assert_allclose(np_C, np_D)

    print("pass test_vadd")


def test_range_for():
    def kernel(A: int32[20]):
        for i in range(10):
            A[i] = i
        for i in range(10, 20):
            A[i] = i
        for i in range(0, 20, 2):
            A[i] = i  # * 2

    s = process(kernel)
    # print(s.module)
    # mod = s.build()
    # np_A = np.zeros((20,), dtype=np.int32)
    # kernel(np_A)
    # np_B = np.zeros((20,), dtype=np.int32)
    # mod(np_B)
    # np.testing.assert_allclose(np_A, np_B)


def test_variable_bound_for():
    def kernel(A: int32[10]):
        for i in range(10):
            for j in range(i + 1, 10):
                for k in range(j * 2, 10):
                    A[k] += i - j

    s = process(kernel)
    # print(s.module)
    # mod = s.build()
    # np_A = np.zeros((10,), dtype=np.int32)
    # kernel(np_A)
    # np_B = np.zeros((10,), dtype=np.int32)
    # mod(np_B)
    # np.testing.assert_allclose(np_A, np_B)


def test_variable_bound_for_2():
    def kernel() -> int32[10]:
        B: int32[10] = 0
        for i in range(10):
            for j in range(i, i + 1):
                B[i] += j
        return B

    s = process(kernel)
    # print(s.module)
    # mod = s.build()
    # np_A = np.zeros((10,), dtype=np.int32)
    # kernel(np_A)
    # np_B = np.zeros((10,), dtype=np.int32)
    # mod(np_B)
    # np.testing.assert_allclose(np_A, np_B)


def test_scf_for():
    def kernel(A: int32[10], B: int32[10]):
        for i in range(10):
            for j in range(A[i], 10, A[i]):
                for k in range(A[i] - 1, A[i] + 2):
                    B[k] += i - j

    s = process(kernel)
    # print(s.module)
    # mod = s.build()
    # np_A = np.zeros((10,), dtype=np.int32) + 1
    # np_B = np.zeros((10,), dtype=np.int32)
    # kernel(np_A, np_B)
    # np_C = np.zeros((10,), dtype=np.int32) + 1
    # np_D = np.zeros((10,), dtype=np.int32)
    # mod(np_C, np_D)
    # np.testing.assert_allclose(np_B, np_D)


if __name__ == "__main__":
    test_vadd()
    # test_range_for()
    # test_variable_bound_for()
    # test_variable_bound_for_2()
    # test_scf_for()
