# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import numpy as np
import pytest
from src.main import process
from allo.ir.types import bool, int8, int32, float32, index, ConstExpr
import allo.backend.hls as hls
import io
from contextlib import redirect_stdout


def test_range_for():
    def kernel(A: int32[20]):
        for i in range(10):
            A[i] = i
        for i in range(10, 20):
            A[i] = i
        for i in range(0, 20, 2):
            A[i] = i * 2

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
    test_range_for()
    test_variable_bound_for()
    test_variable_bound_for_2()
    test_scf_for()
