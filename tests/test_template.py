# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
from allo.ir.types import int32, float32, float16


def test_basic_template():
    def kernel[Ty, M, N](A: "Ty[M, N]") -> "Ty[M, N]":
        B: Ty[M, N] = A
        return B

    s = process(kernel, instantiate=[float32, 4, 4])
    np_A = np.random.rand(4, 4).astype(np.float32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A)

    s = process(kernel, instantiate=[int32, 8, 4])
    np_A = np.random.randint(0, 10, size=(8, 4)).astype(np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A)

    def kernel2[Ty, M, N](A: "Ty[M, N]") -> "Ty[M, N]":
        B: Ty[M, N] = A + 1
        return B

    s = process(kernel2, instantiate=[float32, 4, 4])
    np_A = np.random.rand(4, 4).astype(np.float32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A + 1)

    s = process(kernel2, instantiate=[int32, 8, 4])
    np_A = np.random.randint(0, 10, size=(8, 4)).astype(np.int32)
    np_B = s(np_A)
    assert np.array_equal(np_B, np_A + 1)

    def kernel[TyA, TyB, TyC, M, K, N](A: "TyA[M, K]", B: "TyB[K, N]") -> "TyC[M, N]":
        C: TyC[M, N]
        for i in range(M):
            for j in range(N):
                acc: TyC = 0
                for k in range(K):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc
        return C

    s = process(kernel, instantiate=[float32, float32, float32, 32, 16, 32])
    np_A = np.random.randn(32, 16).astype(np.float32)
    np_B = np.random.randn(16, 32).astype(np.float32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    assert np.allclose(np_C, allo_C, rtol=1e-4, atol=1e-4)

    s = process(kernel, instantiate=[int32, int32, int32, 8, 4, 8])
    np_A = np.random.randint(0, 10, size=(8, 4)).astype(np.int32)
    np_B = np.random.randint(0, 10, size=(4, 8)).astype(np.int32)
    np_C = np.matmul(np_A, np_B)
    allo_C = s(np_A, np_B)
    assert np.array_equal(np_C, allo_C)

    print("test_basic_template passed")


if __name__ == "__main__":
    test_basic_template()
