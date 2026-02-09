# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
import allo
from allo.ir.types import int32, index, uint1, float32


def test_meta_for():
    # [NOTE]: semantic test only, currently cannot unroll the loop
    def kernel(A: int32[20]) -> int32[20]:
        with allo.meta_for(10) as i:
            A[i] = i
        return A

    s = process(kernel)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i] = i
    assert np.allclose(s(np_A), gold)

    def kernel(A: int32[20]) -> int32[20]:
        with allo.meta_for(10) as i:
            A[i + 1] = i
        return A

    s = process(kernel)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i + 1] = i
    assert np.allclose(s(np_A), gold)

    def kernel(A: int32[20]) -> int32[20]:
        with allo.meta_for(10) as i:
            A[i * 2] = i
        return A

    s = process(kernel)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i * 2] = i
    assert np.allclose(s(np_A), gold)

    def kernel(A: int32[20]):
        with allo.meta_for(10) as i:
            A[i] = i
        with allo.meta_for(10, 20) as i:
            A[i] = i
        with allo.meta_for(0, 20, 2) as i:
            A[i] = i * 2

    s = process(kernel)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        gold[i] = i
    for i in range(10, 20):
        gold[i] = i
    for i in range(0, 20, 2):
        gold[i] = i * 2
    s(np_A)
    assert np.allclose(np_A, gold)

    def kernel(A: int32[20]) -> int32[20]:
        with allo.meta_for(10) as i:
            with allo.meta_for(2) as j:
                A[i * 2 + j] = i
        return A

    s = process(kernel)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        for j in range(2):
            gold[i * 2 + j] = i
    assert np.allclose(s(np_A), gold)

    def kernel(A: int32[10]):
        with allo.meta_for(10) as i:
            with allo.meta_for(i + 1, 10) as j:
                with allo.meta_for(j * 2, 10) as k:
                    A[k] += i - j

    s = process(kernel)
    np_A = np.zeros((10,), dtype=np.int32)
    gold = np.zeros((10,), dtype=np.int32)
    for i in range(10):
        for j in range(i + 1, 10):
            for k in range(j * 2, 10):
                gold[k] += i - j
    s(np_A)
    assert np.allclose(np_A, gold)

    def kernel(B: int32[10]):
        with allo.meta_for(10) as i:
            with allo.meta_for(i, i + 1) as j:
                B[i] += j

    s = process(kernel)
    np_B = np.zeros((10,), dtype=np.int32)
    gold = np.zeros((10,), dtype=np.int32)
    for i in range(10):
        for j in range(i, i + 1):
            gold[i] += j
    s(np_B)
    assert np.allclose(np_B, gold)

    print("test_meta_for passed")


def test_meta_if():
    pass


if __name__ == "__main__":
    test_meta_for()
    test_meta_if()
