# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
import allo
from allo.ir.types import int32
from allo.template import meta_for as allo_for


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
        with allo.template.meta_for(10) as i:
            A[i] = i
        with allo_for(10, 20) as i:
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

    def get_constant(value: int):
        return value

    def kernel(A: int32[20]) -> int32[20]:
        with allo.meta_for(get_constant(10)) as i:
            with allo.meta_for(get_constant(2)) as j:
                A[i * 2 + j] = i
        return A

    s = process(kernel)
    np_A = np.zeros((20,), dtype=np.int32)
    gold = np.zeros((20,), dtype=np.int32)
    for i in range(10):
        for j in range(2):
            gold[i * 2 + j] = i
    assert np.allclose(s(np_A), gold)

    print("test_meta_for passed")


def test_meta_if():
    def get_True():
        return True

    def get_False():
        return False

    def kernel(A: int32[10]) -> int32[10]:
        with allo.meta_if(10 > 5):
            A[0] = 1
        with allo.meta_if(10 < 5):
            A[0] = 5
        return A

    s = process(kernel)
    np_A = np.zeros((10,), dtype=np.int32)
    gold = np.zeros((10,), dtype=np.int32)
    gold[0] = 1
    assert np.allclose(s(np_A), gold)

    def kernel(A: int32[10]) -> int32[10]:
        with allo.meta_if(10 > 5):
            A[0] = 1
        with allo.meta_elif(10 > 5):
            A[0] = 2

        with allo.meta_if(10 < 5):
            A[1] = 5
        with allo.meta_elif(True):
            A[1] = 10
        return A

    s = process(kernel)
    np_A = np.zeros((10,), dtype=np.int32)
    gold = np.zeros((10,), dtype=np.int32)
    gold[0] = 1
    gold[1] = 10
    assert np.allclose(s(np_A), gold)

    def kernel(A: int32[10]) -> int32[10]:
        with allo.meta_if(10 > 5):
            A[0] = 1  # hit
        with allo.meta_elif(10 > 5):
            A[0] = 2  # miss
        with allo.meta_else():
            A[0] = 3  # miss

        with allo.meta_if(10 < 5):
            A[1] = 5  # miss
        with allo.meta_elif(True):
            A[1] = 10  # hit
        with allo.meta_else():
            A[1] = 15  # miss
        return A

    s = process(kernel)
    np_A = np.zeros((10,), dtype=np.int32)
    gold = np.zeros((10,), dtype=np.int32)
    gold[0] = 1
    gold[1] = 10
    assert np.allclose(s(np_A), gold)

    def kernel(A: int32[10]) -> int32[10]:
        with allo.meta_if(10 > 5):
            A[0] = 1  # hit
        with allo.meta_elif(10 > 5):
            A[0] = 2  # miss
        with allo.meta_else():
            A[0] = 3  # miss

        with allo.meta_if(10 < 5):
            A[1] = 5  # miss
        with allo.meta_elif(False):
            A[1] = 10  # miss
        with allo.meta_else():
            A[1] = 15  # hit
        return A

    s = process(kernel)
    np_A = np.zeros((10,), dtype=np.int32)
    gold = np.zeros((10,), dtype=np.int32)
    gold[0] = 1
    gold[1] = 15
    assert np.allclose(s(np_A), gold)

    def kernel(A: int32[10]) -> int32[10]:
        with allo.meta_if(get_True()):
            A[0] = 1  # hit
        with allo.meta_elif(get_True()):
            A[0] = 2  # miss
        with allo.meta_else():
            A[0] = 3  # miss

        with allo.meta_if(get_False()):
            A[1] = 5  # miss
        with allo.meta_elif(get_False()):
            A[1] = 10  # miss
        with allo.meta_else():
            A[1] = 15  # hit
        return A

    s = process(kernel)
    np_A = np.zeros((10,), dtype=np.int32)
    gold = np.zeros((10,), dtype=np.int32)
    gold[0] = 1
    gold[1] = 15
    assert np.allclose(s(np_A), gold)


if __name__ == "__main__":
    test_meta_for()
    test_meta_if()
