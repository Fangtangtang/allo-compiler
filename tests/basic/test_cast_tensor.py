# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import math
from src.main import process
from allo.ir.types import (
    int16,
    int32,
    uint16,
    uint32,
    float16,
    float32,
    float64,
    bfloat16,
    Fixed,
    index,
    UFixed,
)


def test_cast_assign():
    def kernel1(a: int16) -> uint16[8]:
        b: int16[8] = a
        c: uint16[8] = b
        return c

    s = process(kernel1)
    assert np.array_equal(s(1), np.array([1] * 8, dtype=np.uint16))
    assert np.array_equal(s(2), np.array([2] * 8, dtype=np.uint16))

    def kernel1(a: int16) -> int32[8]:
        b: int16[8] = a
        c: int32[8] = b
        return c

    s = process(kernel1)
    assert np.array_equal(s(1), np.array([1] * 8, dtype=np.int32))
    assert np.array_equal(s(2), np.array([2] * 8, dtype=np.int32))
    assert np.array_equal(s(-1), np.array([-1] * 8, dtype=np.int32))
    assert np.array_equal(s(-2), np.array([-2] * 8, dtype=np.int32))

    def kernel1(a: int16) -> int32[8]:
        b: int16[8] = a
        c: index[8] = b
        return c

    s = process(kernel1)
    assert np.array_equal(s(1), np.array([1] * 8, dtype=np.int32))
    assert np.array_equal(s(2), np.array([2] * 8, dtype=np.int32))

    def kernel1(a: int16) -> float32[8]:
        b: int16[8] = a
        c: float32[8] = b
        return c

    s = process(kernel1)
    assert np.array_equal(s(1), np.array([1.0] * 8, dtype=np.float32))
    assert np.array_equal(s(2), np.array([2.0] * 8, dtype=np.float32))

    def kernel1(a: int16[4, 4]) -> int32[4, 4]:
        return a

    s = process(kernel1)
    np_A = np.random.randint(-100, 100, (4, 4), dtype=np.int16)
    assert np.array_equal(s(np_A), np_A.astype(np.int32))

    def kernel1(a: int16[4, 4]) -> uint32[4, 4]:
        return a

    s = process(kernel1)
    np_A = np.random.randint(0, 100, (4, 4), dtype=np.int16)
    assert np.array_equal(s(np_A), np_A.astype(np.uint32))

    def kernel1(a: int16[4, 4]) -> float32[4, 4]:
        return a

    s = process(kernel1)
    np_A = np.random.randint(-100, 100, (4, 4), dtype=np.int16)
    assert np.array_equal(s(np_A), np_A.astype(np.float32))

    def kernel1(a: float32[4, 4]) -> int32[4, 4]:
        return a

    s = process(kernel1)
    np_A = np.random.rand(4, 4).astype(np.float32)
    assert np.array_equal(s(np_A), np_A.astype(np.int32))

    def kernel2(a: float32[4]) -> uint32[8, 4]:
        b: float32[8, 4] = a
        c: int32[8, 4] = b
        return c

    s = process(kernel2)
    np_A = np.random.rand(4).astype(np.float32)
    np_B = np.array([np_A] * 8, dtype=np.float32)
    assert np.array_equal(s(np_A), np_B.astype(np.int32))

    print("pass test_cast_assign")


def test_cast_fixed():
    # [NOTE]: fixed point not supported for llvm backend
    def kernel1(a: int16) -> uint16[8]:
        b: int16[8] = a
        c: Fixed(12, 4)[8] = b
        return c

    s = process(kernel1)

    def kernel2(a: int16) -> uint16[8]:
        b: int16[8] = a
        c: UFixed(12, 4)[8] = b
        return c

    s = process(kernel2)

    def kernel3(a: int16) -> float16[8]:
        b: int16[8] = a
        c: Fixed(12, 4)[8] = b
        return c

    s = process(kernel3)

    def kernel4(a: int16) -> float32[8]:
        b: int16[8] = a
        c: UFixed(12, 4)[8] = b
        return c

    s = process(kernel4)

    def kernel5(a: float32) -> float32[8]:
        b: float32[8] = a
        c: Fixed(20, 12)[8] = b
        return c

    s = process(kernel5)

    def kernel6(a: float32) -> float32[8]:
        b: float32[8] = a
        c: UFixed(20, 12)[8] = b
        return c

    s = process(kernel6)

    print("pass test_cast_fixed")


def test_cast_arithmetic():
    def kernel1(a: int16[8]) -> uint16[8]:
        b: int16[8] = a
        c: uint16[8] = b + 1
        return c + b

    s = process(kernel1)
    np_A = np.random.randint(0, 100, (8,), dtype=np.int16)
    np_B = np_A.astype(np.uint16)
    np_C = np_B + 1
    np_D = np_C + np_B
    assert np.array_equal(s(np_A), np_D)

    def kernel2(a: int16[4]) -> uint16[4]:
        b: int16[8, 4] = a
        c: uint16[4] = b[0] + b[1]
        d: int32[4] = b[2] + b[3]
        return c + d

    s = process(kernel2)
    np_A = np.random.randint(0, 100, (4,), dtype=np.int16)
    np_B = np.broadcast_to(np_A, (8, 4))
    np_C = np_B[0] + np_B[1]
    np_D = np_B[2] + np_B[3]
    np_E = np_C + np_D
    assert np.array_equal(s(np_A), np_E)

    def kernel3(a: int16[4]) -> float32[4]:
        b: int16[8, 4] = a
        c: float32[4] = b[0] + b[1]
        d: float32[4] = b[2] + b[3]
        return c + d

    s = process(kernel3)
    np_A = np.random.randint(-100, 100, (4,))
    np_B = np.broadcast_to(np_A, (8, 4))
    np_C = np_B[0] + np_B[1]
    np_D = np_B[2] + np_B[3]
    np_E = np_C + np_D
    assert np.allclose(s(np_A), np_E)

    def kernel4(a: int16[4], b: float32[4]) -> float32[4]:
        c: float32[2, 4] = a[0] + b[0]
        d: float32[4] = a[1] + b[1]
        return c[0] + d

    s = process(kernel4)
    np_A = np.random.randint(-100, 100, (4,))
    np_B = np.random.rand(4).astype(np.float32)
    np_C = np.broadcast_to(np_A[0] + np_B[0], (2, 4))
    np_D = np_A[1] + np_B[1]
    np_E = np_C[0] + np_D
    assert np.allclose(s(np_A, np_B), np_E)

    print("pass test_cast_arithmetic")


if __name__ == "__main__":
    test_cast_assign()
    test_cast_fixed()
    test_cast_arithmetic()
