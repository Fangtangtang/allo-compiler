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


if __name__ == "__main__":
    test_cast_assign()
