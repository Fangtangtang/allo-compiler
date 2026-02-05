# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import process
from allo.ir.types import int16, int32, uint16, bool, float32, Fixed


def test_assign_with_casting():
    def kernel1(a: int16) -> int32:
        b: int32 = a
        return b

    s = process(kernel1)
    assert s(1) == 1
    assert s(1000) == 1000
    assert s(-1) == -1
    assert s(-1000) == -1000
    assert s(32767) == 32767
    assert s(-32768) == -32768

    def kernel2(a: int16):
        b: Fixed(12, 4) = a

    s = process(kernel2)


def test_arith_with_casting():
    def kernel1(a: int16, b: int16) -> int32:
        c: int32 = a + b
        return c

    s = process(kernel1)
    assert s(1, 2) == 3
    assert s(1000, 2000) == 3000
    assert s(-1, -2) == -3
    assert s(-1000, -2000) == -3000
    assert s(32767, 1) == 32768
    assert s(-32768, -1) == -32769


if __name__ == "__main__":
    test_assign_with_casting()
    # test_arith_with_casting()
