# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import tempfile
import numpy as np
from src.main import to_hls
from allo.ir.types import int32, int16, uint16, Fixed, float32, UFixed
from allo.spmw import kernel


def test_arith():
    @kernel
    def kernel1() -> int32[10]:
        A: int32[10] = 1
        return A

    with tempfile.TemporaryDirectory() as tmpdir:
        s = to_hls(kernel1, project=tmpdir)
        np_A = np.zeros((10,), dtype=np.int32)
        np_B = np_A + 1
        s(np_A)
        assert np.array_equal(np_A, np_B)


def test_fixed_cast():
    @kernel
    def kernel1(a: float32) -> int32:
        b: Fixed(12, 4) = a
        c: int32 = b
        return c

    with tempfile.TemporaryDirectory() as tmpdir:
        s = to_hls(kernel1, project=tmpdir)
        ret = np.zeros((1,), dtype=np.int32)
        s(-5.0, ret)
        assert ret[0] == -5.0

    @kernel
    def kernel2(a: float32) -> int32:
        b: Fixed(20, 12) = a
        c: int32 = b
        return c

    with tempfile.TemporaryDirectory() as tmpdir:
        s = to_hls(kernel2, project=tmpdir)
        ret = np.zeros((1,), dtype=np.int32)
        s(5.0, ret)
        assert ret[0] == 5.0

    @kernel
    def kernel3(a: float32) -> int32:
        b: UFixed(12, 4) = a
        c: int32 = b
        return c

    with tempfile.TemporaryDirectory() as tmpdir:
        s = to_hls(kernel3, project=tmpdir)
        ret = np.zeros((1,), dtype=np.int32)
        s(-5.0, ret)
        print(ret)
        assert ret[0] > 0

    @kernel
    def kernel4(a: float32) -> int32:
        b: UFixed(20, 12) = a
        c: int32 = b
        return c

    with tempfile.TemporaryDirectory() as tmpdir:
        s = to_hls(kernel4, project=tmpdir)
        ret = np.zeros((1,), dtype=np.int32)
        s(-5.0, ret)
        print(ret)
        assert ret[0] > 0

    @kernel
    def kernel5(a: int32) -> float32:
        b: Fixed(12, 4) = a
        c: float32 = b
        return c

    with tempfile.TemporaryDirectory() as tmpdir:
        s = to_hls(kernel5, project=tmpdir)
        ret = np.zeros((1,), dtype=np.float32)
        s(-5.0, ret)
        print(ret)
        assert math.isclose(ret[0], -5, rel_tol=1e-3, abs_tol=1e-3)

    @kernel
    def kernel6(a: int32) -> float32:
        b: UFixed(20, 12) = a
        c: float32 = b
        return c

    with tempfile.TemporaryDirectory() as tmpdir:
        s = to_hls(kernel6, project=tmpdir)
        ret = np.zeros((1,), dtype=np.float32)
        s(-5.0, ret)
        assert ret[0] > 0

    @kernel
    def kernel9(a: int16) -> uint16:
        b: Fixed(12, 4) = a
        c: uint16 = b
        return c

    with tempfile.TemporaryDirectory() as tmpdir:
        s = to_hls(kernel9, project=tmpdir)
        ret = np.zeros((1,), dtype=np.uint16)
        s(5.0, ret)
        assert ret[0] == 5.0

    @kernel
    def kernel10(a: int16) -> uint16:
        b: UFixed(20, 12) = a
        c: uint16 = b
        return c

    with tempfile.TemporaryDirectory() as tmpdir:
        s = to_hls(kernel10, project=tmpdir)
        ret = np.zeros((1,), dtype=np.uint16)
        s(5.0, ret)
        assert ret[0] == 5.0

    @kernel
    def kernel12(a: int16) -> uint16:
        b: UFixed(20, 12) = a
        b_: UFixed(12, 4) = b
        c: uint16 = b_
        return c

    with tempfile.TemporaryDirectory() as tmpdir:
        s = to_hls(kernel12, project=tmpdir)
        ret = np.zeros((1,), dtype=np.uint16)
        s(5.0, ret)
        assert ret[0] == 5.0

    print("test_fixed_cast passed")


if __name__ == "__main__":
    test_arith()
    test_fixed_cast()
