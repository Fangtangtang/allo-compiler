# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
from allo.ir.types import int32, float32, Int, ConstExpr


def test_basic_call():
    def helper_func(x: int32) -> int32:
        return x * 2

    def helper_func2(x: int32) -> int32:
        return helper_func(x) * 2

    def helper_func3(x: int32) -> int32:
        return helper_func(x) * helper_func(x)

    def kernel1(x: int32) -> int32:
        ret: int32 = helper_func(x)
        return ret

    s = process(kernel1)
    assert s(2) == 4
    assert s(3) == 6
    assert s(-4) == -8

    def kernel2(x: int32) -> int32:
        ret: int32 = helper_func2(x)
        return ret

    s = process(kernel2)
    assert s(2) == 8
    assert s(3) == 12
    assert s(-4) == -16

    def kernel3(x: int32) -> int32:
        ret: int32 = helper_func3(x)
        return ret

    s = process(kernel3)
    assert s(2) == 16
    assert s(3) == 36
    assert s(-4) == 64

    print("test_basic_call passed")


if __name__ == "__main__":
    test_basic_call()
