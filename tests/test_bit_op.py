# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import numpy as np
from src.main import to_hls, process, build
from allo.ir.types import int32, uint16, Fixed, float32, UFixed
from allo.spmw import kernel


def test_get_bit():
    @kernel
    def kernel1(a: int32) -> int32:
        return a[28:32]

    s = process(kernel1)
    pass

    @kernel
    def kernel2(a: int32) -> int32:
        return a[28 : int32.bits]

    s = process(kernel2)

    @kernel
    def kernel3() -> int32:
        a: int32
        a[28 : int32.bits] = 1
        return a

    s = process(kernel3)


if __name__ == "__main__":
    test_get_bit()
