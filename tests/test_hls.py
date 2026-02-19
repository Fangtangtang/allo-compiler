# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.main import to_hls
from allo.ir.types import int32, uint16, bool, float32
from allo.spmw import kernel


def test_arith():
    @kernel
    def kernel1() -> int32[10]:
        A: int32[10] = 1
        return A

    s = to_hls(kernel1)
    np_A = np.zeros((10,), dtype=np.int32)
    np_B = np_A + 1
    s(np_A)
    assert np.array_equal(np_A, np_B)


if __name__ == "__main__":
    test_arith()
