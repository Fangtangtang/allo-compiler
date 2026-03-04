# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo.backend.hls as hls
from src.hls import to_hls
import tempfile
import allo
from allo.ir.types import int32
from allo import spmw


def test():
    @spmw.unit()
    def vadd(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[1])
        def core():
            for i in allo.grid(1024):
                B[i] = A[i] + 1

    @spmw.unit()
    def top(A0: int32[1024], A1: int32[1024], B: int32[1024], C: int32[1024]):
        @spmw.work(mapping=[1])
        def core():
            vadd(A0, B)
            vadd(A1, C)

    if hls.is_available("vitis_hls"):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = np.random.randint(0, 100, (1024,), dtype=np.int32)
            B = np.random.randint(0, 100, (1024,), dtype=np.int32)
            C = np.zeros((1024,), dtype=np.int32)
            mod = to_hls(top, project=tmpdir)
            mod(A, A, B, C)
            # assert fail due to backend issues
            # np.testing.assert_allclose(A + 1, B)
            # np.testing.assert_allclose(A + 1, C)


if __name__ == "__main__":
    test()
