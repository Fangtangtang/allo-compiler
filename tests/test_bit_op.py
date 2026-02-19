# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import numpy as np
from src.main import to_hls, process
from allo.ir.types import int32, uint16, Fixed, float32, UFixed
from allo.spmw import kernel


def test_get_bit():
    @kernel
    def kernel1(a: int32) -> int32:
        return a[28:32]

    s = process(kernel1)
    pass


if __name__ == "__main__":
    test_get_bit()
