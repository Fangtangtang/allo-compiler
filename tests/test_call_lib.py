# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process
import allo
from allo.ir.types import int32, float32, Int, ConstExpr


def test_call_lib():
    bs = 1
    seq_len = 12
    hidden_size = 768

    def kernel(
        inp: float32[bs, seq_len, hidden_size],
        gamma: float32[hidden_size],
        beta: float32[hidden_size],
    ) -> float32[bs, seq_len, hidden_size]:
        A = allo.layernorm(inp, gamma, beta)
        B = allo.gelu(A)
        return B

    s = process(kernel)
    # allo.customize(kernel)


if __name__ == "__main__":
    test_call_lib()
