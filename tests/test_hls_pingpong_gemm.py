# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import float32, Stream
from allo import spmw
import tempfile
import numpy as np
from src.hls import to_hls


def test_cooperative_gemm():
    Ty = float32
    M, N, K = 16, 16, 16
    P0, P1 = 2, 2
    Mt, Nt = M // P0, N // P1

    @spmw.unit()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[Mt, Nt], 2][P0, P1]

        @spmw.work(mapping=[P0, P1], inputs=[A, B])
        def gemm0(local_A: Ty[M, K], local_B: Ty[K, N]):
            pi, pj = spmw.get_wid()
            C_out: Ty[Mt, Nt] = 0
            for i in range(pi * Mt, (pi + 1) * Mt):
                for j in range(pj * Nt, (pj + 1) * Nt):
                    c: Ty = 0
                    for k in range(K // 2):
                        c += local_A[i, k] * local_B[k, j]
                    C_out[i - pi * Mt, j - pj * Nt] = c
            pipe[pi, pj].put(C_out)

        @spmw.work(mapping=[P0, P1], inputs=[A, B], outputs=[C])
        def gemm1(local_A: Ty[M, K], local_B: Ty[K, N], local_C: Ty[M, N]):
            pi, pj = spmw.get_wid()
            C_out: Ty[Mt, Nt] = pipe[pi, pj].get()
            for i in range(pi * Mt, (pi + 1) * Mt):
                for j in range(pj * Nt, (pj + 1) * Nt):
                    c: Ty = 0
                    for k in range(K // 2, K):
                        c += local_A[i, k] * local_B[k, j]
                    local_C[i, j] = C_out[i - pi * Mt, j - pj * Nt] + c

    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod = to_hls(top, project=tmpdir)
        C = np.zeros((M, N), dtype=np.float32)
        mod(A, B, C)
        np.testing.assert_allclose(C, np.dot(A, B), atol=1e-5)
        print("Passed!")


if __name__ == "__main__":
    test_cooperative_gemm()
