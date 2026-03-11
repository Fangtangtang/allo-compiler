# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process_spmw
from src.hls import to_hls
from allo.ir.types import int32, float32, Stream, ConstExpr, index
from allo import spmw
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


def test_shard_1D():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[4])
        def core():
            x = spmw.axes()
            local_A = A.shard([x])
            # local_ = A.shard([x])
            local_B: int32[256] = B.shard([x])
            local_B[:] = local_A + 1

    s = process_spmw(top)


# def test_shard_2D():
#     LyA = [S(0), R]
#     M, N = 64, 64

#     @spmw.unit()
#     def top(A: int32[M, N], B: int32[M, N]):
#         @spmw.work(mapping=[4], inputs=[A], outputs=[B])
#         def core(local_A: int32[M, N] @ LyA, local_B: int32[M, N] @ LyA):
#             local_B[:, :] = local_A + 1

#     s = process_spmw(top)

#     @spmw.unit()
#     def top(A: int32[64, 64], B: int32[64, 64]):
#         @spmw.work(mapping=[2, 2], inputs=[A], outputs=[B])
#         def core(
#             local_A: int32[64, 64] @ [S(0), S(1)], local_B: int32[64, 64] @ [S(0), S(1)]
#         ):
#             local_B[:, :] = local_A + 1

#     s = process_spmw(top)


# def test_get_wid_1D_1():
#     @spmw.unit()
#     def top(A: int32[1024], B: int32[1024]):
#         @spmw.work(mapping=[1])
#         def core():
#             for i in range(1024):
#                 B[i] = A[i] + 1

#     mod = to_hls(top)


# def test_get_wid_1D_2():
#     vlen = 1024
#     P = 4
#     tlen = vlen // P

#     @spmw.unit()
#     def top(A: int32[vlen], B: int32[vlen]):
#         @spmw.work(mapping=[P])
#         def core():
#             pi: ConstExpr[index] = spmw.get_wid()
#             for i in range(tlen * pi, tlen * (pi + 1)):
#                 B[i] = A[i] + 1

#     mod = to_hls(top)


# def test_cooperative_gemm():
#     Ty = float32
#     M, N, K = 16, 16, 16
#     P0, P1 = 2, 2
#     Mt, Nt = M // P0, N // P1

#     @spmw.unit()
#     def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
#         pipe: Stream[Ty[Mt, Nt], 2][P0, P1]

#         @spmw.work(mapping=[P0, P1])
#         def gemm0():
#             pi, pj = spmw.get_wid()
#             C_out: Ty[Mt, Nt] = 0
#             for i in range(pi * Mt, (pi + 1) * Mt):
#                 for j in range(pj * Nt, (pj + 1) * Nt):
#                     c: Ty = 0
#                     for k in range(K // 2):
#                         c += A[i, k] * B[k, j]
#                     C_out[i - pi * Mt, j - pj * Nt] = c
#             pipe[pi, pj].put(C_out)

#         @spmw.work(mapping=[P0, P1])
#         def gemm1():
#             pi, pj = spmw.get_wid()
#             C_out: Ty[Mt, Nt] = pipe[pi, pj].get()
#             for i in range(pi * Mt, (pi + 1) * Mt):
#                 for j in range(pj * Nt, (pj + 1) * Nt):
#                     c: Ty = 0
#                     for k in range(K // 2, K):
#                         c += A[i, k] * B[k, j]
#                     C[i, j] = C_out[i - pi * Mt, j - pj * Nt] + c

#     s = process_spmw(top)


if __name__ == "__main__":
    test_shard_1D()
    # test_shard_2D()
    # test_get_wid_1D_1()
    # test_get_wid_1D_2()
    # test_cooperative_gemm()
