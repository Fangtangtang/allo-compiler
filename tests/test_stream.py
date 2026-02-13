# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process_spmw
import allo
from allo.ir.types import int32, Stream
from allo import spmw
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


def test_scalar_stream():
    @spmw.unit()
    def top1(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32]

        @spmw.work(mapping=[1], inputs=[A])
        def producer(local_A: int32[16, 16]):
            pipe.put(local_A[0, 0])

        @spmw.work(mapping=[1], outputs=[B])
        def consumer(local_B: int32[16, 16]):
            local_B[0, 0] = pipe.get()

    s = process_spmw(top1)

    @spmw.unit()
    def top2(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32]

        @spmw.work(mapping=[1], inputs=[A])
        def producer(local_A: int32[16, 16]):
            for i, j in allo.grid(16, 16):
                pipe.put(local_A[i, j])

        @spmw.work(mapping=[1], outputs=[B])
        def consumer(local_B: int32[16, 16]):
            for i, j in allo.grid(16, 16):
                local_B[i, j] = pipe.get()

    s = process_spmw(top2)


def test_tensor_stream():
    @spmw.unit()
    def top(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32[16, 16]]

        @spmw.work(mapping=[1], inputs=[A])
        def producer(local_A: int32[16, 16]):
            pipe.put(local_A)

        @spmw.work(mapping=[1], outputs=[B])
        def consumer(local_B: int32[16, 16]):
            local_B[:, :] = pipe.get()

    s = process_spmw(top)


def test_stream_array():
    @spmw.unit()
    def top(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32[4, 16]][4]

        @spmw.work(mapping=[4], inputs=[A])
        def producer(local_A: int32[16, 16] @ [S(0), R]):
            pi = spmw.get_wid()
            pipe[pi].put(local_A)

        @spmw.work(mapping=[4], outputs=[B])
        def consumer(local_B: int32[16, 16] @ [S(0), R]):
            pi = spmw.get_wid()
            local_B[:, :] = pipe[pi].get()

    s = process_spmw(top)

    @spmw.unit()
    def top2(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32][16, 16]

        @spmw.work(mapping=[1], inputs=[A])
        def producer(local_A: int32[16, 16]):
            pi = spmw.get_wid()
            for i, j in allo.grid(16, 16):
                pipe[i + pi, j].put(local_A[i, j])

        @spmw.work(mapping=[1], outputs=[B])
        def consumer(local_B: int32[16, 16]):
            pi = spmw.get_wid()
            for i, j in allo.grid(16, 16):
                local_B[i, j] = pipe[i + pi, j].get()

    s = process_spmw(top2)


if __name__ == "__main__":
    test_scalar_stream()
    test_tensor_stream()
    test_stream_array()
