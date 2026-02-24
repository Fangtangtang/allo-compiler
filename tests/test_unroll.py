# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from src.aie import parse, to_aie, to_hls
import allo
from allo.ir.types import int32, Stream
from allo import spmw
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


def test_shard_1D_1():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(local_A: int32[1024] @ [S(0)], local_B: int32[1024] @ [S(0)]):
            local_B[:] = local_A

    s = to_aie(top)
    np_A = np.random.randint(0, 100, (1024,), dtype=np.int32)
    np_B = np.zeros((1024,), dtype=np.int32)
    s(np_A, np_B)
    assert np.array_equal(np_A, np_B)


def test_shard_1D_2():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(local_A: int32[1024] @ [S(0)], local_B: int32[1024] @ [S(0)]):
            local_B[:] = local_A + 1

    s = to_aie(top)
    np_A = np.random.randint(0, 100, (1024,), dtype=np.int32)
    np_B = np.zeros((1024,), dtype=np.int32)
    s(np_A, np_B)
    assert np.array_equal(np_A + 1, np_B)


def test_shard_1D_3():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(local_A: int32[1024] @ [S(0)], local_B: int32[1024] @ [S(0)]):
            pi = spmw.get_wid()
            local_B[:] = local_A + pi

    s = to_aie(top)
    np_A = np.random.randint(0, 100, (1024,), dtype=np.int32)
    np_B = np.zeros((1024,), dtype=np.int32)
    gold = np_A.copy()
    for i in range(4):
        gold[256 * i : 256 * (i + 1)] += i
    s(np_A, np_B)
    assert np.array_equal(gold, np_B)


def test_shard_1D_4():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[4], inputs=[A], outputs=[B])
        def core(local_A: int32[1024] @ [S(0)], local_B: int32[1024] @ [S(0)]):
            pi = spmw.get_wid()
            if pi > 1:
                local_B[:] = local_A + pi
            else:
                local_B[:] = local_A

    s = to_aie(top)
    np_A = np.random.randint(0, 100, (1024,), dtype=np.int32)
    np_B = np.zeros((1024,), dtype=np.int32)
    gold = np_A.copy()
    for i in range(4):
        if i > 1:
            gold[256 * i : 256 * (i + 1)] += i
    s(np_A, np_B)
    assert np.array_equal(gold, np_B)


def test_scalar_stream_1():
    @spmw.unit()
    def top1(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32]

        @spmw.work(mapping=[1], inputs=[A])
        def producer(local_A: int32[16, 16]):
            pipe.put(local_A[0, 0])

        @spmw.work(mapping=[1], outputs=[B])
        def consumer(local_B: int32[16, 16]):
            local_B[0, 0] = pipe.get()

    # s = parse(top1)
    s = to_hls(top1)


def test_scalar_stream_2():
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

    s = parse(top2)


def test_tensor_stream():
    @spmw.unit()
    def top(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32[16, 16]][4]

        @spmw.work(mapping=[4], inputs=[A])
        def producer(local_A: int32[16, 16]):
            pi = spmw.get_wid()
            pipe[pi].put(local_A)

        @spmw.work(mapping=[4], outputs=[B])
        def consumer(local_B: int32[16, 16]):
            pi = spmw.get_wid()
            local_B[:, :] = pipe[pi].get()

    s = parse(top)


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

    s = parse(top)

    @spmw.unit()
    def top2(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32][16, 16]

        @spmw.work(mapping=[1], inputs=[A])
        def producer(local_A: int32[16, 16]):
            pi = spmw.get_wid()
            with allo.meta_for(16) as i:
                with allo.meta_for(16) as j:
                    pipe[i + pi, j].put(local_A[i, j])

        @spmw.work(mapping=[1], outputs=[B])
        def consumer(local_B: int32[16, 16]):
            pi = spmw.get_wid()
            with allo.meta_for(16) as i:
                with allo.meta_for(16) as j:
                    local_B[i, j] = pipe[i + pi, j].get()

    s = parse(top2)


if __name__ == "__main__":
    # test_shard_1D_1()
    # test_shard_1D_2()
    # test_shard_1D_3()
    # test_shard_1D_4()
    test_scalar_stream_1()
    # test_scalar_stream_2()
    # test_tensor_stream()
    # test_stream_array()
