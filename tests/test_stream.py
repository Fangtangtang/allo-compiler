# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process_spmw
from allo.ir.types import int32, Stream
from allo import spmw
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


def test_stream():
    @spmw.unit()
    def top(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32[16, 16]]

        @spmw.work(mapping=[1], inputs=[A])
        def producer(local_A: int32[16, 16]):
            pipe.put(local_A)

        @spmw.work(mapping=[1], outputs=[B])
        def consumer(local_B: int32[16, 16]): ...

        # local_B[:, :] = pipe.get()

    s = process_spmw(top)


if __name__ == "__main__":
    test_stream()
