# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from src.main import process
from allo.ir.types import int32, Int
from allo.memory import Memory, Layout

S = Layout.Shard
R = Layout.Replicate


def test_memory1():
    mm = Memory(resource="URAM")

    def kernel(a: int32[32] @ mm) -> int32[32]:
        b: int32[32]
        for i in range(32):
            b[i] = a[i] + 1
        return b

    s = process(kernel)


def test_layout1():
    LyA = [S(0)]

    def kernel(a: int32[32] @ LyA) -> int32[32]:
        b: int32[32]
        for i in range(32):
            b[i] = a[i] + 1
        return b

    s = process(kernel)


if __name__ == "__main__":
    test_memory1()
    test_layout1()
