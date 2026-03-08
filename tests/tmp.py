from src.main import process_spmw
from allo.ir.types import int32, index
from allo import spmw
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


def test_shard_1D():
    @spmw.unit()
    def top(A: int32[1, 1024], B: int32[2, 1024]):
        @spmw.work(grid=[2, 4])
        def core():
            local_A = A.shard([None, 1])  # replicate on the first dimension
            # type annotation is optional
            local_B: int32[1, 256] = B.shard([0, 1])
            local_B[:] = local_A + 1

    @spmw.unit()
    def top(A: int32[1, 1024], B: int32[2, 1024]):
        @spmw.work(grid=[("x", 2), ("y", 4)])
        def core():
            local_A = A.shard([None, "y"])
            local_B: int32[1, 256] = B.shard(["x", "y"])
            local_B[:] = local_A + 1

    @spmw.unit()
    def top(A: int32[1, 1024], B: int32[2, 1024]):
        @spmw.work(grid=[2, 4])
        def core():
            x, y = spmw.axes()  # axes of the work grid
            local_A = A.shard([None, y])
            local_B: int32[1, 256] = B.shard([x, y])
            wid: index = x.id  # coordinate along axis x
            local_B[:] = local_A + wid
