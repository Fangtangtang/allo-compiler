# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from src.main import process_spmw, process
from allo.ir.types import int32, Int
from allo import spmw


def test_region():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(mapping=[1], inputs=[A], outputs=[B])
        def core(local_A: int32[1024], local_B: int32[1024]):
            local_B[:] = local_A + 1

    s = process_spmw(top)


"""
func.func @top(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) attributes {dataflow, itypes = "sss"} {
    %t = bufferization.to_tensor %arg0 : memref<1024xi32> to tensor<1024xi32>
    %1 = sdy.manual_computation(%t) in_shardings=[<@mesh, [{"a"}]>] out_shardings=[<@mesh, [{"a"}]>]
        manual_axes={"a"} (%arg3: tensor<1024xi32>) {
        %m0 = bufferization.to_buffer %arg3: memref<1024xi32>
        %m1 = memref.alloc() : memref<1024xi32>
        call @__top__core(%m0 , %m1) : (memref<1024xi32>, memref<1024xi32>) -> ()
        %t1 = bufferization.to_tensor %m1 : memref<1024xi32> to tensor<1024xi32>
        sdy.return %t1 : tensor<1024xi32>
    } : (tensor<1024xi32>) -> (tensor<1024xi32>)
    %m = bufferization.to_buffer %1: memref<1024xi32>
    memref.copy %m, %arg1 : memref<1024xi32> to memref<1024xi32>
}
"""

if __name__ == "__main__":
    test_region()
