# allo-compiler
The compiler infra of allo

## Examples
### Assignment
source code
```python
def kernel1() -> int32:
    A: int32 = 0
    B: int32 = 0
    A += B
    return A
```

processed ast
```txt
def kernel1() -> __allo__[i32, (), None]:
    A: __allo__[i32, (), None] = __allo__.constant(0, __allo__[i32, (), None])
    B: __allo__[i32, (), None] = __allo__.constant(0, __allo__[i32, (), None])
    A: __allo__[i32, (), None] = __allo__.Add(A, B, __allo__[i32, (), None])
    return A 
```

AnnAssign, Assign, AugAssign are normalized to AnnAssign, so IR builder handles fewer cases.

Allo IR
```mlir
module {
  func.func @kernel1() -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "A"} : memref<i32>
    affine.store %c0_i32, %alloc[] : memref<i32>
    %c0_i32_0 = arith.constant 0 : i32
    %alloc_1 = memref.alloc() {name = "B"} : memref<i32>
    affine.store %c0_i32_0, %alloc_1[] : memref<i32>
    %0 = affine.load %alloc[] : memref<i32>
    %1 = affine.load %alloc_1[] : memref<i32>
    %2 = arith.addi %0, %1 : i32
    affine.store %2, %alloc[] : memref<i32>
    %3 = affine.load %alloc[] : memref<i32>
    return %3 : i32
  }
}
```
### Broadcasting
source code
```python
def kernel2() -> int32:
    a: int32 = 1
    b: int32[32, 32] = a
    return b[0, 0]
```

processed ast

```txt
def kernel2() -> __allo__[i32, (), None]:
    a: __allo__[i32, (), None] = __allo__.constant(1, __allo__[i32, (), None])
    b: __allo__[i32, (32, 32), None] = __allo__.broadcast(a, (0, 1), __allo__[i32, (32, 32), None])
    return b[0, 0] 
```

Allo IR
```mlir
module {
  func.func @kernel2() -> i32 {
    %c1_i32 = arith.constant 1 : i32
    %alloc = memref.alloc() {name = "a"} : memref<i32>
    affine.store %c1_i32, %alloc[] : memref<i32>
    %0 = affine.load %alloc[] : memref<i32>
    %alloc_0 = memref.alloc() : memref<32x32xi32>
    linalg.fill ins(%0 : i32) outs(%alloc_0 : memref<32x32xi32>)
    %alloc_1 = memref.alloc() {name = "b"} : memref<32x32xi32>
    memref.copy %alloc_0, %alloc_1 : memref<32x32xi32> to memref<32x32xi32>
    %1 = affine.load %alloc_1[0, 0] : memref<32x32xi32>
    return %1 : i32
  }
}
```

### Vector Addition
source code
```python
def vadd(A: float32[32], B: float32[32], C: float32[32]):
    for i in range(32):
        # syntax sugar for lib op "add"
        C[i] = A[i] + B[i]
```

processed ast
```txt
def vadd(A: __allo__[f32, (32,), None], B: __allo__[f32, (32,), None], C: __allo__[f32, (32,), None]):
    for i in range(0, 32, 1):
        C[i]: __allo__[f32, (), None] = __allo__.Add(A[i], B[i], __allo__[f32, (), None])
    return 
```

BinOps only exist on raw AST, the ast processor will transform them into CallOps which call a builtin funciton (e.g., `__allo__.Add` for ast.Add)

Allo IR
```mlir
module {
  func.func @vadd(%arg0: memref<32xf32>, %arg1: memref<32xf32>, %arg2: memref<32xf32>) {
    affine.for %arg3 = 0 to 32 {
      %0 = affine.load %arg0[%arg3] : memref<32xf32>
      %1 = affine.load %arg1[%arg3] : memref<32xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg2[%arg3] : memref<32xf32>
    }
    return
  }
}
```