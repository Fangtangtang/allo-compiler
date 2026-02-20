# Frontend Syntax

This document provides a comprehensive reference for the Allo frontend syntax and semantics.
Allo uses a Python-based domain-specific language (DSL) that requires **strict type annotations** to enable hardware synthesis and optimization.

## Program Structure
An Allo program consists of [**functions**](#function-definition) annotated with specific decorators. Currently supported decorators include:

* `@kernel`: Defines a basic Allo kernel.
* 🔀`@unit()`: Defines an SPMW module.
* 🔀`@work(mapping: list[int], inputs=None, outputs=None)`: Defines a work within an SPMW module. 
    * A function annotated with `@work` must be declared inside a `@unit` function and cannot exist independently.

## Data Types

### Base Data Types
Currently, Allo supports four base data types.

#### `Index`
An array index.

#### Integers
* `Int`: An integer of variable bitwidth. `Int(bits)`
* `UInt`: An unsigned integer of variable bitwidth. `UInt(bits)`

#### Floating points
A floating point decimal number.

| Format        | Bits | Exponent | Fraction |
|---------------|------|----------|----------|
| float64(FP64) | 64   | 11       | 52       |
| float32(FP32) | 32   | 8        | 23       |
| float16(FP16) | 16   | 5        | 10       |
| bfloat16(BF16)| 16   | 8        | 7        |


#### Fixed points
* `Fixed`: A fixed point decimal. `Fixed(bits, fracs)`
* `UFixed`: An unsigned fixed point decimal. `UFixed(bits, fracs)`

### Derived Types
Derived types are constructed from base data types. They extend base types with additional structural or semantic properties.

#### Tensor Types
A tensor type is constructed by attaching [**compile-time**](#compile-time-constants) shape information
to a [base data type](#base-data-types) using bracket notation:

```txt
BaseType[Dim0, Dim1, ..., DimN]
```

A tensor type represents a multi-dimensional array whose element type
is the given base data type.

For example:
```python
int32[32, 32]
```
denotes a 2-dimensional tensor of shape `(32, 32)` whose element type is `int32`.

#### `ConstExpr`
`ConstExpr[T]` is a derived type that denotes a [compile-time constant](#compile-time-constants) of base type `T`:

```txt
ConstExpr[BaseType]
```

A `ConstExpr` variable must:
* Be parameterized by a [base data type](#base-data-types).
* Be initialized at the point of declaration. Uninitialized `ConstExpr` declarations are illegal.

For example:
```python
base: ConstExpr[int32] = 2
mult: ConstExpr[int32] = 3
offset: ConstExpr[int32] = base * mult  # Computed at compile time: 6
```

#### 🔀`Stream`
`Stream[T, D]` declares a point-to-point FIFO channel for communication:

```txt
Stream[ElementType, Depth]
```
* `ElementType` is the type of elements carried by the stream. It can be either a [base data type](#base-data-types) or a [tensor type](#tensor-types).
* `Depth` (optional) is the FIFO depth of the stream. If omitted, the default depth is 2.

For example:

```python
# Basic scalar stream
pipe: Stream[float32, 4]  # Stream of float32 with depth 4
# Stream of unsigned integers
stream: Stream[UInt(16), 4]
# Default depth (2)
default_pipe: Stream[int32] # Stream of int32 with depth 2
```

##### Stream Arrays
A stream array is constructed by attaching shape information to a stream type using bracket notation:
```txt
Stream[T, D][Dim0, Dim1, ..., DimN]
```
This declares a multi-dimensional array of independent FIFO channels.

For example:
```python
# 2D array of streams
fifo_A: Stream[float32, 4][4, 4]
```
### Refinement Types
Refinement types extend an existing type, including [base type](#base-data-types) and [derived type](#derived-types), by attaching additional attributes using the following syntax:
```txt
Type @ Refinement
```

#### 🔀Tensor Layouts

### Type Casting

* [Implicit Casting](https://cornell-zhang.github.io/allo/gallery/dive_01_data_types.html#implicit-casting)

* [Explicit Casting](https://cornell-zhang.github.io/allo/gallery/dive_01_data_types.html#explicit-casting)

## Function Definition 

### Function Signature
Allo functions are defined as Python functions with explicit type annotations for all arguments and return types.
Arguments and return types can be [scalar](#base-data-types) or [tensor types](#tensor-types). The syntax follows Python's type hint notation:

```python
def kernel(arg1: Type1[Shape1], arg2: ScalarType) -> ReturnType[Shape]:
    # function body
    return result

```
**Example: Scalar Arguments**

```python
def kernel(A: int32) -> int32:
    return A + 1
```

**Example: Matrix Multiplication**

```python
from allo.ir.types import int32

def gemm(A: int32[32, 32], B: int32[32, 32]) -> int32[32, 32]:
    C: int32[32, 32] = 0
    for i, j, k in allo.grid(32, 32, 32):
        C[i, j] += A[i, k] * B[k, j]
    return C

```
#### Argument Semantics
Allo distinguishes between scalar and tensor arguments in terms of passing semantics and mutability:

* Scalar arguments
  * Passed by value.
  * Treated as **read-only** inside the function body.
  * Reassigning a scalar argument inside the function results in undefined behavior.

* Tensor arguments
  * Passed by reference.
  * Can be both **read and written** inside the function.
  * Modifications to tensor arguments are visible to the caller.

#### Multiple Return Values

Functions can return multiple values as a tuple:

```python
def kernel(A: int32[M], B: int32[M]) -> (int32[M], int32[M]):
    res0: int32[M] = 0
    res1: int32[M] = 0
    for i in range(M):
        res0[i] = A[i] + 1
        res1[i] = B[i] + 1
    return res0, res1

```
The caller can unpack the returned tuple:

```python
C, D = callee(A[i], B[i])

```
To ignore certain return values, use underscore:

```python
C, _ = callee(A[0], B[0])  # Ignore second return value


```
#### No Return Value

Functions that don't return a value can omit the return type annotation, use ``-> None``,
or have an empty return:

```python
def kernel(A: int32[32]):
    pass  # No return

def kernel(A: int32[32]) -> None:
    return

def kernel(A: int32[32]):
    return None
```

## Compile-time Constants
// TODO: different levels

