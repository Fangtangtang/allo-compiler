# Frontend Syntax

This document provides a comprehensive reference for the Allo frontend syntax and semantics.
Allo uses a Python-based domain-specific language (DSL) that requires **strict type annotations** to enable hardware synthesis and optimization.

*🔀 for dataflow programming

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

#### Type Casting

* [Implicit Casting](https://cornell-zhang.github.io/allo/gallery/dive_01_data_types.html#implicit-casting)

* [Explicit Casting](https://cornell-zhang.github.io/allo/gallery/dive_01_data_types.html#explicit-casting)

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

<!-- TODO -->

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
## Variables
Different from native Python, Allo requires the program to be strongly and statically typed.
Variables are declared using Python's type annotation syntax:
```txt
var: Type = init_value
```
* `Type` can be [base type](#base-data-types) or [derived type](#derived-types) with supported [efinement type](#refinement-types).
* `init_value` is optional. 

For example:
```python
# Declaration with initialization
x: int32 = 0
# Declaration without initialization
y: int32
# Assignment after declaration
z = x + y
# Declaration for 1D tensor
A: int32[10] = 0
```

### Assignment Semantics

[Assignments](#assignment-statements) in Allo follow **value-copy semantics**.

An assignment copies the value of the right-hand side expression into the target variable. It **does not** transfer object ownership or create reference aliases.

For example:
```python
temp: int32[4, 4] = 0
outp: int32[4, 4] = temp  # Copy temp to outp

# Copy from argument
def kernel(inp: int32[8, 8]) -> int32[8, 8]:
    outp: int32[8, 8] = inp
    return outp
```

Type Compatibility:
- The value being assigned must be type-compatible with the target variable.
- For [tensor types](#tensor-types), broadcasting may be implicitly performed if required and valid.
   - For example:
        ```python
        A: int32[4] = 0     # Scalar broadcast to tensor
        ```

- For [base types](#base-data-types), implicit casting is allowed when it is safe and well-defined.
   - For example:
        ```python
        x: int32 = 0
        y: float32 = x      # Implicit cast if allowed
        ```
- `ConstExpr`s are **immutable** and cannot be reassigned.

### Variable Scoping
Allo enforces C++-style **Block Scoping** rules, which differs from standard Python.

*   **Scope Boundaries**: branch, loop body.
*   **Rule**: A variable declared for the first time inside a block is **local** to that block. It is not visible after the block exits. Inner blocks can read/write variables defined in outer blocks.

Variables should be declared in the scope where they are used or parenting scopes:
```python
def kernel(a: int32) -> int32:
    r: int32 = 0  # Declare outside conditional
    if a == 0:
        r = 1
    else:
        r = 4
    return r
```

Local variables are accessible within its scope:
```python
def kernel(a: int32) -> int32:
    r: int32 = 0
    if a > 0:
        t: int32 = 1  # Local to if-branch
        r = r + t
    return r
```

Allo enforces strict symbol uniqueness within a scope.
Declaring a variable with the same name more than once in the same scope results in a symbol conflict.

```python
r: int32 = 0
# annotated assignment without init_value is taken as varibale declaration
r: int32 # Error: symbol conflict
```

#### Invalid Scoping
The following patterns will raise errors:

**Using variables outside the loop:**

```python
# ERROR: tmp is not accessible outside loop
def kernel(n: int32) -> int32:
    for i in range(n):
        tmp: int32 = i
    return tmp  # Error: tmp not in scope

# ERROR: r is not accessible outside branches
def kernel(a: int32) -> int32:
    if a == 0:
        r: int32 = 1
    else:
        r: int32 = 4
    return r  # Error: r not in scope
```

**Redefining loop variables in nested loops:**

```python
# ERROR: Cannot redefine i in nested loop
def kernel(n: int32) -> int32:
    s: int32 = 0
    for i in range(n):
        for i in range(n):  # Error: i already defined
            s = s + i
    return s
```

## Expressions
### Arithmetic Expressions
#### Unary Arithmetic Operators
| Operator | Description     | Example |
|----------|-----------------|---------|
| ``+``    | Unary plus      |   `+a`  |
| ``-``    | Unary negation  |   `-a`  |

#### Binary Arithmetic Expressions
| Operator | Description     | Example |
|----------|-----------------|---------|
| ``+``    | Addition        |`a + b`  |
| ``-``    | Subtraction     | `a - b` |
| ``*``    | Multiplication  | `a * b` |
| ``/``    | Division (float)| `a / b` |
| ``//``   | Floor division  | `a // b`|
| ``%``    | Modulo          | `a % b` |

### Logic Expressions
#### Logical Operators
| Operator | Description | Example   |
| -------- | ----------- | --------- |
| `and`    | Logical AND | `a and b` |
| `or`     | Logical OR  | `a or b`  |
| `not`    | Logical NOT | `not a`   |

#### Comparison Expressions
All standard comparison operators are supported: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``.

**Restriction**: Chained comparisons are currently not supported.


### Call Expressions

Allo adopts Python's function call syntax:
```python
callee(arg0, arg1, ..., argN)
```

Only positional arguments are supported. Keyword arguments, default arguments, and variadic arguments (`*args`, `**kwargs`) are not supported.

- The number of arguments must exactly match the function signature.
- Each argument must be type-compatible with its corresponding parameter.
- Implicit casting (for base types) and broadcasting (for tensor types)
  may be performed if the conversion is valid under the type rules.

## Statements
### Assignment Statements
#### Assignment
The syntax of assignment in Allo follows standard Python syntax:
```python
target = expression
```

Multiple assignment and tuple unpacking are also supported:
```python
a, b = 1, 2
x, y = foo()
```
**Restriction**: assignments where the left-hand side and right-hand side refer to overlapping variables are currently not supported.
```python
a, b = b, a   # Not supported
```

#### Augmented Assignment
Augmented assignment with operators supported in [binary arithmetic](#binary-arithmetic-expressions) work on both scalars and tensor elements.

#### Annotated Assignment
Annotated Assignment without right-hand side expression is taken as varibale declaration. Otherwise, if the target does not exist in current [scope](#variable-scoping), it's taken as declareation and initialization with right-hand side expression; if it exists, it's taken as reassignment.

A variable may be reassigned using type annotation syntax, provided that the annotated type matches the original declared type. The annotated type must be identical to the original declaration type.

For example:
```python
r: int32 = 0
# annotated assignment with init_value is taken as annotated reassignment
r: int32 = 2
```

### Meta-Programming
#### Meta For (Compile-Time Loop Unrolling)

## Built-in Functions
### 🔀Meta Functions

## Others
### Templates

### Compile-Time Constants
<!-- TODO: different levels -->

