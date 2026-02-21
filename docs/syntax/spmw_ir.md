# SPMW IR Syntax

## Used MLIR Dialect
* `arith`: Constant construction and arithmetic operations.
* `affine`: Structured loop construction and affine memory access representation.
* `func`: Function definition and function calls.
* `memref`: Memory representation and management.
* `scf`: Control flow

* `allo`: Out-of-tree custom operations specific to this framework.

* 🔀`sdy`: Tensor layout representation and unit's structure construction.
* 🔀`tensor`: `sdy` operates on `tensor` types. However, this framework primarily uses `memref` for intermediate representation and interoperability.
* 🔀`bufferization`: This enables bridging between `sdy` (tensor-based semantics) and the predominantly `memref`-based lowering pipeline.

## Module Structure

The entire program is represented as a single MLIR module.

Within the module, the top-level entities consist of:
* Global symbols 
* Function definitions

All program components exist at the module level. There is no nesting of functions inside other functions in the module structure.

## Attributes
