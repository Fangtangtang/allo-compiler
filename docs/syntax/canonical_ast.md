# Canonical AST Syntax

The **Canonical AST** is the structured representation of the source code after AST preprocessing. It serves as the canonicalized semantic form used by later compilation stages.

## Core Structure
The canonical AST is organized around a **symbol table**:
* Key: The unique name of a program-level global component (symbol mangling must be applied to guarantee uniqueness).
* Value: The corresponding global component, such as:
  * Function
  * Global variable
  * Global constant

All global components exist at the same level, **functions are not nested** in the canonical representation.

Function templates are fully instantiated during canonicalization.
For each template, the compiler generates the required concrete instances based on actual usage in the program. After instantiation: Only the instantiated function instances remain. **Template definitions themselves are removed** from the canonical AST.


## AST Nodes

The canonical AST mainly contains the following nodes.

### `ast.Name`

### `ast.Constant`

### `ast.Subscript`
`ast.Subscript` represents indexed access expressions in the canonical AST. It is used for:
* Accessing a tensor slice
* Accessing a tensor element
* Accessing specific bits of a scalar
* Any other structured indexing operation

f slicing syntax (`ast.Slice`) is used, all optional parameters are explicitly filled in.

### `ast.BoolOp`

### ⭐`ast.Assign`
Handles several special cases where the assignment shouldn't be canonicalized to `AnnAssign`.

For all this cases, the right-hand side is a Call expression.

#### Meta Function Calls
Meta functions are called only once at the entry of the function to set some meta data. 

The results are meta data with reserved name.

#### Multi-Result Function Calls

If the right-hand side of an assignment is a multi-result function call that requires unpacking, the assignment cannot be safely transformed into an `AnnAssign`.
Therefore, such assignments remain standard assignment statements in the canonical AST.

### `ast.AnnAssign`
All assignment statements (except [the special cases](#astassign)) are canonicalized to `AnnAssign`.

### `ast.For`
Loop structures except `while` loops are transformed to `for` loop.

All optional `range` arguments are explicitly filled in.

The `type_comment` field of `ast.For` is used to encode additional loop semantics.
For example: `"unroll"` indicates that the loop corresponds to a compile-time unrolled loop (e.g., a [`meta.for`](./frontend.md/#meta-for-compile-time-loop-unrolling) construct).

These annotations do not change the structural representation of the loop but provide semantic guidance for later compilation stages.

### `ast.While`

### `ast.If`

### `ast.Return`

### ⭐`ast.Call`
#### Builtin Functions
Binary operations, implicit casting, implicit broadcasting are all transformed into corresponding builtin function calls.

#### General Functions
All function calls in function bodies are also represented as `ast.Call`.

The callee must refer to a **fully resolved symbol **in the global symbol table.

### ⭐`ast.FunctionDef`