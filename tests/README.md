# Allo Compiler Tests

This directory contains the test suite for the Allo compiler, organized by feature stability and category.

## Directory Structure

### `tests/basic`
Contains verified tests for core language features that are fully supported.

- **Data Types & Memory**:
  - `test_assign.py`: Variable assignment and scalar operations.
  - `test_subscript.py`: Array/Tensor indexing and access.
  - `test_cast_*.py`: Type casting (basic types, binary operations, tensors).
  - `test_np_arr.py`: NumPy array initialization, slicing, and operations.

- **Control Flow**:
  - `test_branch.py`: Conditional statements (if/else).
  - `test_builtin_loop.py`: Allo's built-in loop types (`allo.grid`, `allo.reduction`).
  - `test_for.py`: For loops.
  - `test_while.py`: While loops.

- **Operations**:
  - `test_binary.py`: Binary arithmetic and logical operations.
  - `test_unary.py`: Unary operations.

- **Functions & Modules**:
  - `test_call.py`: Function calls.
  - `test_multi_return.py`: Functions returning multiple values.
  - `test_template.py`: Template functions and parametrization.

- **Meta-Programming**:
  - `test_meta_prog.py`: Meta-programming constructs (`allo.meta_for`, `allo.meta_if`, `allo.meta_elif`, `allo.meta_else`).

### `tests/` (Root)
Contains tests for advanced or experimental features currently under development.

- **Customization**:
  - `test_custom_handler.py`: Custom instruction handlers and behavior overrides.

- ...
## Running Tests

To run the full test suite, use `pytest` from the project root (currently fail):

```bash
pytest tests
```

To run only the verified basic tests:

```bash
pytest tests/basic
```
