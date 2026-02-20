# SPMW High-Level IR Design Overview

This document describes the design and implementation of the high-level IR for SPMW in Allo. 

## 1. Motivation

<!-- TODO -->

## 2. Architecture

The compilation flow involves several stages of transformation:

1.  [**Preprocessing**](./ir/ast_preprocessor.md): The Python source code is parsed into an AST and canonicalized into a structured representation, performing semantic checking and type inference.
2.  [**IR Generation**](./ir/ir_builder.md): An `IRBuilder` traverses the canonical AST and constructs the SPMW IR using MLIR dialects.
3.  **Analysis & Optimization**: <!-- TODO -->
4.  **Backend Lowering**: <!-- TODO -->


## 3. Design Principles

* **Reuse of MLIR Ecosystem**: The framework leverages existing MLIR dialects and reuses established MLIR infrastructure wherever possible. Reinventing core components (new Dialect operations, new passes) is avoided unless strictly necessary.

* **Modularity**: Different stages of the compiler flow are decoupled into independent submodules. The design aims to:
    * Minimize the propagation of analysis data across stages.
    * Ensure each submodule operates on a well-defined semantic structure.
    * Keep each stage focused on unique, clear responsibility.

    Ideally, adjacent stages should interface through dumped semantic structures and structural parsing rather than tightly coupled in-memory APIs.

    For example: After AST preprocessing, the dumped AST (in a Python-like representation) can be parsed by the IR builder and be directly used to construct the corresponding IR directly.

    This design also allows advanced users to **manually construct intermediate semantic structures**, effectively bypassing certain compilation stages when needed.

* **Extensibility**: The compilation flow is expected to evolve. The base framework should:
    * Expose stable extension interfaces.
    * Provide a single, well-defined entry point for integrating new features.

    New functionality should be introduced through these extension interfaces.
    If this is not feasible, the correct approach is to **refactor the base framework**, rather than introducing ad-hoc patches.

* **Avoid Magic Numbers and Magic Strings**: Parsing or matching based on magic numbers or magic strings is fragile and error-prone. In some cases, such usage may be difficult to avoid (for example, distinguishing signed and unsigned types in MLIR via specific attribute names). When unavoidable:
    * The **purpose** of the magic value must be clearly documented.
    * Its **construction and parsing** rules must be explicitly specified.
