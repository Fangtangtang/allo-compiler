# SPMW High-Level IR Design Overview

This document describes the design and implementation of the high-level IR for SPMW in Allo. 

## 1. Motivation

<span style="color:red">TODO</span>

## 2. Architecture

The compilation flow involves several stages of transformation:

1.  [**Preprocessing**](./ir/ast_preprocessor.md): The Python source code is parsed into an AST and canonicalized into a structured representation, performing semantic checking and type inference.
2.  [**IR Generation**](./ir/ir_builder.md): An `IRBuilder` traverses the canonical AST and constructs the SPMW IR using MLIR dialects.
3.  **Analysis & Optimization**: <span style="color:red">TODO</span>
4.  **Backend Lowering**: <span style="color:red">TODO</span>


## 3. Design Principles

*   **Reuse of MLIR Ecosystem**: Leverages existing MLIR dialects (like `sdy` for sharding) where possible maximize compatibility and reuse existing infrastructure.
*   **Modularity**: 
*   **Extensibility**: 
