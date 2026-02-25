# Passes for Work / Unit Instantiation
## `instantiate_for_hls`

Flatten the SPMW program for the HLS backend.

The current implementation is a non-optimized version. Its primary purpose is to directly interface with Allo's existing HLS backend.

### Overview

The transformation consists of the following steps:

1. Instantiate work instances at grid points

   * Perform instantiation on each grid point.
   * Replace all metadata with constant values for each instance.
   * Unroll all `meta_for` loops.
   * Replace stream array accesses with accesses to concrete stream instances.

2. Construct the top module

   * Create a top module function.
   * Move shared resources (e.g., streams) from global scope into the body of the top module function.

3. Update work instance function signatures

   * Modify the function signature of each work instance.
   * Move shared resources into function parameters.
   * Update all operations that use shared resources accordingly.

4. Call work instances from the top module

   * Insert calls to each instantiated work instance inside the top module.
   * Pass shared resources explicitly as function arguments.

### Important Notes

1. Shared resources are moved under the top module and passed as function parameters to work instances in order to match the expectations of the HLS code generation.

2. The current HLS backend does **not support data sharding**.

3. The top module function must be annotated with the `"dataflow"` attribute. This attribute is used to generate the corresponding `dataflow` pragma in the HLS backend.

4. Be careful with the naming of work instances and the top module.
   The backend code generation may use `startswith`-based matching to identify specific functions.
   Incorrect naming may result in incorrect matching and unintended rewrites.
