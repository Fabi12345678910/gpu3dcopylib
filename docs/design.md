# 3D Copy Lib: Design Notes

Status of the design discussion for porting the 2D copy library (`old/gpu2dcopylib`) to 3D, and the next steps.

## Current state

- The API skeleton compiles: `include/copylib/*.hpp` holds the declarations, `src/*.cpp` the definitions with empty bodies. `src/utils.cpp` is ported as-is.
- `CMakeLists.txt` defines a single `copylib::copylib` target intended for `add_subdirectory` vendoring.
- The skeleton does not reflect all decisions below yet (see [Upcoming steps](#upcoming-steps)).

## Decisions

### Data layout

A layout describes a box inside a 3D allocation. All values are in bytes. `d0` is the innermost dimension and always contiguous, `d2` the outermost.

| Field | Meaning |
| --- | --- |
| `base` / `staging` | allocation base pointer, or a placeholder for an unplaced staging buffer |
| `d0_stride` | size of one full row of the allocation |
| `d1_stride` | size of one full plane of the allocation |
| `dN_start_offset`, `dN_end_offset` | start and end of the box along dimension N, relative to the allocation |

The outermost extent of the allocation is never needed, so there is no `d2_stride`.

Derived quantities:

```
row extent  = d0_end_offset - d0_start_offset
rows        = (d1_end_offset - d1_start_offset) / d0_stride
planes      = (d2_end_offset - d2_start_offset) / d1_stride
total bytes = row extent * rows * planes
first byte  = base + d0_start_offset + d1_start_offset + d2_start_offset
```

Example: the box `[2,5) x [3,7) x [4,10)` of `int[12,16,20]` has `d0_stride = 80`, `d1_stride = 1280`, `d0: [16,40)`, `d1: [240,560)`, `d2: [2560,6400)`. That is 24 bytes x 4 rows x 3 planes = 288 bytes, starting at byte 2816.

A layout is valid if:

- `d0_stride` and `d1_stride` are non-zero, and `d1_stride` is a multiple of `d0_stride`.
- The `d1` offsets are multiples of `d0_stride`, the `d2` offsets multiples of `d1_stride`.
- The box fits: `d0_end - d0_start <= d0_stride`, `d1_end - d1_start <= d1_stride`.
- The encoding is canonical: `d0_start < d0_stride`, `d1_start < d1_stride`. Otherwise one box has several encodings, which breaks `operator==`, hashing and plan chaining.
- `base` is at least 2-byte aligned, since staging placeholders are detected by their lowest byte.

### Byte window

Each layout carries a half-open window `[start, end)` of byte offsets into the **packed** box (gaps excluded), with `0 <= start < end <= total_bytes`.

- Constructors set the full box explicitly. There is no "0 means everything" shorthand.
- A copy spec is valid if both windows have the same length. Source and target may have different shapes, so reshape copies such as `int[1,1,6]` to `int[1,2,3]` are allowed.
- Chunking keeps the box and only narrows the window. A staged chunk gets the window `[0, end - start)` in its staging buffer.
- There is no `fragment_length`. Windows are counted in bytes rather than 3D indices or fragment indices because:
  - byte offsets match staging buffer offsets and `chunk_size` directly;
  - reshape copies use the same numbers on both sides;
  - a byte offset has exactly one encoding, while a 3D end position is ambiguous at row and plane boundaries.

### Native 2D/3D copies are dropped

The following go away: `use_2D_copy`, `use_3D_copy`, `is_2d_copy_available()`, `is_3d_copy_available()`, `possibility::needs_2d_copy`/`needs_3d_copy`, the 2D staging layout, and the CUDA dependency (`COPYLIB_USE_CUDA`). Staging buffers are always 1D. `copy_properties` keeps only `use_kernel` and stays an enum for future flags.

## Pipeline recap

`manifest_strategy` = `apply_chunking` -> `apply_staging` -> `apply_d2d_implementation`

| Step | Input | Output |
| --- | --- | --- |
| `apply_chunking` | one spec | `parallel_copy_set` of independent single-spec plans, each at most `chunk_size` bytes. `chunk_size == 0` means no chunking. |
| `apply_staging` | each single-spec plan | a sequential plan: `[gather, staging -> staging, scatter]`, or a subset of it. Only the middle copy crosses devices, and it is always contiguous. |
| `apply_d2d_implementation` | each plan | device-to-device copies optionally routed through host staging buffers |

Why stage at all: the gather/scatter kernels are cheap because they run in parallel inside one device's memory. Crossing the bus is expensive per call, so each chunk should cross in one large contiguous copy.

## Remaining restrictions and implementation notes

- Chunk boundaries have to be rounded to a local alignment: the largest power of two up to 64 that divides both row extents and all strides. It keeps kernels on wide element types and is computed per copy, not stored. `chunk_size` must be at least this alignment.
- The kernel's `int32` index path has to check the full index span of the window on both sides, not individual fields.
- Contiguity (`is_contiguous`, `unit_stride`) should be defined over the window. A window inside one row is a single `queue.copy`.
- The general 3D kernel needs two divmods per side per element. Mitigations: special-case 1D/2D boxes, or split same-extent copies into box-shaped pieces run as `nd_range<2>`/`<3>` kernels without division.
- Overlap checks for copies within one allocation can only be conservative.
- Direct d2d copies with host staging must pack the staging layout or size it by `total_extent()`.
- Strided copies involving the host become `memcpy` loops without native 2D copies. Benchmark against the 2D library before accepting this.

## Open decisions

These decide the public API and should be settled before implementing the backend:

1. **Ownership.** Should the executor take the caller's devices and queues instead of creating its own? The SimSYCL system configuration, CPU affinity changes and benchmark data buffers should move out of the library.
2. **Blocking vs. async execution.** Currently `execute_copy` blocks, and plan steps wait on each other with `wait_and_throw()`. Alternative: chain steps with `sycl::event`s and return an event per plan. This determines the `execute_copy` return types. The thread pool is also a function-local `static` sized on the first call.
3. **Error handling.** `COPYLIB_ENSURE` calls `std::exit(1)`. Proposal: invalid input from the caller throws; internal invariants are checked in debug builds only.

These can be settled during implementation:

4. **Staging alignment bug** in the 2D fulfiller (`size + alignment % size` does not round up). Fix it with proper round-up alignment.
5. **Staging memory reuse.** All staging buffers of a set currently have to fit in `buffer_size` at once.
6. **Strategy selection.** Should the library offer `select_strategy(spec, exec)` with benchmark-based thresholds?
7. **Testing approach.** Property tests on SimSYCL comparing against a byte-by-byte reference copy.

## Upcoming steps

1. Settle open decisions 1-3.
2. Update the skeleton:
   - add the byte window to `data_layout` and its constructors;
   - remove the native 2D/3D copy API and `COPYLIB_USE_CUDA`;
   - rework the accessors (`fragment_offset`, `layer_offset`, `contiguous_*`, `dimensions`; maybe `dN` naming);
   - update the README table and example.
3. Write tests first:
   - unit tests for the layout math, using the `int[12,16,20]` examples;
   - a property test: random shapes, boxes, windows and chunk sizes, manifested, executed on SimSYCL, compared with a reference copy;
   - add a `COPYLIB_BUILD_TESTS` option that is only on for top-level builds.
4. Implement the core: layout accessors (move them into the header as `constexpr`), `is_valid`, `normalize`, `apply_chunking`, `apply_staging`, `apply_d2d_implementation`, `manifest_strategy`, and `is_equivalent` as a tiling check.
5. Implement the backend: staging fulfiller with correct alignment, the `execute_copy` paths (host `memcpy`, contiguous copy, merged 1D runs), and the kernels (`int32` span check, special cases).
6. Add strategy selection and port the benchmarks. Compare against the 2D library, especially strided copies involving the host.
