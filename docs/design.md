# 3D Copy Lib: Design Notes

Status of the design discussion for porting the 2D copy library (`old/gpu2dcopylib`) to 3D, and the next steps.

## Current state

- The API skeleton compiles: `include/copylib/*.hpp` holds the declarations, `src/*.cpp` the definitions with empty bodies. `src/utils.cpp` is ported as-is.
- `CMakeLists.txt` defines a single `copylib::copylib` target intended for `add_subdirectory` vendoring.
- The skeleton does not reflect all decisions below yet (see [Upcoming steps](#upcoming-steps)).

## Decisions

### Data layout

A layout describes a box inside a 3D allocation. `d0` is the innermost dimension and always contiguous, `d2` the outermost. `d0` counts bytes, `d1` rows and `d2` planes.

| Field | Meaning | Unit |
| --- | --- | --- |
| `base` / `staging` | allocation base pointer, or a placeholder for an unplaced staging buffer | |
| `d0_stride` | one full row of the allocation | bytes |
| `d1_stride` | one full plane of the allocation | rows |
| `d0_start_offset`, `d0_end_offset` | start and end of the box within a row | bytes |
| `d1_start_offset`, `d1_end_offset` | start and end of the box within a plane | rows |
| `d2_start_offset`, `d2_end_offset` | start and end of the box within the allocation | planes |

The outermost extent of the allocation is never needed, so there is no `d2_stride`.

Derived quantities:

```
row extent  = d0_end_offset - d0_start_offset
rows        = d1_end_offset - d1_start_offset
planes      = d2_end_offset - d2_start_offset
plane bytes = d1_stride * d0_stride
total bytes = row extent * rows * planes
first byte  = base + d2_start_offset * plane bytes + d1_start_offset * d0_stride + d0_start_offset
```

Example: the box `[2,5) x [3,7) x [4,10)` of `int[12,16,20]` has `d0_stride = 80`, `d1_stride = 16`, `d0: [16,40)`, `d1: [3,7)`, `d2: [2,5)`. That is 24 bytes x 4 rows x 3 planes = 288 bytes, starting at byte 2816.

Counting rows and planes rather than bytes keeps the three divisibility rules a byte-based encoding would need out of `is_valid`, and makes the bounds below the only constraints.

A layout is valid if:

- `d0_stride` and `d1_stride` are non-zero.
- The box fits, which also makes the encoding canonical: `0 <= d0_start < d0_end <= d0_stride` and `0 <= d1_start < d1_end <= d1_stride`. Otherwise one box has several encodings, which breaks `operator==`, hashing and plan chaining.
- `0 <= d2_start < d2_end`. The allocation's plane count is unknown, so there is no upper bound.
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

### Normal form and equality

`operator==` and hashing compare the encoding field by field, window included. The canonical-encoding rule only makes
the encoding unique for fixed strides, so the same bytes can still be encoded differently, e.g. a single row inside a
large allocation versus the same bytes built with the 1D constructor. Comparing bytes therefore means normalizing first,
which requires `normalize` to produce a **unique** encoding for each box, not just a collapsed one.

The normal form is defined by the box's maximal contiguous runs, which depend only on the bytes, not on the allocation.
With `o` the offset of the first run, `L` the run length, and all runs of a box having the same length:

| Runs | Form | Encoding |
| --- | --- | --- |
| one | 1D | `d0_stride = o + L`, `d1_stride = 1`, `d0: [o, o + L)`, `d1: [0, 1)`, `d2: [0, 1)` |
| `n` evenly spaced by `s` bytes | 2D | `d0_stride = s`, `d0: [o % s, o % s + L)`, `d1: [o / s, o / s + n)`, `d1_stride = o / s + n`, `d2: [0, 1)` |
| a two-level grid, planes `t` bytes apart | 3D | `d0_stride = s`, `d1_stride = t / s`, `d0: [o % s, o % s + L)`, `d1: [(o % t) / s, (o % t) / s + n)`, `d2: [o / t, o / t + m)` |

- The window is unchanged: collapsing keeps the packed order, which is ascending address order.
- The 1D constructor produces exactly the 1D normal form.
- Evenly spaced runs include the case where full-height partial rows continue across a plane boundary with the same
  spacing, so such a layout collapses to 2D even though no two of its rows are adjacent.
- A single-plane layout loses its rows-per-plane count, which the bytes do not determine.
- `normalize(spec)` normalizes each side independently. Each side is read in its own packed order, so this preserves the
  copy exactly; the 2D library's "as far as both sides allow" rule existed because fragment structure had to match.

### Chunking edge cases

- A `chunk_size` below the local alignment is raised to the alignment.
- Chunk boundaries sit at multiples of the alignment in absolute packed offsets, not relative to the window start, so
  that every interior boundary stays aligned for the kernels. Only the window's own ends can be unaligned: the window
  `[3, 288)` with alignment 8 and `chunk_size` 64 becomes `[3, 64)`, `[64, 128)`, …, `[256, 288)`.

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

- Chunk boundaries have to be rounded to a local alignment: the largest power of two up to 64 that divides the row extents and `d0_stride` of both sides (the plane size is a multiple of `d0_stride`, so it adds nothing). It keeps kernels on wide element types and is computed per copy, not stored. `chunk_size` must be at least this alignment.
- The kernel's `int32` index path has to check the full index span of the window on both sides, not individual fields.
- Contiguity (`is_window_contiguous` on `data_layout`) is defined over the window. A window inside one row is a single `queue.copy`. There is no spec-level predicate; the backend asks both layouts.
- The general 3D kernel needs two divmods per side per element, which is what `data_layout::offset_at()` computes. Mitigations: special-case 1D/2D boxes, or split same-extent copies into box-shaped pieces run as `nd_range<2>`/`<3>` kernels without division. Which of those applies is a property of the copy, not of one layout, so it is reduced from both sides at once (as Celerity does in `layout_nd_copy`) rather than reported per layout.
- Overlap checks for copies within one allocation can only be conservative.
- Direct d2d copies with host staging pack the staging layout. Staging buffers are always 1D, so their size is the window length.
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
2. ~~Update the skeleton.~~ Done: the window is in `data_layout` and its constructors, the 1D constructors take a
   length instead of a `fragment_length`, the native 2D/3D copy API and `COPYLIB_USE_CUDA` are gone, and the README
   table and example are updated. The accessors were reworked by dropping rather than defining them:
   - `dimensions()` is gone. How many strides a copy needs is a property of the pair of layouts, not of one of them,
     so it is reduced from both sides at once when the copy is executed.
   - `total_extent()` is gone, subsumed by `end_offset()`. Its only use was sizing staging buffers, which are now
     always 1D and packed, so their size is the window length.
   - `layer_count()`, `layer_offset()` and `fragment_offset()` are gone. A window may start or end mid-row, so
     per-fragment indexing cannot describe what a copy transfers. They are replaced by `offset_at()`, the closed form
     for a single byte, and by `for_each_contiguous_run()`, the iteration primitive over a window.
   - `unit_stride()` is renamed to `is_window_contiguous()`, which is what it means once it is defined over the window.
3. Write tests first, ahead of the implementation. In progress, see [testing.md](testing.md) for the layered plan and
   for how progress is reported in CI.
4. Implement the core: the layout accessors, `is_valid`, `normalize`, `apply_chunking`, `apply_staging`, `apply_d2d_implementation` and `manifest_strategy`. `is_equivalent` was dropped: the 2D implementation was of little use, and the tests check plans against their own reference model instead.
5. Implement the backend: staging fulfiller with correct alignment, the `execute_copy` paths (host `memcpy`, contiguous copy, merged 1D runs), and the kernels (`int32` span check, special cases).
6. Add strategy selection and port the benchmarks. Compare against the 2D library, especially strided copies involving the host.
