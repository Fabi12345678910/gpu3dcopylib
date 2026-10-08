# 3D Copy Lib: Design Notes

Status of the design discussion for porting the 2D copy library (`old/gpu2dcopylib`) to 3D, and the next steps.

## Current state

- The library is implemented, and every test layer passes: on SimSYCL in CI, and on AdaptiveCpp's CPU backend, see [testing.md](testing.md).
- `CMakeLists.txt` defines a single `copylib::copylib` target intended for `add_subdirectory` vendoring.
- Benchmarks, strategy selection and the Celerity integration come next, see [Upcoming steps](#upcoming-steps).

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

- `d0_stride` and `d1_stride` are positive.
- The box fits, which also makes the encoding canonical: `0 <= d0_start < d0_end <= d0_stride` and `0 <= d1_start < d1_end <= d1_stride`. Otherwise one box has several encodings, which breaks `operator==`, hashing and plan chaining.
- `0 <= d2_start < d2_end`. The allocation's plane count is unknown, so there is no upper bound.
- `base` is assumed to be at least 2-byte aligned, since staging placeholders are detected by their lowest byte. `is_valid` does not check it.

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

`manifest_strategy` = `normalize` -> `apply_chunking` -> `apply_staging` -> `apply_d2d_implementation`

| Step | Input | Output |
| --- | --- | --- |
| `normalize` | the caller's spec | the same copy with both sides in their normal form, so that the later steps see the highest `copy_alignment` and the fewest runs. |
| `apply_chunking` | one spec | `parallel_copy_set` of independent single-spec plans, each at most `chunk_size` bytes. `chunk_size == 0` means no chunking. |
| `apply_staging` | each single-spec plan | a sequential plan: `[gather, staging -> staging, scatter]`, or a subset of it. A side is staged exactly when its window is not one contiguous run; host-to-host copies are never staged. Only the middle copy crosses devices, and it is always contiguous. The strategy's properties replace the spec's. |
| `apply_d2d_implementation` | each plan | device-to-device copies optionally routed through host staging buffers |

Why stage at all: the gather/scatter kernels are cheap because they run in parallel inside one device's memory. Crossing the bus is expensive per call, so each chunk should cross in one large contiguous copy.

## Remaining restrictions and implementation notes

- Chunk boundaries have to be rounded to a local alignment: `copy_alignment()`, the largest power of two up to 64 that divides the row extents, `d0_stride` and `d0_start_offset` of both sides and the shift between the two windows (the plane size is a multiple of `d0_stride`, so it adds nothing; the bases are unknown while planning, so the kernel narrows its element further for a base aligned to less). It keeps kernels on wide element types and is computed per copy, not stored. A smaller `chunk_size` is raised to it.
- The kernel's `int32` index path has to check the full index span of the window on both sides, not individual fields.
- Contiguity (`is_window_contiguous` on `data_layout`) is defined over the window. A window inside one row is a single `queue.copy`. There is no spec-level predicate; the backend asks both layouts.
- The general 3D kernel needs two divmods per side per element, which is what `data_layout::offset_at()` computes. Mitigations: special-case 1D/2D boxes, or split same-extent copies into box-shaped pieces run as `nd_range<2>`/`<3>` kernels without division. Which of those applies is a property of the copy, not of one layout, so it is reduced from both sides at once (as Celerity does in `layout_nd_copy`) rather than reported per layout.
- Overlap checks for copies within one allocation can only be conservative.
- Direct d2d copies with host staging pack the staging layout. Staging buffers are always 1D, so their size is the window length.
- Unstaged strided copies involving the host become one copy per run without native 2D copies: `queue.copy`, or `memcpy` between two host ends. Benchmark against the 2D library before accepting this.
- Host memory given by the caller may be pinned (`sycl::malloc_host`) or pageable. Kernels cannot read pageable memory, so a copy involving pageable host memory must not take the kernel path.

## Open decisions

These decide the public API and were settled before implementing the backend:

1. **Ownership.** Decided for now: the executor creates and owns its queues, see [async-execution.md](async-execution.md#decisions). The caller chooses the devices: the executor takes (device, context) pairs, so that `d_i` is the caller's device and its queues and staging memory live in the context the caller's memory belongs to, e.g. one context per device as Celerity creates them. Shortcuts take devices in their default contexts, or the first N GPUs. Taking the caller's queues as lanes is left to the Celerity integration, where copies are expected to run as `immediate` instructions that Celerity tracks through `copy_handle`. The SimSYCL system configuration and the data buffers (`dev_buffer`, `host_buffer` and their getters) move into the tests; the executor keeps only its staging buffers. Host staging is pinned only on request: with `COPYLIB_ALLOC_CPU_IDS` set, the executor allocates each device's host staging buffer while running on the CPU given for that device; without it, nothing is pinned. The 2D library's guess (half the hardware threads, split evenly by device index) is gone, since it was often wrong and could make construction fail in a restricted CPU set. A constructor parameter for the CPUs is left to the Celerity integration.
2. **Blocking vs. async execution.** Decided: `execute_copy` hands the plans to worker threads on an executor-owned `BS::thread_pool` and returns a handle with `is_complete()` and `wait()`, see [async-execution.md](async-execution.md#decisions).
3. **Error handling.** Decided: every failure throws `copylib::error`, including broken internal invariants, so `COPYLIB_ENSURE` throws instead of calling `std::exit(1)`. Failures inside a worker are reported as a message through `copy_handle::error()`, see [async-execution.md](async-execution.md#decisions).

These were left to the implementation:

4. **Staging alignment bug** in the 2D fulfiller (`size + alignment % size` does not round up). Fixed: the fulfiller rounds every staging size up to the 128-byte staging alignment.
5. **Staging memory reuse.** Decided: one slice of the staging buffers per pool worker, reused for every plan that worker runs, see [async-execution.md](async-execution.md#decisions). Only a single plan has to fit into a slice, not the whole set.
6. **Strategy selection.** Should the library offer `select_strategy(spec, exec)` with benchmark-based thresholds?
7. **Testing approach.** Decided: property tests on SimSYCL against an independent reference model and a byte-by-byte reference copy, see [testing.md](testing.md).

## Upcoming steps

1. ~~Settle open decisions 1-3.~~ Done, see [async-execution.md](async-execution.md#decisions).
2. ~~Update the skeleton.~~ Done: the window is in `data_layout` and its constructors, the 1D constructors take a
   length instead of a `fragment_length`, the native 2D/3D copy API and `COPYLIB_USE_CUDA` are gone, and the README
   table and example are updated. The accessors were reworked by dropping rather than defining them:
   - `dimensions()` is gone. How many strides a copy needs is a property of the pair of layouts, not of one of them,
     so it is reduced from both sides at once when the copy is executed.
   - `total_extent()` is gone. Its only use was sizing staging buffers, which are now
     always 1D and packed, so their size is the window length.
   - `layer_count()`, `layer_offset()` and `fragment_offset()` are gone. A window may start or end mid-row, so
     per-fragment indexing cannot describe what a copy transfers. They are replaced by `offset_at()`, the closed form
     for a single byte, and by a run iterator over the window, since replaced by `for_each_copy_run()`, which pairs the
     runs of both sides of a copy.
   - `unit_stride()` is renamed to `is_window_contiguous()`, which is what it means once it is defined over the window.
3. ~~Write tests first, ahead of the implementation.~~ Done, every layer passes, see [testing.md](testing.md).
4. ~~Implement the core.~~ Done, all planning layers pass: the layout accessors, `is_valid`, `normalize`, `apply_chunking`, `apply_staging`, `apply_d2d_implementation` and `manifest_strategy`. `is_equivalent` was dropped: the 2D implementation was of little use, and the tests check plans against their own reference model instead.
5. ~~Implement the backend.~~ Done: the executor on the caller's (device, context) pairs; the staging fulfiller with correct alignment, placing each plan in its worker's slice; the `execute_copy` paths (host `memcpy`, contiguous copy, merged 1D runs via `for_each_copy_run`); a general 3D kernel with the `int32` span check; and the worker threads with their handle. The kernel's 1D/2D special cases wait for the benchmarks.
6. Add strategy selection and port the benchmarks. Compare against the 2D library, especially strided copies involving the host, and against Celerity's backends. Also open to benchmarking: the number of workers, double buffering within a worker, and pinning the workers.
7. Integrate into Celerity: an `async_event` adapter around `copy_handle`, copies as `immediate` instructions, and the executor built from Celerity's (device, context) pairs, see [async-execution.md](async-execution.md).
