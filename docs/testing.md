# Test Plan

The test suite is written ahead of the implementation, so that the progress of the implementation is visible in CI.

## How progress is reported

Test cases covering functions that are not implemented yet are tagged `[!mayfail]`. Catch2 runs them and reports their
failures, but they do not fail the build, so CI stays green and the failure count shrinks as the implementation lands.
The CI workflow writes the totals into the GitHub run summary:

```
test cases:  78 |  17 passed |  61 failed as expected
assertions: 324 | 140 passed | 184 failed as expected
```

Remove the `[!mayfail]` tag from a case once the functions it covers are implemented, so that it starts guarding them
against regressions. `tests/passing_mayfail.sh <all_tests>` lists the tagged cases that pass, and CI adds that list to
the run summary. A case on it has either been implemented, and its tag should go, or it passes for the wrong reason.

Three rules keep this signal honest:

- **Each test case pairs a positive check with its negative checks.** While `is_valid` is a placeholder returning
  `false`, a lone `CHECK_FALSE(is_valid(malformed))` would pass for the wrong reason. The same goes for comparing two
  results of one placeholder with each other: `normalize(a)` equals `normalize(b)` when both return `{}`. Three cases
  broke this rule when they were first written, and `passing_mayfail.sh` caught all of them.
- **Tests depend on as little unimplemented code as possible.** Layers above the constructors build layouts, specs,
  staging ids and strategies with the `*_from_fields()` helpers in `test_utils.hpp` rather than through constructors,
  so a failure points at one layer.
- **Planning tests do not trust the library to check itself.** `is_equivalent` is one of the functions under test, so
  the planning layers are checked against the independent oracle described below instead.

## The oracle

`test_utils.hpp` contains a reference model of the library's semantics that shares no code with it:

- `reference_offsets(layout)` lists the allocation offset of every byte in the window, in packed order, by the most
  direct loop over the raw fields. `reference_runs(layout)` groups those into maximal contiguous runs.
- `simulate(plan_or_set)` executes plans symbolically. Instead of moving data it tracks, for every byte written, which
  byte its value originally came from, following values through staging buffers. It also records whether any address
  was written more than once and whether two plans of a set touched the same memory.
- `implements(plan_or_set, spec)` is the ground truth for `is_equivalent`: every byte of the target window ends up
  holding the corresponding source byte, each exactly once, nothing outside staging buffers is written, and no two
  plans of a set share memory. Overlapping chunks fail it even though they produce the right bytes, because design.md
  specifies `is_equivalent` as a tiling check.

`tests/reference_tests.cpp` checks the oracle against hand-computed values and hand-built plans, including the failure
modes it must detect. Those tests pass from the start; if one breaks, the problem is in the test utilities.

Comparisons of offset lists go through `first_difference()`, which reports the first mismatching index. Comparing the
vectors directly makes Catch2 print every element on failure, which for real layouts buries the CI log under thousands
of numbers.

Everything in `src/` currently returns a default-constructed value rather than having an empty body: a non-void function
falling off its end is undefined behaviour, which crashes the test binary instead of reporting a failing assertion.

## Layers

| # | Layer | File | Status |
| --- | --- | --- | --- |
| 0 | Generic utilities | `tests/utils_tests.cpp` | passing (`src/utils.cpp` is ported from the 2D library) |
| — | The oracle itself | `tests/reference_tests.cpp` | passing |
| 1 | Layout math | `tests/layout_tests.cpp` | written |
| 2 | Validation | `tests/validation_tests.cpp` | written |
| 3 | Normalization | `tests/normalization_tests.cpp` | written |
| 4 | `is_equivalent` | `tests/equivalence_tests.cpp` | written |
| 5 | Chunking | `tests/chunking_tests.cpp` | written |
| 6 | Staging | | to do |
| 7 | d2d implementation | | to do |
| 8 | `manifest_strategy` | | to do |
| 9 | Support: hashing and formatting | | to do |
| 10 | Backend data correctness | | to do |
| 11 | Ordering and dependency structure | | to do |
| 12 | Async handle semantics | | to do |
| 13 | Executor and build | | partially covered by the CMake setup |

### 1. Layout math

Pure, no SYCL, and mostly `constexpr`. Built on the `int[12,16,20]` example from [design.md](design.md#data-layout),
which is available as `copylib_testing::reference_box`. Covers the box constructors, the window defaulting to the whole
box, the 1D constructor producing exactly the 1D normal form, `total_bytes`, `end_offset`, `offset_at` and
`for_each_contiguous_run` (each checked against the oracle for every byte), the contiguity predicates over the window,
field-wise `operator==`, `base_ptr`, and `staging_id` round-tripping.

### 2. Validation

One accept and one reject per rule in [design.md](design.md#data-layout): non-zero strides, plane stride a multiple of
the row stride, offset alignment, box-fits, canonical encoding, 2-byte base alignment, window bounds, equal window
lengths across a spec, and the reshaping case. Plus: `is_valid` must be total, i.e. never crash on arbitrary field
values — worth adding as a randomized case.

### 3. Normalization

Two groups. The semantic cases hold for any correct collapse: the byte mapping is preserved (checked by comparing the
reference offsets before and after, since a structural comparison would miss a wrong collapse), the window is unchanged,
each row of the normalized box is one maximal contiguous run, normalization is idempotent, and its result is valid.

The normal-form cases pin the exact encoding from [design.md](design.md#normal-form-and-equality), stated with the
`normal_form::one_run` and `normal_form::uniform_runs` builders that mirror the design table: single runs become 1D,
evenly spaced runs become 2D (including partial rows continuing across a plane boundary), and a two-level grid stays as
it is. The uniqueness cases then check that different encodings of the same bytes normalize to identical fields, which
is what makes field-wise `operator==` sufficient for comparing bytes. `normalize(spec)` normalizes each side
independently, shown with a spec whose source collapses fully while its target cannot.

### 4. `is_equivalent`

The 2D suite used `is_equivalent` as the oracle for its planning tests without testing a single negative case, so a
broken `is_equivalent` would have let all of them pass. Here every case states its verdict for both the library and
`implements()`. Negative cases: an empty plan or set, steps of a staged plan in the wrong order, a shifted target
window, scattering to the wrong allocation or the wrong device, a missing chunk, overlapping chunks, a duplicated
chunk, a stray plan writing outside the spec, and concurrent plans sharing a staging buffer. Positive cases include a
shuffled set and a reshaping copy through staging.

### 5. Chunking

No chunking for `chunk_size == 0`. For a range of chunk sizes, the chunks implement the spec according to the oracle,
each is a single direct copy of at most `chunk_size` bytes, the box fields are identical across all chunks because
chunking narrows only the window, and source and target windows advance in lockstep, reshaping included. Boundaries
are multiples of the local alignment (8 for the reference box). Chunks are as large as alignment permits: `96` gives
`96, 96, 96`, `64` gives `64, 64, 64, 64, 32`, and `77` is rounded down to `72` per chunk, since each chunk may be at
most `chunk_size`. A partial window is chunked within itself.

The edge cases from [design.md](design.md#chunking-edge-cases): a `chunk_size` below the alignment is raised to it, and
an unaligned window keeps its interior boundaries at absolute aligned offsets, so `[3, 288)` with `chunk_size` 64
becomes `61, 64, 64, 64, 32`.

### 6. Staging

Not desired, not necessary, and required, for the source end, the target end and both. The plan shape is
`[gather, staging -> staging, scatter]` or a subset. The middle copy is always contiguous and 1D and is the only step
crossing devices. A staged chunk gets the window `[0, end - start)` in its staging buffer. Repeated staging ids imply
identical size, device and host flag.

### 7. d2d implementation

All four implementations, applied to staged and unstaged plans. The device of each step is correct, only the host hop
crosses the bus, and `host_staging_at_both` adds exactly one extra host-to-host copy.

### 8. `manifest_strategy`

The cross product of copy type, properties, d2d implementation, chunking and reshaping, asserting `is_valid(set)`,
`is_equivalent(set, spec)` and property propagation for every combination. Randomized over shapes, boxes, windows and
strategies: no SYCL, so thousands of cases cost little.

### 9. Support: hashing and formatting

Ported from the 2D suite and extended for the window — a hash ignoring `start` and `end` is a silent correctness bug.
Two encodings of the same box must compare and hash equal, which is what canonical encoding buys. The Catch2
`StringMaker` specializations for `data_layout` and `copy_spec` belong here too: they call the formatters, and Catch2
invokes them exactly when an assertion fails.

### 10. Backend data correctness

Needs SYCL. The oracle is a **host-side reference copy**, not an expected-value computation inside the validation
kernel as in the 2D suite: in 3D that mapping is the thing under test, and duplicating it in the test risks reproducing
the bug.

Fill the target with a sentinel, copy, compare the whole buffer against the reference — that checks the copied bytes and
that untouched bytes stayed untouched, which is the likely failure mode of chunk alignment rounding. Assert the source
is unchanged. Forced cases: whole allocation, single row, window inside one row, windows crossing row and plane
boundaries, reshaping, one byte, odd row extents (alignment 1) and extents with alignment 64. The same spec with and
without `use_kernel` must produce identical bytes. Then the randomized version of all of it, with a fixed seed and an
override for reproduction.

### 11. Ordering and dependency structure

SimSYCL executes everything at submit time, so a copy set with **no dependency edges at all** still produces correct
bytes and layer 10 gives no coverage here. See [async-execution.md](async-execution.md#eager-assignment).

The mechanism is a recording seam: an executor that logs `(lane, command, dependencies)` instead of submitting.
Assertions on the recorded graph: every command of one call lands on the lane the call was given; the last command on
the assigned lane transitively depends on the tail of every private lane; completion of the returned handle implies
completion of every recorded command; no staging buffer is reused before the last command reading it completed; a flush
is issued after submission. This is pure data, so it runs in CI.

Worth building before the backend, because it decides whether the executor is observable at all.

### 12. Async handle semantics

`is_complete()` is monotonic, never blocks, and is safe to call repeatedly including before submission. Work proceeds
without anyone polling. A forced failure invokes the injectable failure hook exactly once and no exception escapes.

## What CI can and cannot prove

| Runs in CI on SimSYCL | Needs real hardware |
| --- | --- |
| Layers 1-9 in full | Actual concurrency and ordering |
| Layer 10 data correctness | Peer access and true multi-device |
| Layer 11 recorded structure | The kernel's int32 index path (needs > 2 GiB, opt-in) |
| Layer 12 handle semantics | Any performance claim |

SimSYCL is only the SYCL implementation used by the dev container and CI. It is not a design target.

## Accessor semantics

Four accessors had semantics that nothing pinned down. Rather than deciding them, they were dropped, because no caller
in the pipeline needed them — see step 2 of [design.md](design.md#upcoming-steps). What the tests pin now:

- `total_bytes()`: bytes covered by the box, gaps excluded.
- `window_length()`: `end - start`, the number of bytes a copy actually transfers.
- `end_offset()`: the first byte past the box, **relative to the allocation base**, so that it can be bounds-checked
  against a buffer size. That is its only remaining caller, which settles the question by elimination.
- `offset_at(packed_offset)`: the allocation offset of one byte of the box, gaps excluded. Byte *i* of a copy is at
  `offset_at(start + i)`.
- `is_contiguous()`, `contiguous_fragments()`, `contiguous_layers()`: defined over the window, not the box. A window
  inside a single row is contiguous even when its box is not.
- `for_each_contiguous_run(layout, f)`: the runs are ordered, non-overlapping, and their lengths sum to
  `window_length()`.

## The 2D library as a reference

`old/gpu2dcopylib` is a local, gitignored copy and **cannot be built**: `lib/copylib_core.hpp` is a half-finished edit
towards 3D in which the members the constructors and `total_bytes()` still use are commented out. The backend, tests
and benchmarks in that tree are pristine 2D and are worth reading and porting; the tree as a whole is not a runnable
oracle. Pristine upstream is `PeterTh/gpu2dcopylib` at `8984ee5`.
