# Async Execution: Analysis

Background for open decisions 1-3 in [design.md](design.md#open-decisions); the decisions taken are recorded in [Decisions](#decisions). This document records how the 2D library executes copies, how Celerity (the main consumer of this library) runs asynchronous work, and what that implies for the execution API of the 3D library.

Sources:

- The 2D library is referenced by path in `old/gpu2dcopylib`, the local reference copy. It is gitignored, so those references are plain paths rather than links.
- Celerity links are pinned to `celerity-runtime` commit [`1039045`][cel-commit].

## Goal

`execute_copy` on a `parallel_copy_set` returns without waiting, and a single handle reports the state of the whole set.

## Option space

### Handle types

| Option | Mechanism | Strengths | Weaknesses |
| --- | --- | --- | --- |
| **A. Terminal `sycl::event`** | An empty command that `depends_on` all leaf events | Native SYCL type, composes with the caller's SYCL graph | Needs cross-context `depends_on`, which Celerity's setup rules out (see [below](#one-sycl-context-per-device)). Cannot represent host work running on threads. One extra submission per set. |
| **B. Leaf events** | The handle holds the submitted events and polls `command_execution_status` | No threads. All work is submitted when the call returns. Per-step profiling data is kept. | Cannot make submission decisions at runtime. Host steps need `host_task`. |
| **C. Worker thread** | The handle wraps the completion of a thread running the plans | Arbitrary control flow. Exact error attribution. Host steps are trivial. | One blocked thread per in-flight set. "Incomplete" also covers "not submitted yet". Thread handoff latency on every call. |

Completion callbacks are not an option: SYCL 2020 has no portable event callback.

B and C can expose the same interface. They differ in who submits (the caller or a worker), what `is_complete()` measures (device status or worker progress), thread cost, and how errors are attributed.

### What needs runtime-driven submission

B can express any graph of work that is known at call time, CPU steps included via `host_task`. It cannot express a submission that depends on something observed at runtime:

- **A d2d copy between devices in different SYCL contexts.** `depends_on` does not portably accept an event from another context. The only way to order the two sides is a thread that waits on the first event and then submits the second.
- **Staging buffer recycling** (decision 5). With 4096 chunks, B submits everything at once, so all 4096 staging buffers are live at the same time. Recycling a buffer means submitting chunk *i+N* only after chunk *i* has finished.
- **Aborting a plan after a failure.** B has already submitted the remaining steps.
- **Bounding queue depth** for very large copy sets.
- **Lazy staging fulfilment**, i.e. assigning a buffer address when a step is reached rather than before anything is submitted. This is what makes recycling possible.
- **Re-selecting the strategy** after measuring the first chunk (decision 6).

Host staging steps are not on this list. `host_staging_at_both` is a fixed chain of five steps, which `host_task` expresses. Preferring a thread there is a matter of cost, not expressiveness: a `host_task` in an in-order queue blocks that lane, and dispatch through the SYCL runtime is not free for small copies.

## The 2D library

### Ordering by blocking, not by events

The 2D library never keeps a `sycl::event`. The return values of `queue.copy()`, `queue.memcpy()` and kernel submissions are discarded, and `depends_on` is not used anywhere.

Ordering within a plan comes from two sources:

- in-order queues (`old/gpu2dcopylib/lib/copylib_backend.cpp:210`);
- a blocking `wait_and_throw()` on the previous step's queue whenever the target changes (`copylib_backend.cpp:320` and `:335`).

A staged d2d copy (gather, cross, scatter) therefore stops the calling thread twice.

### Parallelism through a thread pool

`execute_copy(set)` (`copylib_backend.cpp:466-510`):

1. fulfils the staging of every plan on the calling thread;
2. splits the plans across a function-local `static BS::thread_pool` with one worker per queue index, where single-spec plans alternate between source and target device;
3. waits on all futures.

Workers block in `wait_and_throw()`. The set level is already shaped like option C; only the final join makes the API synchronous.

### Errors are fatal but unstructured

- `COPYLIB_ENSURE` prints, asserts and calls `std::exit(1)` (`old/gpu2dcopylib/lib/utils.hpp:124-132`).
- No `async_handler` is installed; the queue property lists only carry `in_order` and vendor flags (`copylib_backend.cpp:209-220`).

A SYCL async error therefore surfaces as an exception from `wait_and_throw()`. On a thread-pool worker, that exception escapes the thread function and calls `std::terminate`. The current behaviour is closer to a panic than to the exception model proposed in decision 3.

### Staging lifetime relies on blocking

`staging_fulfiller` (`copylib_backend.cpp:385`) hands out increasing offsets into a fixed buffer per device and never frees them. A new fulfiller is created for every call (`:462` for plans, `:476` for sets), so every call starts at offset 0.

This is only safe because a call cannot return before its copies have finished. Once `execute_copy` returns early, the next call hands out staging memory that is still in use. This hazard exists regardless of which handle type is chosen.

### Tests and benchmarks

- Benchmarks put `exec.barrier()` around the timed region (`old/gpu2dcopylib/benchmarks/intra_device.cpp:56-61`).
- Tests follow `execute_copy` with an explicit queue wait (`old/gpu2dcopylib/tests/backend_tests.cpp:130-131`).

Neither breaks under an async API. Without an explicit wait, the benchmarks would time submission instead of completion.

## Celerity

### `async_event` only supports polling

[`async_event`][cel-async-event] offers `is_complete()`, an optional `get_native_execution_time()`, and a `get_result()` that only allocations use. It has no `wait()` and no error channel. The contract ([async_event.h:23-25][cel-async-event-contract]) requires:

- once `is_complete()` returns true, it always returns true;
- polling is cheap and can be repeated;
- the operation proceeds in the background even when nobody polls.

### The executor thread never blocks

The executor loop polls in-flight instructions, polls the submission queue, and issues at most one instruction per iteration so that it keeps checking for completions in between ([live_executor.cc:442-449][cel-executor-loop]). Completion is detected only through `is_complete()`.

### One SYCL context per device

Each device state creates its own context from its single device ([sycl_backend.cc:122][cel-context]). Cross-device `depends_on` is therefore unavailable in the configuration this library would run in.

### Dependencies are resolved without events, and lanes are inputs

`out_of_order_engine` assigns an instruction once either all predecessors have completed, or all incomplete predecessors are on the same in-order queue ([out_of_order_engine.h:84-86][cel-engine-assign]). No cross-queue event dependencies are used.

The engine, not the backend, picks the queue. The backend interface requires asynchronous work to be "explicitly assigned to in-order queue ids as assigned by `out_of_order_engine`" ([backend.h:19-20][cel-backend-contract]), and the copy entry point takes device and lane as parameters:

```cpp
async_event enqueue_device_copy(device_id device, size_t device_lane, const void* source_base, void* dest_base,
    const region_layout& source_layout, const region_layout& dest_layout, const region<3>& copy_region, size_t elem_size);
```

### Eager assignment

How that second case works matters for this library. `try_mark_for_assignment` ([out_of_order_engine.cc:188-227][cel-try-mark]) marks an instruction as eagerly assignable when (1) all of its incomplete predecessors are on the same device and lane, and (2) one of them is that lane's `last_incomplete_submission`, which is recorded for every assignment at [:438-440][cel-lane-bookkeeping]. `assign_one` then dispatches it although its predecessors are still running ([:413-421][cel-eager-assign]):

```cpp
// After an instruction is popped from the assignment queue, "eager-assignability" is unconditional. Force-assign to the same lane to implicitly fulfill
// all remaining dependencies through queue ordering.
assigned.device = eagerly_assignable->device;
assigned.lane = eagerly_assignable->lane;
```

The dependency is discharged by queue order alone. The predecessor's `async_event` is never consulted on this path, so a correct event does not protect it.

This sets the invariant for the copy entry point ([live_executor.cc:785-789][cel-copy-dispatch]): **everything one call submits must go to the (device, lane) it was given**. If the copy for instruction C on (device 0, lane 2) also placed work on lane 3, on device 1, or on a host thread, then a kernel K that depends on C — force-assigned to lane 2 and submitted while C is still running — would be ordered behind only the part on lane 2 and would race with the rest. Nothing detects this: the assertion in that branch only checks the engine's own bookkeeping, which stays self-consistent because the engine cannot observe queues it did not hand out. The result is intermittently wrong data, not a crash.

The number of commands per call is not constrained; Celerity's own chunked copy submits one `memcpy` per contiguous chunk. The constraint is one *lane* per call.

### Options B and C already coexist behind one handle

| Implementation | Mechanism | `is_complete()` |
| --- | --- | --- |
| [`sycl_event`][cel-sycl-event] | Keeps the last event, plus the first when profiling | One `command_execution_status` query on the last event |
| [`thread_queue::event`][cel-thread-queue] | `std::promise` / `std::future` on a worker thread | `future.wait_for(0s)`, then cached |
| [`delayed_async_event`][cel-delayed-event] | Shared state with an `atomic_bool`, filled in by a submission thread | `false` until filled, then forwarded |

Device kernels and copies involving a device go to in-order SYCL queues. Host tasks and host-to-host copies go to `thread_queue` workers. Optional per-device submission threads only submit; they exist to keep launch latency off the executor thread.

Using `std::future` / `std::promise` for `delayed_async_event` instead of an atomic flag cost 50% in system-level benchmarks ([sycl_backend.cc:320-321][cel-future-cost]).

### One event for many commands is the last event on one queue

`nd_copy_device_chunked` submits one `queue.memcpy` per contiguous chunk and returns only the last event ([sycl_generic_backend.cc:22-36][cel-generic-chunked]). This works because every chunk goes to the same in-order queue. The first event is only kept to compute the profiling duration.

### Async errors panic

- The queue `async_handler` is `report_errors`, which removes duplicate messages and calls `utils::panic` ([sycl_backend.cc:85-106][cel-report-errors], installed at [:177][cel-queue-create]).
- `check_async_errors()` calls `throw_asynchronous()` on every queue ([sycl_backend.cc:288-307][cel-check-errors]) once every 256 executor iterations.
- The CUDA backend panics on any CUDA error and on a failed `cudaEventQuery`.
- `thread_queue` does not catch exceptions ([thread_queue.h:125-137][cel-thread-queue-execute]): a throwing host job terminates the process.

No exception from the async layer ever reaches application code.

### Polling needs an explicit flush

AdaptiveCpp does not guarantee that command groups are scheduled until something waits on them, and Celerity cannot wait without blocking its loop. Every submission path therefore calls `flush(queue)` ([sycl_backend.cc:76-82][cel-flush]).

### Copy backends

- The generic backend copies device memory as a loop of 1D `memcpy` calls ([sycl_generic_backend.cc:22-57][cel-generic-chunked]) and stages d2d copies through host memory without parallelism ([:65][cel-generic-staging]).
- The CUDA backend uses native `cudaMemcpy2DAsync` and `cudaMemcpy3DAsync` for strided copies ([sycl_cuda_backend.cc:34-64][cel-cuda-copy]).

## Consequences for the 3D library

### 1. Shape the handle like `async_event_impl`

Poll-only, monotonic, cheap, non-blocking, non-throwing, with an optional execution time. Celerity can then integrate it with a single adapter class. Anything beyond that, such as `wait()` or rethrowing errors, is something Celerity would have to adapt away.

### 2. Poll one event per lane, not per chunk

A copy set spans several queues, so Celerity's single-last-event approach does not apply directly, but it does apply per lane: because queues are in-order, the last event on a (device, lane) pair implies completion of everything submitted there before it. A set of 4096 chunks on four lanes polls four events. Keeping the first event per lane as well provides profiling data at no extra cost.

### 3. Take device and lane as parameters

If the executor creates its own queues and chooses targets itself (as the 2D library does with `queue_idx` and alternating devices), Celerity's engine can no longer rely on "same in-order lane implies ordering", which its [eager assignment](#eager-assignment) depends on. Decision 1 is therefore a correctness requirement for integration, not only a cleanup. The executor should accept the caller's devices and in-order queues, mirroring `enqueue_device_copy(device, lane, ...)`.

This library is fastest with more than one lane per device, which that invariant forbids. See [Working within the lane constraint](#working-within-the-lane-constraint) for the two ways around it.

### 4. Fail through an injectable handler, not exceptions

Celerity turns async errors into a panic. An event that carries an `exception_ptr` would be unwrapped and panicked on at the boundary. What matters to the consumer is that errors are never swallowed, come with a clear message, and never cause blocking.

- Keep caller input checks as the non-fatal, queryable `is_valid()` family.
- Route all other failures through a replaceable failure handler, so Celerity can forward them to `utils::panic` and keep its logging.
- Do not call `std::exit` from worker threads: it runs static destructors while other threads still hold device state.

### 5. Reclaim staging memory on completion

Without blocking, the offset allocator hands out live buffers. Celerity's allocation tracking does not see this library's staging buffers, so reclamation has to be driven by lane completion inside the library. This requires submission that reacts to runtime state (see [What needs runtime-driven submission](#what-needs-runtime-driven-submission)) for chunked sets, whatever the handle looks like.

### 6. Flush after submitting

This is invisible today because every call blocks. With a polled handle on AdaptiveCpp, an unflushed queue may never start work. Adopt Celerity's `flush()` workaround.

### 7. Benchmark against Celerity's CUDA backend

This library clearly improves on Celerity's generic backend, which uses 1D memcpy loops and unparallelised host staging. Celerity's CUDA backend, however, uses native 2D/3D copies. If native 2D/3D copies and `COPYLIB_USE_CUDA` are dropped, strided copies on CUDA hardware must beat them using kernels and staging alone. The comparison in [Remaining restrictions](design.md#remaining-restrictions-and-implementation-notes) should include `sycl_cuda_backend`, not only the 2D library.

## Working within the lane constraint

The library performs best with two lanes per device, which the [invariant above](#eager-assignment) forbids. There are two ways around it, and which one applies depends on whether the second lane is on the same device or on the other one.

### Route 1: a private lane, joined back into the assigned one

Use the assigned lane plus a second queue that the library creates itself. Both are on the same device and therefore share a `sycl::context` ([sycl_backend.cc:122][cel-context]), so `depends_on` between them is legal. Only cross-device dependencies are ruled out by Celerity's per-device contexts.

The last command submitted on the assigned lane is then a join: an empty command group depending on the tail of the private lane. Anything the engine queues behind it on that lane, eagerly assigned or not, is ordered after both lanes.

- The private queue must not be taken from Celerity's lane pool. A lane with no in-flight assigned instructions counts as free ([out_of_order_engine.cc:152-158][cel-free-lane]) and will be handed out to other work.
- The join also answers the single-event question: it is the last command on the assigned lane and transitively covers the private lane, so one event and one poll describe the whole copy.
- The private queue is not covered by `check_async_errors()`, so it needs its own handler wired to the failure hook from [consequence 4](#4-fail-through-an-injectable-handler-not-exceptions).
- Cost is one extra submission per copy.
- Requires the executor to take a device or context from the caller (decision 1).

No changes to Celerity are needed for this route.

### Route 2: `target::immediate` for lanes spanning two devices

If the second lane belongs to the other device, as in a staged d2d copy with the gather on the source and the scatter on the target device, route 1 does not apply: there is no shared context, so there is no join.

Celerity already has a category for operations that schedule themselves. `target::immediate` is documented for "instructions where asynchronicity is managed outside the backend (p2p transfers through communicator and receive_arbiter)" ([out_of_order_engine.h:26-29][cel-immediate]). Such instructions are assigned no lane, are assigned only once all predecessors have actually completed, and their successors cannot be eagerly assigned onto them, because `try_mark_for_assignment` bails out on a predecessor with a different target. Ordering falls back entirely to `is_complete()`.

They remain asynchronous: `send_instruction` and the receive instructions use this target and still have `issue_async` overloads returning an `async_event` ([live_executor.cc:386-389][cel-immediate-async]).

This route permits arbitrary internal parallelism: several lanes, both devices, host threads. It requires an upstream change, namely classifying copies backed by this library as `immediate` rather than `device_queue` in the engine's target dispatch ([out_of_order_engine.cc:259-274][cel-submit-dispatch]).

Its cost is that the copy waits for its predecessors to complete instead of being queued behind them, and its successors do the same, so each copy pays roughly one polling cycle at either end. That is negligible for a large chunked copy and a regression for a small one, which argues for gating by size: small copies stay on the single-lane path, large ones go through `immediate` (decision 6).

### Which one

Build for route 1, and keep route 2 open by putting queue acquisition behind a seam, so that both "the assigned lane plus private lanes" and "lanes chosen entirely by the library" are expressible. Then measure: two lanes have to beat one in-order queue plus the join submission. Propose route 2 upstream only with those numbers in hand.

## Decisions

Taken on 2026-09-25 for building the library itself. Celerity integration follows later and may revisit them; in
particular, [consequence 3](#3-take-device-and-lane-as-parameters) makes caller-provided lanes a requirement there.

| Topic | Decision |
| --- | --- |
| Handle | **Option C.** `execute_copy` hands the plans to worker threads and returns immediately. Workers block between steps as in the 2D library, and the handle tracks their completion. |
| Waiting | The handle has a blocking `wait()` next to the non-blocking `is_complete()`. Tests and benchmarks wait; a Celerity adapter would only poll. |
| Thread pool | `BS::thread_pool`, as in the 2D library, but owned by the executor rather than a function-local `static`. It still has to be added as a dependency. |
| Queues | The executor creates and owns its in-order queues, as in the 2D library. |
| Staging lifetime | Open. Until it is settled, staged calls are not blocked: overlapping ones are undefined behaviour, and the executor warns when a staged call starts while another is in flight. |
| Error handling | Every failure throws `copylib::error`: invalid input, broken internal invariants and resource failures alike, so `COPYLIB_ENSURE` throws instead of calling `std::exit`. A failure inside a worker cannot reach the caller as an exception and is reported as a message through `copy_handle::error()` instead. The executor's queues get an async handler that rethrows, so that asynchronous SYCL errors reach that report instead of the default handler, which terminates. |

Compared with the analysis above, this leaves out three things for now, all motivated by Celerity: polling events
(option B) for the parts that need no runtime decisions, taking the caller's queues, and the injectable failure handler
of [consequence 4](#4-fail-through-an-injectable-handler-not-exceptions); a Celerity adapter turns exceptions into
panics instead. Keeping queue selection behind one function in the executor leaves room for caller-provided lanes later.

## Proposed interface

Implemented in `src/backend.cpp` as proposed here.

```cpp
namespace copylib {

namespace detail {
	struct copy_state; // shared by a handle and the workers running its plans
}

// Completion state of one execute_copy call. Copies of a handle share the same state.
class copy_handle {
  public:
	// true once every plan has finished, successfully or not; monotonic, never blocks, never throws
	[[nodiscard]] bool is_complete() const;

	// blocks until is_complete() is true; never throws
	void wait() const;

	// the first failure of any plan once complete, empty on success
	[[nodiscard]] std::optional<std::string> error() const;

	// time from the call until the last plan finished, once complete
	[[nodiscard]] std::optional<std::chrono::nanoseconds> execution_time() const;

  private:
	explicit copy_handle(std::shared_ptr<detail::copy_state> state);
	friend copy_handle execute_copy(executor& exec, const parallel_copy_set& set);

	std::shared_ptr<detail::copy_state> state;
};

// hands the plans of the set to the executor's thread pool and returns immediately
[[nodiscard]] copy_handle execute_copy(executor& exec, const parallel_copy_set& set);

} // namespace copylib
```

In use:

```cpp
copylib::executor exec(buffer_size, 2, 2);
const auto set = copylib::manifest_strategy(spec, strategy, copylib::basic_staging_provider{});

const auto handle = copylib::execute_copy(exec, set);
// ... other work ...
handle.wait();
if(const auto error = handle.error()) { copylib::utils::err_print("copy failed: {}\n", *error); }
```

The per-spec and per-plan `execute_copy` overloads become synchronous building blocks in `detail`, run by the workers.
Only `execute_copy` creates handles, so there is no public constructor and no empty handle.

### Handle details

- **A counter, not futures.** `copy_state` holds an atomic count of unfinished plans, the start and end time, the first
  error, and a mutex with a condition variable for `wait()`. Each worker decrements the count when its plan ends, and
  the last one records the end time and notifies. `is_complete()` is a single atomic load and monotonic by
  construction. Keeping the pool's per-task `std::future`s instead would make every poll cost one check per plan, 4096
  for a large chunked set, and `future::get()` rethrows, which the handle must not.
- **Failures complete, too.** A worker catches whatever its plan throws (a `sycl::exception` from waiting on a step, or
  any `std::exception`), records the first message and ends that plan; the other plans run to completion, so `wait()`
  always returns. Since `COPYLIB_ENSURE` throws, a failed check inside a worker is reported the same way.
- **Workers wait on their own step's event**, not on the whole queue as the 2D library's `wait_and_throw()` did.
  Several calls can share the executor's queues, and waiting on a queue would also wait for the other calls' work.
- **Dropping a handle** neither blocks nor cancels the copy, because the workers keep the state alive. Source and target
  memory must stay valid until the handle completes, as for any asynchronous copy. Destroying the executor waits for its
  pool, and with it for every copy in flight.
- **Calls are independent.** Several can be in flight, their plans interleave on the pool, and nothing orders one call
  against another. A second copy that reads what the first one writes has to wait on the first handle.
- **Execution time** is wall-clock time from the call to the last completion (`std::chrono::steady_clock`), so it
  includes time spent queued in the pool. Per-step device times would need profiling-enabled queues, which can be added
  later.
- **Pool size** stays as in the 2D library, one worker per queue index, until benchmarks say otherwise.

## Open questions

- **Staging lifetime.** The 2D fulfiller hands out offsets from 0 on every call, which was safe only because calls
  blocked until their copies finished. With option C, two calls in flight receive the same staging memory. The options
  are an allocator owned by the executor that releases a call's buffers when its last plan finishes, serialising staged
  calls so that a second one waits for the first, or staging memory from the caller (next point). Whichever it is,
  reclamation hangs off the same completion point as the handle. Until then, overlapping staged calls are undefined behaviour and the executor only warns about them.
- **Staging ownership.** Staging buffers could come from the caller's allocator, which Celerity tracks, instead of from
  buffers owned by the executor.
- **Multi-lane execution under Celerity**, part of the integration: which [route](#working-within-the-lane-constraint)
  to take, pending the measurement described in [Which one](#which-one). A third option, not worked out here, is to keep
  manifesting as a pure planning API that Celerity's instruction graph generator calls, so that it emits one instruction
  per plan and assigns each its own lane.
- **Dropping native 2D/3D copies**, pending the CUDA backend comparison above.

## Testing note

SimSYCL is only the SYCL implementation of the devcontainer and CI; it is not a design target. It executes synchronously: `event::wait()` and `handler::depends_on` do nothing, and every event reports `complete`. CI therefore checks that the data is copied correctly, but not the ordering: a set with missing dependencies or early staging reuse still produces correct bytes. There is no test-only hook to record submissions and no dedicated ordering test: correct bytes from the same tests run on an asynchronous SYCL implementation are taken as sufficient, see layer 10 in [testing.md](testing.md).

[cel-commit]: https://github.com/celerity/celerity-runtime/tree/10390458eb2d46a74525df7ac2420a1d80acdf2a
[cel-async-event]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/include/async_event.h#L49-L72
[cel-async-event-contract]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/include/async_event.h#L23-L25
[cel-executor-loop]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/live_executor.cc#L442-L449
[cel-context]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/backend/sycl_backend.cc#L122
[cel-engine-assign]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/include/out_of_order_engine.h#L84-L86
[cel-try-mark]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/out_of_order_engine.cc#L188-L227
[cel-lane-bookkeeping]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/out_of_order_engine.cc#L438-L440
[cel-eager-assign]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/out_of_order_engine.cc#L413-L421
[cel-copy-dispatch]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/live_executor.cc#L785-L789
[cel-free-lane]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/out_of_order_engine.cc#L152-L158
[cel-immediate]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/include/out_of_order_engine.h#L26-L29
[cel-immediate-async]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/live_executor.cc#L386-L389
[cel-submit-dispatch]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/out_of_order_engine.cc#L259-L274
[cel-backend-contract]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/include/backend/backend.h#L19-L20
[cel-sycl-event]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/include/backend/sycl_backend.h#L26-L39
[cel-thread-queue]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/include/thread_queue.h#L84-L114
[cel-delayed-event]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/include/backend/sycl_backend.h#L42-L63
[cel-future-cost]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/backend/sycl_backend.cc#L320-L321
[cel-generic-chunked]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/backend/sycl_generic_backend.cc#L22-L57
[cel-generic-staging]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/backend/sycl_generic_backend.cc#L65
[cel-report-errors]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/backend/sycl_backend.cc#L85-L106
[cel-queue-create]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/backend/sycl_backend.cc#L177
[cel-check-errors]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/backend/sycl_backend.cc#L288-L307
[cel-thread-queue-execute]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/include/thread_queue.h#L125-L137
[cel-flush]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/backend/sycl_backend.cc#L76-L82
[cel-cuda-copy]: https://github.com/celerity/celerity-runtime/blob/10390458eb2d46a74525df7ac2420a1d80acdf2a/src/backend/sycl_cuda_backend.cc#L34-L64
