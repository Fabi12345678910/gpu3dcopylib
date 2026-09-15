# GPU 3D Copy Lib [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/celerity/celerity-runtime/blob/master/LICENSE)

This library intends to provide a simple very high performance way to copy strided data with a 3D layout between GPU devices, and between CPU and GPU memory spaces.

It is implemented in C++ and SYCL, and provides a C++ API.

This library is based on `https://github.com/PeterTh/gpu2dcopylib`, with the main difference being the implementation of a 3rd copy dimension.

This library was specifically designed for the use within celerity, and therefore may has interface features specific to the requirements presented by the celerity runtime

## Status

The API is complete, the implementation is not: every function is currently a skeleton with an empty body.
`src/utils.cpp` is the only exception, it is carried over from the 2D library as-is.

## Prerequisites

- A SYCL implementation
    - The library is intended to work with SimSYCL (in CI), DPC++, and AdaptiveCpp
- CMake 3.24 or later
- [optional] CUDA Toolkit for CUDA-backend-specific interop features

## Building

This library is meant to be vendored into a larger project:

```cmake
add_subdirectory(vendor/gpu3dcopylib)
target_link_libraries(my_target PRIVATE copylib::copylib)
```

Because the public headers expose SYCL types, the consuming target needs SYCL applied to it as well.
With SimSYCL this happens automatically; with AdaptiveCpp and DPC++ the consumer has to call
`add_sycl_to_target(TARGET my_target)` on its own targets, as device compilation requires it anyway.

It can also be built standalone:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/sycl
cmake --build build
```

The CMake script will report which SYCL implementation it has found and is using.

### Configuration Options

- `COPYLIB_USE_FMT`: Use the fmt library instead of relying on the C++20 `std::format`. Default: `OFF`
- `COPYLIB_USE_CUDA`: Enable CUDA-backend-specific interop features. Default: `OFF`

Both dependencies have to be findable by CMake if they are enabled; nothing is fetched automatically.

## Data Layout

All copies are described by a `data_layout`, a box of data inside a larger 3D allocation. All values are in bytes, `d0` is the innermost dimension and always contiguous, `d2` the outermost:

| Field | Meaning |
| --- | --- |
| `base` | base address of the allocation (or a `staging_id` for a staging buffer which has not been placed yet) |
| `d0_stride` | size of one full row of the allocation |
| `d1_stride` | size of one full plane of the allocation |
| `dN_start_offset`, `dN_end_offset` | start and end of the box along dimension N |

The outermost extent of the allocation is never needed. For an `int` allocation of shape `[12, 16, 20]`, `d0_stride` is `20 * 4` and `d1_stride` is `16 * 20 * 4`.

## Usage

To use the library for copy operations, first, an instance of an executor must be created.  
Then, copy operations can be specified, turned into optimized parallel copy sets, and executed.

Here is a complete example which manually specifies all parameters to serve as a reference:

```cpp
#include <copylib/copylib.hpp>

using namespace copylib;

// === 1. Initialization

const int64_t buffer_size = 128 * 1024 * 1024; // 128 MiB for staging buffers
const int64_t queues_per_device = 2; // number of in-order queues per device for asynchronicity
executor exec(buffer_size, 2, queues_per_device); // create an executor
utils::print(exec.get_info()); // [optional] print information about the executution environment

// === 2. Specifying a copy operation

// provide these pointers as appropriate to source and target memory
intptr_t src_ptr = 0x1; // pointer to the source
intptr_t dst_ptr = 0x2; // pointer to the destination

// an int allocation of shape [12, 16, 20], of which the box [2, 5) x [3, 7) x [4, 10) is copied; all values are in bytes
const int64_t elem = sizeof(int);
const int64_t d0_stride = 20 * elem; // size of one full row
const int64_t d1_stride = 16 * d0_stride; // size of one full plane

// source data layout
const data_layout source_layout{src_ptr, d0_stride, d1_stride, 4 * elem, 3 * d0_stride, 2 * d1_stride, 10 * elem, 7 * d0_stride, 5 * d1_stride};
const data_layout target_layout{dst_ptr, source_layout}; // target data layout, same structure as the source

// copy from device 0 to device 1
const copy_spec spec{device_id::d0, source_layout, device_id::d1, target_layout};
COPYLIB_ENSURE(is_valid(spec), "Invalid copy spec: {}", spec); // [optional] check if the copy spec is valid

// === 3. Manifesting the copy operations into an optimized parallel copy set

const copy_type type = copy_type::staged; // perform linearization
const copy_properties props = copy_properties::use_kernel; // use a kernel for linearization, generally faster
const d2d_implementation d2d = d2d_implementation::host_staging_at_source; // use manual host staging
const int64_t chunk_size = 1024*1024; // generate 1 MiB chunks
const copy_strategy strat(type, props, d2d, chunk_size); // create a strategy
const auto copy_set = manifest_strategy(spec, strat, basic_staging_provider{}); // manifest the copy set
COPYLIB_ENSURE(is_valid(copy_set), "Invalid copy set: {}", copy_set); // [optional] validate the copy set

// === 4. Executing the copy set

execute_copy(exec, copy_set);
```

## Benchmarks and Utilities

Not yet ported from the 2D library:

- `utils/info`: Print information about the execution environment and its features
- `benchmarks/manifest`: Micro-benchmark measuring strategy manifesting performance
- `benchmarks/intra_device`: Benchmark for intra-device linearization performnce
- `benchmarks/chunk_parallel`: Benchmark for optimized device-to-device copy performance
- `benchmarks/full_set`: Perform a very large run of various benchmarks to characterize platform performance
