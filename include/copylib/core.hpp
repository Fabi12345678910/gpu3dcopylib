#pragma once

#include "utils.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace copylib {

enum class device_id : int16_t {
	host = -1,
	d0 = 0,
	d1 = 1,
	d2 = 2,
	d3 = 3,
	d4 = 4,
	d5 = 5,
	d6 = 6,
	d7 = 7,
	count = 8,
};

#pragma pack(push, 0)
struct staging_id {
	constexpr static uint8_t staging_id_flag = 0b00000001;
	uint8_t is_staging_id = staging_id_flag;
	uint8_t on_host = false;
	device_id did = device_id::d0;
	uint32_t index = 0;

	staging_id() = default;
	staging_id(bool on_host, device_id did, uint32_t index);

	constexpr bool operator==(const staging_id& other) const = default;
	constexpr bool operator!=(const staging_id& other) const = default;
};
#pragma pack(pop)
static_assert(sizeof(staging_id) == sizeof(intptr_t));
static_assert(offsetof(staging_id, is_staging_id) == 0);

// 3D data layout used as the source or destination of a copy operation
// all values are in bytes; d0 is the innermost dimension and always contiguous, d2 the outermost
struct data_layout {
	union {
		intptr_t base = 0;
		staging_id staging;
	};

	int64_t d0_stride = 0; // size of one full row of the allocation
	int64_t d1_stride = 0; // size of one full plane of the allocation

	int64_t d0_start_offset = 0;
	int64_t d1_start_offset = 0;
	int64_t d2_start_offset = 0;

	int64_t d0_end_offset = 0;
	int64_t d1_end_offset = 0;
	int64_t d2_end_offset = 0;

	// half-open window of byte offsets into the packed box, i.e. with the gaps excluded, with 0 <= start < end <= total_bytes().
	// The constructors set it to cover the whole box; chunking keeps the box and narrows only the window.
	int64_t start = 0;
	int64_t end = 0;

	data_layout() = default;

	// a contiguous 1D layout of `length` bytes, starting `offset` bytes into the allocation
	data_layout(intptr_t base, int64_t offset, int64_t length);
	data_layout(intptr_t base, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset, int64_t d1_start_offset, int64_t d2_start_offset,
	    int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset);
	// the same box and window as `layout`, but in the allocation at `base`
	data_layout(intptr_t base, const data_layout& layout);

	data_layout(staging_id staging, int64_t offset, int64_t length);
	data_layout(staging_id staging, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset, int64_t d1_start_offset, int64_t d2_start_offset,
	    int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset);
	data_layout(staging_id staging, const data_layout& layout);

	// the same layout with the window narrowed to [start, end)
	[[nodiscard]] data_layout with_window(int64_t start, int64_t end) const;

	// the number of bytes this layout actually copies, i.e. the length of its window
	[[nodiscard]] constexpr int64_t window_length() const { return end - start; }

	// bytes covered by the box, excluding the gaps
	[[nodiscard]] constexpr int64_t total_bytes() const { /* TODO: implement */ return 0; }
	// offset just past the last byte of the box, relative to the allocation base, for bounds checks against a buffer size
	[[nodiscard]] constexpr int64_t end_offset() const { /* TODO: implement */ return 0; }

	// byte offset in the allocation of the byte `packed_offset` bytes into the box, with the gaps excluded.
	// This is the closed form the copy kernels need: byte i of a copy sits at offset_at(start + i).
	[[nodiscard]] constexpr int64_t offset_at(int64_t packed_offset) const { /* TODO: implement */ return 0; }

	// shape predicates, all defined over the window rather than over the whole box
	[[nodiscard]] constexpr bool is_contiguous() const { /* TODO: implement */ return false; }        // the window is a single contiguous run
	[[nodiscard]] constexpr bool contiguous_fragments() const { /* TODO: implement */ return false; } // rows are adjacent, so they can be collapsed
	[[nodiscard]] constexpr bool contiguous_layers() const { /* TODO: implement */ return false; }    // planes are adjacent, so they can be collapsed

	[[nodiscard]] constexpr bool is_unplaced_staging() const { /* TODO: implement */ return false; }
	[[nodiscard]] constexpr std::byte* base_ptr() const { /* TODO: implement */ return nullptr; }

	// Compares the encoding field by field, window included. Two layouts describing the same bytes with different strides
	// compare unequal; normalize both first to compare the bytes they describe.
	bool operator==(const data_layout& other) const;
	bool operator!=(const data_layout& other) const;
};

// Invokes f(offset_in_allocation, run_length) for each contiguous run of bytes covered by the layout's window.
// This is the iteration primitive for the copy paths, the host memcpy fallback, the coverage check in is_equivalent and
// the reference implementation in the tests. Per-fragment indexing cannot serve that purpose, because a window may
// start or end in the middle of a row, making its first and last runs partial.
template <typename F>
void for_each_contiguous_run(const data_layout& layout, F&& f) {
	(void)layout, (void)f; // TODO: implement
}

enum class copy_properties {
	none = 0x0000,
	use_kernel = 0x0001,  // whether to use a kernel to perform the copy
};
copy_properties operator|(copy_properties a, copy_properties b);
bool operator&(copy_properties a, copy_properties b);

// a copy specification describes a single copy operation from a source data layout and device to a destination data layout and device
struct copy_spec {
	device_id source_device;
	data_layout source_layout;
	device_id target_device;
	data_layout target_layout;

	copy_properties properties = copy_properties::none;

	copy_spec(device_id src_dev, const data_layout& src_layout, device_id tgt_dev, const data_layout& tgt_layout);
	copy_spec(device_id src_dev, const data_layout& src_layout, device_id tgt_dev, const data_layout& tgt_layout, copy_properties p);

	[[nodiscard]] bool is_contiguous() const;
	[[nodiscard]] copy_spec with_properties(copy_properties p) const;

	bool operator==(const copy_spec&) const = default;
	bool operator!=(const copy_spec&) const = default;
};

// a copy plan is a list of one or more copy specifications which need to be enacted subsequently to implement one semantic copy operation
using copy_plan = std::vector<copy_spec>;

// a parallel copy set is a set of independent copy plans which can be enacted concurrently
using parallel_copy_set = std::vector<copy_plan>;

// defines the strategy type used to copy data between memories
enum class copy_type {
	direct, // copy directly from source to destination using copy operations
	staged, // stage/unstage to a linearized buffer to perform the copy
};

// defines how to deal with device to device copy operations
enum class d2d_implementation {
	direct,                 // directly copy from device to device
	host_staging_at_source, // stage in host memory at the source device
	host_staging_at_target, // stage in host memory at the target device
	host_staging_at_both,   // stage in host memory at both devices, with extra copy operation
};

// defines the strategy used to copy data between memories
struct copy_strategy {
	copy_type type = copy_type::direct;
	copy_properties properties = copy_properties::none;
	d2d_implementation d2d = d2d_implementation::direct;
	int64_t chunk_size = 0; // the size of each chunk to split the copy into, in bytes; 0 means no chunking

	copy_strategy() = default;
	copy_strategy(copy_type t);
	copy_strategy(int64_t c);
	copy_strategy(copy_type t, copy_properties p);
	copy_strategy(copy_type t, copy_properties p, d2d_implementation d);
	copy_strategy(copy_type t, copy_properties p, int64_t c);
	copy_strategy(copy_type t, copy_properties p, d2d_implementation d, int64_t c);

	constexpr bool operator==(const copy_strategy&) const = default;
	constexpr bool operator!=(const copy_strategy&) const = default;
};

// validate whether a given data layout is sound
bool is_valid(const data_layout& layout);

// validate whether a given copy spec is sound
bool is_valid(const copy_spec& spec);

// validate whether a given copy plan is sound
bool is_valid(const copy_plan& plan);

// validate whether a given copy set is sound
bool is_valid(const parallel_copy_set& set);

// check whether a given copy plan implements a given copy specification
bool is_equivalent(const copy_plan& plan, const copy_spec& spec);

// check whether the given copy set implements the given copy specification
bool is_equivalent(const parallel_copy_set& set, const copy_spec& spec);

// Collapse densely packed dimensions into the unique normal form of the box (see "Normal form" in docs/design.md):
// a single contiguous run becomes a 1D layout, evenly spaced runs a 2D one, and anything else stays 3D.
// Two layouts describe the same box bytes and window exactly when their normal forms compare equal.
data_layout normalize(const data_layout& layout);

// Normalize each layout of a copy spec independently. Each side is read in its own packed order, so this preserves
// exactly what the spec copies.
copy_spec normalize(const copy_spec& spec);

// apply given properties to the given copy spec
copy_spec apply_properties(const copy_spec& spec, const copy_properties& props);

// apply chunking to the given copy spec if requested by the strategy
parallel_copy_set apply_chunking(const copy_spec& spec, const copy_strategy& strategy);

using staging_buffer_provider = std::function<staging_id(device_id, bool, int64_t)>;

class basic_staging_provider {
  public:
	staging_id operator()(device_id did, bool on_host, int64_t size);

  private:
	uint32_t next_staging_idx = 0;
};

// apply staging to the given spec if requested by the strategy
copy_plan apply_staging(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider);

// apply staging to each copy spec in the given parallel copy set if requested by the strategy
parallel_copy_set apply_staging(const parallel_copy_set& set, const copy_strategy& strategy, const staging_buffer_provider& staging_provider);

// apply the desired d2d implementation to the given copy plan
copy_plan apply_d2d_implementation(const copy_plan& plan, const d2d_implementation d2d, const staging_buffer_provider& staging_provider);

// apply the desired d2d implementation to the given parallel copy set (by applying it to each copy plan)
parallel_copy_set apply_d2d_implementation(const parallel_copy_set& set, const d2d_implementation d2d, const staging_buffer_provider& staging_provider);

// manifests the copy strategy on the given copy spec, applying chunking and staging as necessary
parallel_copy_set manifest_strategy(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider);

} // namespace copylib
