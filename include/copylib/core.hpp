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

	int64_t d0_stride = 0;
	int64_t d1_stride = 0;
	
	int64_t d0_start_offset = 0;
	int64_t d1_start_offset = 0;
	int64_t d2_start_offset = 0;

	int64_t d0_end_offset = 0;
	int64_t d1_end_offset = 0;
	int64_t d2_end_offset = 0;

	data_layout();
	data_layout(intptr_t base, int64_t offset, int64_t fragment_length);
	data_layout(intptr_t base, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset, int64_t d1_start_offset, int64_t d2_start_offset,
	    int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset);
	data_layout(intptr_t base, const data_layout& layout);

	data_layout(staging_id staging, int64_t offset, int64_t fragment_length);
	data_layout(staging_id staging, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset, int64_t d1_start_offset, int64_t d2_start_offset,
	    int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset);
	data_layout(staging_id staging, const data_layout& layout);

	// these should be moved into the header and marked constexpr (as in the 2D library) once implemented
	[[nodiscard]] int64_t total_bytes() const;   // bytes covered, excluding the gaps
	[[nodiscard]] int64_t total_extent() const;  // bytes spanned, including the gaps
	[[nodiscard]] int64_t fragment_length() const;
	[[nodiscard]] int64_t fragment_count() const;
	[[nodiscard]] int64_t layer_count() const;
	[[nodiscard]] bool contiguous_fragments() const;
	[[nodiscard]] bool contiguous_layers() const;
	[[nodiscard]] bool unit_stride() const;
	[[nodiscard]] int32_t dimensions() const;
	[[nodiscard]] int64_t fragment_offset(int64_t layer, int64_t fragment) const;
	[[nodiscard]] int64_t layer_offset(int64_t layer) const;
	[[nodiscard]] int64_t end_offset() const;
	[[nodiscard]] bool is_unplaced_staging() const;
	[[nodiscard]] std::byte* base_ptr() const;

	bool operator==(const data_layout& other) const;
	bool operator!=(const data_layout& other) const;
};

enum class copy_properties {
	none = 0x0000,
	use_kernel = 0x0001,  // whether to use a kernel to perform the copy
	use_2D_copy = 0x0010, // whether to use a native 2D copy operation, if available
	use_3D_copy = 0x0100, // whether to use a native 3D copy operation, if available
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

// collapse dimensions which are densely packed, i.e. turn a contiguous 3D layout into a 2D or 1D one
data_layout normalize(const data_layout& layout);

// collapse dimensions in both layouts of a copy spec, as far as both of them allow
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
