#include <copylib/core.hpp>

#include <copylib/support.hpp> // IWYU pragma: keep

// Skeleton: every function below returns a default-constructed value so that calling it is well-defined while the
// implementation is missing. Without that, a non-void function falling off its end is undefined behaviour and the
// test binary crashes instead of reporting failing assertions. Replace the placeholder as each function is written.

namespace copylib {

staging_id::staging_id(bool on_host, device_id did, uint32_t index) {}

data_layout::data_layout(intptr_t base, int64_t offset, int64_t length) {}

data_layout::data_layout(intptr_t base, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset, int64_t d1_start_offset, int64_t d2_start_offset,
    int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset) {}

data_layout::data_layout(intptr_t base, const data_layout& layout) {}

data_layout::data_layout(staging_id staging, int64_t offset, int64_t length) {}

data_layout::data_layout(staging_id staging, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset, int64_t d1_start_offset, int64_t d2_start_offset,
    int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset) {}

data_layout::data_layout(staging_id staging, const data_layout& layout) {}

data_layout data_layout::with_window(int64_t start, int64_t end) const { return {}; }

bool data_layout::operator==(const data_layout& other) const { return {}; }

bool data_layout::operator!=(const data_layout& other) const { return {}; }

copy_properties operator|(copy_properties a, copy_properties b) { return {}; }

bool operator&(copy_properties a, copy_properties b) { return {}; }

copy_spec::copy_spec(device_id src_dev, const data_layout& src_layout, device_id tgt_dev, const data_layout& tgt_layout)
    : source_device(src_dev), target_device(tgt_dev) {}

copy_spec::copy_spec(device_id src_dev, const data_layout& src_layout, device_id tgt_dev, const data_layout& tgt_layout, copy_properties p)
    : source_device(src_dev), target_device(tgt_dev) {}

bool copy_spec::is_contiguous() const { return {}; }

// copy_spec has no default constructor, so the placeholder returns the input unchanged
copy_spec copy_spec::with_properties(copy_properties p) const { return *this; }

copy_strategy::copy_strategy(copy_type t) {}

copy_strategy::copy_strategy(int64_t c) {}

copy_strategy::copy_strategy(copy_type t, copy_properties p) {}

copy_strategy::copy_strategy(copy_type t, copy_properties p, d2d_implementation d) {}

copy_strategy::copy_strategy(copy_type t, copy_properties p, int64_t c) {}

copy_strategy::copy_strategy(copy_type t, copy_properties p, d2d_implementation d, int64_t c) {}

bool is_valid(const data_layout& layout) { return {}; }

bool is_valid(const copy_spec& spec) { return {}; }

bool is_valid(const copy_plan& plan) { return {}; }

bool is_valid(const parallel_copy_set& set) { return {}; }

bool is_equivalent(const copy_plan& plan, const copy_spec& spec) { return {}; }

bool is_equivalent(const parallel_copy_set& set, const copy_spec& spec) { return {}; }

data_layout normalize(const data_layout& layout) { return {}; }

// copy_spec has no default constructor, so the placeholder returns the input unchanged
copy_spec normalize(const copy_spec& spec) { return spec; }

copy_spec apply_properties(const copy_spec& spec, const copy_properties& props) { return spec; }

parallel_copy_set apply_chunking(const copy_spec& spec, const copy_strategy& strategy) { return {}; }

staging_id basic_staging_provider::operator()(device_id did, bool on_host, int64_t size) { return {}; }

copy_plan apply_staging(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) { return {}; }

parallel_copy_set apply_staging(const parallel_copy_set& set, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) { return {}; }

copy_plan apply_d2d_implementation(const copy_plan& plan, const d2d_implementation d2d, const staging_buffer_provider& staging_provider) { return {}; }

parallel_copy_set apply_d2d_implementation(const parallel_copy_set& set, const d2d_implementation d2d, const staging_buffer_provider& staging_provider) {
	return {};
}

parallel_copy_set manifest_strategy(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) { return {}; }

} // namespace copylib
