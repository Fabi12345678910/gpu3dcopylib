#include <copylib/core.hpp>

#include <copylib/support.hpp> // IWYU pragma: keep

namespace copylib {

staging_id::staging_id(bool on_host, device_id did, uint32_t index) {}

data_layout::data_layout() {}

data_layout::data_layout(intptr_t base, int64_t offset, int64_t fragment_length) {}

data_layout::data_layout(intptr_t base, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset, int64_t d1_start_offset, int64_t d2_start_offset,
    int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset) {}

data_layout::data_layout(intptr_t base, const data_layout& layout) {}

data_layout::data_layout(staging_id staging, int64_t offset, int64_t fragment_length) {}

data_layout::data_layout(staging_id staging, int64_t d0_stride, int64_t d1_stride, int64_t d0_start_offset, int64_t d1_start_offset, int64_t d2_start_offset,
    int64_t d0_end_offset, int64_t d1_end_offset, int64_t d2_end_offset) {}

data_layout::data_layout(staging_id staging, const data_layout& layout) {}

int64_t data_layout::total_bytes() const {}

int64_t data_layout::total_extent() const {}

int64_t data_layout::fragment_length() const {}

int64_t data_layout::fragment_count() const {}

int64_t data_layout::layer_count() const {}

bool data_layout::contiguous_fragments() const {}

bool data_layout::contiguous_layers() const {}

bool data_layout::unit_stride() const {}

int32_t data_layout::dimensions() const {}

int64_t data_layout::fragment_offset(int64_t layer, int64_t fragment) const {}

int64_t data_layout::layer_offset(int64_t layer) const {}

int64_t data_layout::end_offset() const {}

bool data_layout::is_unplaced_staging() const {}

std::byte* data_layout::base_ptr() const {}

bool data_layout::operator==(const data_layout& other) const {}

bool data_layout::operator!=(const data_layout& other) const {}

copy_properties operator|(copy_properties a, copy_properties b) {}

bool operator&(copy_properties a, copy_properties b) {}

copy_spec::copy_spec(device_id src_dev, const data_layout& src_layout, device_id tgt_dev, const data_layout& tgt_layout)
    : source_device(src_dev), target_device(tgt_dev) {}

copy_spec::copy_spec(device_id src_dev, const data_layout& src_layout, device_id tgt_dev, const data_layout& tgt_layout, copy_properties p)
    : source_device(src_dev), target_device(tgt_dev) {}

bool copy_spec::is_contiguous() const {}

copy_spec copy_spec::with_properties(copy_properties p) const {}

copy_strategy::copy_strategy(copy_type t) {}

copy_strategy::copy_strategy(int64_t c) {}

copy_strategy::copy_strategy(copy_type t, copy_properties p) {}

copy_strategy::copy_strategy(copy_type t, copy_properties p, d2d_implementation d) {}

copy_strategy::copy_strategy(copy_type t, copy_properties p, int64_t c) {}

copy_strategy::copy_strategy(copy_type t, copy_properties p, d2d_implementation d, int64_t c) {}

bool is_valid(const data_layout& layout) {}

bool is_valid(const copy_spec& spec) {}

bool is_valid(const copy_plan& plan) {}

bool is_valid(const parallel_copy_set& set) {}

bool is_equivalent(const copy_plan& plan, const copy_spec& spec) {}

bool is_equivalent(const parallel_copy_set& set, const copy_spec& spec) {}

data_layout normalize(const data_layout& layout) {}

copy_spec normalize(const copy_spec& spec) {}

copy_spec apply_properties(const copy_spec& spec, const copy_properties& props) {}

parallel_copy_set apply_chunking(const copy_spec& spec, const copy_strategy& strategy) {}

staging_id basic_staging_provider::operator()(device_id did, bool on_host, int64_t size) {}

copy_plan apply_staging(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) {}

parallel_copy_set apply_staging(const parallel_copy_set& set, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) {}

copy_plan apply_d2d_implementation(const copy_plan& plan, const d2d_implementation d2d, const staging_buffer_provider& staging_provider) {}

parallel_copy_set apply_d2d_implementation(const parallel_copy_set& set, const d2d_implementation d2d, const staging_buffer_provider& staging_provider) {}

parallel_copy_set manifest_strategy(const copy_spec& spec, const copy_strategy& strategy, const staging_buffer_provider& staging_provider) {}

} // namespace copylib
