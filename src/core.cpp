#include <copylib/core.hpp>

#include <copylib/support.hpp> // IWYU pragma: keep

// Skeleton: every function below returns a default-constructed value so that calling it is well-defined while the
// implementation is missing. Without that, a non-void function falling off its end is undefined behaviour and the
// test binary crashes instead of reporting failing assertions. Replace the placeholder as each function is written.

namespace copylib {

bool is_valid(const data_layout& layout) {
    return layout.d0_stride > 0 && layout.d1_stride > 0
        && layout.d0_end_offset > layout.d0_start_offset
        && layout.d1_end_offset > layout.d1_start_offset
        && layout.d2_end_offset > layout.d2_start_offset
        && layout.end > layout.start
        && layout.d0_end_offset <= layout.d0_stride
        && layout.d1_end_offset <= layout.d1_stride
        //confirm end actually is inside the copy box
        && (layout.d2_end_offset - layout.d2_start_offset) * 
           (layout.d1_end_offset - layout.d1_start_offset) *
           (layout.d0_end_offset - layout.d0_start_offset) >= layout.end;
}

bool is_valid(const copy_spec& spec) {
    return is_valid(spec.source_layout) && is_valid(spec.target_layout)
        && spec.source_layout.total_bytes() == spec.target_layout.total_bytes();
}

bool is_valid(const copy_plan& plan) {
    for (const copy_spec& spec : plan) {
        if(!is_valid(spec)){return false;};
    }
    return true;
}

bool is_valid(const parallel_copy_set& set) { 
    for (const copy_plan& spec : set) {
        if(!is_valid(spec)){return false;};
    }
    return true;}

bool collapse_d1_onto_d0(data_layout& layout){
    if(!layout.d1_contigious()) return false;

    data_layout old_layout(layout);
    //collapse d1 onto d0
    layout.d0_stride = old_layout.d0_stride * old_layout.d1_stride;
    layout.d0_start_offset = old_layout.d1_start_offset * old_layout.d0_stride;
    layout.d0_end_offset = old_layout.d1_end_offset * old_layout.d0_stride;

    layout.d1_start_offset = old_layout.d2_start_offset;
    layout.d1_end_offset = old_layout.d2_end_offset;

    layout.d1_stride = 1;
    layout.d2_start_offset = 0;
    layout.d2_end_offset = 1;
    return true;
}

bool collapse_d2_onto_d1(data_layout& layout){
    if(!((layout.d1_start_offset == 0 && layout.d1_end_offset == layout.d1_stride))) return false;

    data_layout old_layout(layout);
    //collapse d1 onto d0
 //   layout.d0_stride = old_layout.d0_stride * old_layout.d1_stride;
 //   layout.d0_start_offset = old_layout.d1_start_offset * old_layout.d0_stride;
 //   layout.d0_end_offset = old_layout.d1_end_offset * old_layout.d0_stride;

    layout.d1_start_offset = old_layout.d2_start_offset * old_layout.d1_stride;
    layout.d1_end_offset = old_layout.d2_end_offset * old_layout.d1_stride;

    layout.d1_stride = 1;
    layout.d2_start_offset = 0;
    layout.d2_end_offset = 1;
    return true;
}

data_layout normalize(const data_layout& layout) {
    if(layout.d2_contigious()) return layout;
    
    data_layout new_layout(layout);
    collapse_d2_onto_d1(new_layout);
    collapse_d1_onto_d0(new_layout);
    return new_layout;
}

copy_spec normalize(const copy_spec& spec) {
    if(spec.source_layout.d2_contigious() && spec.target_layout.d2_contigious()) {
        return spec;
    }
    return {spec.source_device, normalize(spec.source_layout), spec.target_device, normalize(spec.target_layout), spec.properties};
}

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
