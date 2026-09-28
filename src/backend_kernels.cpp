#include <copylib/backend.hpp>

#include <copylib/support.hpp> // IWYU pragma: keep

namespace copylib::detail {

// Directly using CUDA threadIdx.x does NOT actually change performance
#define INDEX_X idx.get_global_id(0)

template <typename T, typename IdxType>
void copy_with_kernel_impl(sycl::queue& q, const copy_spec& spec, IdxType preferred_wg_size) {}

template <typename T>
void copy_with_kernel_impl(sycl::queue& q, const copy_spec& spec, int32_t preferred_wg_size) {}

sycl::event copy_with_kernel(sycl::queue& q, const copy_spec& spec, int32_t preferred_wg_size) { return {}; }

} // namespace copylib::detail
