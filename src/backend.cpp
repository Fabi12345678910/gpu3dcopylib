#include <copylib/backend.hpp>

#include <copylib/support.hpp> // IWYU pragma: keep - this is needed for formatting output, IWYU is dumb

#ifdef SIMSYCL_VERSION
#include <simsycl/system.hh>
#endif

#if defined(__ACPP_ENABLE_CUDA_TARGET__) && defined(COPYLIB_CUDA)
#define ACPP_WITH_CUDA true
#else
#define ACPP_WITH_CUDA false
#endif

#if ACPP_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <string>
#include <thread>

namespace copylib {

std::string executor::get_sycl_impl_name() const {}

bool executor::is_2d_copy_available() const {}

bool executor::is_3d_copy_available() const {}

bool executor::is_device_to_device_copy_available() const {}

bool executor::is_peer_memory_access_available() const {}

int32_t executor::get_preferred_wg_size() const {}

int get_cpu_for_gpu_alloc(int gpu_idx, size_t total_gpu_count) {}

std::string executor::get_info() const {}

executor::possibility executor::can_copy(const copy_spec& spec) const {}

executor::possibility executor::can_copy(const parallel_copy_set& set) const {}

void executor::barrier() {}

executor::executor(int64_t buffer_size) : executor(buffer_size, sycl::device::get_devices(sycl::info::device_type::gpu).size(), 1) {}

executor::executor(int64_t buffer_size, int64_t devices_needed, int64_t queues_per_device) : buffer_size(buffer_size) {}

device::device(sycl::device dev, const std::vector<sycl::queue>& queues) : dev(dev), queues(queues) {}

device::~device() {}

sycl::queue& executor::get_queue(device_id id, int64_t queue_idx) {}

sycl::queue& executor::get_queue(const target& tgt) {}

std::byte* executor::get_buffer(device_id id) {}

std::byte* executor::get_staging_buffer(device_id id) {}

std::byte* executor::get_host_buffer(device_id id) {}

std::byte* executor::get_host_staging_buffer(device_id id) {}

int64_t executor::get_buffer_size() const {}

int64_t executor::get_queues_per_device() const {}

template <typename CopyFun>
void copy_via_repeated_1D_copies(CopyFun fun, const data_layout& source_layout, const data_layout& target_layout) {}

executor::target execute_copy(executor& exec, const copy_spec& spec, int64_t queue_idx, bool alternate_device, const executor::target last_target) {}

namespace detail {

	staging_fulfiller::staging_fulfiller(executor& exec) : exec(exec) {}

	void staging_fulfiller::fulfill(data_layout& layout) {}

	void staging_fulfiller::fulfill(copy_spec& spec) {}

} // namespace detail

namespace {

	class noop_fulfiller {
	  public:
		void fulfill(copy_spec&) {}
	};

	template <typename T>
	concept StagingFulfiller = requires(T f, copy_spec c) {
		{ f.fulfill(c) };
	};

	void execute_plan_impl(executor& exec, const copy_plan& plan, StagingFulfiller auto& fulfiller, int64_t queue_idx, bool alternate_device) {}

} // namespace

void execute_copy(executor& exec, const copy_plan& plan) {}

void execute_copy(executor& exec, const parallel_copy_set& set) {}

} // namespace copylib
