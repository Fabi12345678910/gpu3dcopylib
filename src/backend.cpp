#include <copylib/backend.hpp>

#include <copylib/support.hpp> // IWYU pragma: keep - this is needed for formatting output, IWYU is dumb

#include <memory>
#include <string>
#include <thread>
#include <utility>

// Skeleton: every function below returns a default-constructed value so that calling it is well-defined while the
// implementation is missing. Functions returning a reference cannot do that and fail loudly instead.

namespace copylib {

std::string executor::get_sycl_impl_name() const { return {}; }

bool executor::is_device_to_device_copy_available() const { return {}; }

bool executor::is_peer_memory_access_available() const { return {}; }

int32_t executor::get_preferred_wg_size() const { return {}; }

int get_cpu_for_gpu_alloc(int gpu_idx, size_t total_gpu_count) { return {}; }

std::string executor::get_info() const { return {}; }

executor::possibility executor::can_copy(const copy_spec& spec) const { return {}; }

executor::possibility executor::can_copy(const parallel_copy_set& set) const { return {}; }

void executor::barrier() {}

executor::executor(int64_t buffer_size) : executor(buffer_size, sycl::device::get_devices(sycl::info::device_type::gpu).size(), 1) {}

namespace {

	// checked before the pool is built from it, which would start a thread per hardware thread for 0
	std::size_t checked_queues_per_device(int64_t queues_per_device) {
		COPYLIB_ENSURE(queues_per_device > 0, "Need at least one queue per device");
		return static_cast<std::size_t>(queues_per_device);
	}

} // namespace

executor::executor(int64_t buffer_size, int64_t devices_needed, int64_t queues_per_device)
    : buffer_size(buffer_size), pool(checked_queues_per_device(queues_per_device)) {}

device::device(sycl::device dev, const std::vector<sycl::queue>& queues) : dev(dev), queues(queues) {}

device::~device() {}

sycl::queue& executor::get_queue(device_id id, int64_t queue_idx) {
	COPYLIB_ERROR("executor::get_queue is not implemented");
	__builtin_unreachable();
}

sycl::queue& executor::get_queue(const target& tgt) {
	COPYLIB_ERROR("executor::get_queue is not implemented");
	__builtin_unreachable();
}

std::byte* executor::get_staging_buffer(device_id id) { return {}; }

std::byte* executor::get_host_staging_buffer(device_id id) { return {}; }

int64_t executor::get_buffer_size() const { return {}; }

int64_t executor::get_queues_per_device() const { return {}; }

template <typename CopyFun>
void copy_via_repeated_1D_copies(CopyFun fun, const data_layout& source_layout, const data_layout& target_layout) {}

namespace detail {

	step_result execute_copy(executor& exec, const copy_spec& spec, int64_t queue_idx, bool alternate_device, const step_result& last) { return {}; }

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

namespace detail {

	void execute_copy(executor& exec, const copy_plan& plan) {}

	struct copy_state {};

} // namespace detail

copy_handle::copy_handle(std::shared_ptr<detail::copy_state> state) : state(std::move(state)) {}

bool copy_handle::is_complete() const { return {}; }

void copy_handle::wait() const {}

std::optional<std::string> copy_handle::error() const { return {}; }

std::optional<std::chrono::nanoseconds> copy_handle::execution_time() const { return {}; }

copy_handle execute_copy(executor& exec, const parallel_copy_set& set) { return copy_handle(std::make_shared<detail::copy_state>()); }

} // namespace copylib
