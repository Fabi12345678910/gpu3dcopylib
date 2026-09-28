#include <copylib/backend.hpp>

#include <copylib/support.hpp> // IWYU pragma: keep - this is needed for formatting output, IWYU is dumb

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <pthread.h>

// Skeleton: every function below returns a default-constructed value so that calling it is well-defined while the
// implementation is missing. Functions returning a reference cannot do that and fail loudly instead.

namespace copylib {

std::string executor::get_sycl_impl_name() const {
#if defined(SIMSYCL_VERSION)
	return "SimSYCL";
#elif defined(__ADAPTIVECPP__)
	std::string ret = "AdaptiveCPP";
#if defined(__ACPP_ENABLE_CUDA_TARGET__)
	ret += " (CUDA)";
#endif // __ACPP_ENABLE_CUDA_TARGET__
	return ret;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
	return "IntelSYCL";
#else
	return "Unknown SYCL implementation";
#endif
}

bool executor::is_device_to_device_copy_available() const {
#if defined(SIMSYCL_VERSION)
	return true;
#elif defined(__ADAPTIVECPP__)
#if defined(__ACPP_ENABLE_CUDA_TARGET__)
	return true; // CUDA emulates p2p transfers even if ont available
#else
	return false; // assumption for now
#endif
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
	if(gpu_devices.empty()) { return false; }
	return gpu_devices.front().get_info<sycl::info::device::vendor>().find("NVIDIA") != std::string::npos;
#else
	return false;
#endif
}

bool executor::is_peer_memory_access_available() const { return peer_access_available; }

int32_t executor::get_preferred_wg_size() const {
	thread_local int32_t wg_size = -1;
	if(wg_size == -1) {
		auto env_str = std::getenv("COPYLIB_WG_SIZE");
		if(env_str) {
			wg_size = std::stoi(env_str);
		} else {
			if(gpu_devices.empty()) { return 32; }
			if(gpu_devices.front().get_info<sycl::info::device::vendor>().find("Intel") != std::string::npos) {
				wg_size = 128;
			} else {
				wg_size = 32;
			}
		}
	}
	return wg_size;
}

int get_cpu_for_gpu_alloc(int gpu_idx, size_t total_gpu_count) {
	constexpr int max_gpu_idx = static_cast<int>(device_id::count);
	COPYLIB_ENSURE(gpu_idx < max_gpu_idx && gpu_idx >= 0, "Invalid gpu index: {} (needs to be >=0 and <{})", gpu_idx, max_gpu_idx);
	COPYLIB_ENSURE(total_gpu_count <= max_gpu_idx, "Invalid total gpu count: {} (needs to be <{})", total_gpu_count, max_gpu_idx);
	thread_local size_t initialized_for = 0; // the mapping depends on the total count, so it is cached per count
	thread_local std::array<int, max_gpu_idx> cpu_for_gpu;
	if(initialized_for != total_gpu_count) {
		auto env_var = std::getenv("COPYLIB_ALLOC_CPU_IDS");
		if(env_var) {
			const auto cpu_ids = std::string(env_var);
			const auto cpu_ids_split = utils::split(cpu_ids, ',');
			COPYLIB_ENSURE(cpu_ids_split.size() >= total_gpu_count, "Insufficient number of CPU IDs provided in COPYLIB_ALLOC_CPU_IDS: {} (expected {})",
			    cpu_ids_split.size(), total_gpu_count);
			for(size_t i = 0; i < total_gpu_count; i++) {
				cpu_for_gpu[i] = std::stoi(cpu_ids_split[i]);
			}
		} else { // guess
			const auto hw_concurrency = std::thread::hardware_concurrency();
			const auto cores = hw_concurrency / 2; // we just assume 2 threads per core
			for(size_t i = 0; i < total_gpu_count; i++) {
				cpu_for_gpu[i] = cores / total_gpu_count * i;
			}
		}
		initialized_for = total_gpu_count;
	}
	return cpu_for_gpu[gpu_idx];
}

std::string executor::get_info() const {
	auto ret = utils::format("Copylib executor with {} device(s) and buffer size {} bytes\n", devices.size(), buffer_size);
	ret += utils::format("SYCL implementation: {}\n", get_sycl_impl_name());
	ret += utils::format("D2D copy: {}    Peer access: {}    Preferred wg size: {}\n", //
	    is_device_to_device_copy_available(), is_peer_memory_access_available(), get_preferred_wg_size());
	ret += utils::format("Using {} queues per device\n", get_queues_per_device());
	for(size_t i = 0; i < devices.size(); i++) {
		ret += utils::format("    Device {:2}: {} [{}]", i, //
		    gpu_devices[i].get_info<sycl::info::device::name>(), gpu_devices[i].get_info<sycl::info::device::vendor>());
		ret += utils::format(" (host alloc on core {})\n", get_cpu_for_gpu_alloc(i, devices.size()));
	}
	return ret;
}

executor::possibility executor::can_copy(const copy_spec& spec) const {
	const bool d2d = is_device_to_device_copy_available();
	const auto d2d_copy = spec.source_device != spec.target_device && spec.source_device != device_id::host && spec.target_device != device_id::host;
	if(d2d_copy) {
		if(spec.properties & copy_properties::use_kernel) { return possibility::needs_d2d_copy; } // TODO need to be more specific here later
		if(!d2d) { return possibility::needs_d2d_copy; }
	}
	return possibility::possible;
}

executor::possibility executor::can_copy(const parallel_copy_set& cset) const {
	for(const auto& plan : cset) {
		for(const auto& spec : plan) {
			const auto res = can_copy(spec);
			if(res != possibility::possible) { return res; }
		}
	}
	return possibility::possible;
}

void executor::barrier() {
	for(auto& dev : devices) {
		for(auto& q : dev.queues) {
			q.wait_and_throw();
		}
	}
}

executor::executor(int64_t buffer_size) : executor(buffer_size, sycl::device::get_devices(sycl::info::device_type::gpu).size(), 1) {}

namespace {

	// checked before the pool is built from it, which would start a thread per hardware thread for 0
	std::size_t checked_queues_per_device(int64_t queues_per_device) {
		COPYLIB_ENSURE(queues_per_device > 0, "Need at least one queue per device");
		return static_cast<std::size_t>(queues_per_device);
	}

	// enables peer access between every pair of devices, and reports whether every pair supports it
	bool enable_peer_access([[maybe_unused]] device_list& devices) {
#if defined(SIMSYCL_VERSION)
		return true;
#elif defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
		for(size_t dev_idx_a = 0; dev_idx_a < devices.size(); dev_idx_a++) {
			for(size_t dev_idx_b = 0; dev_idx_b < devices.size(); dev_idx_b++) {
				if(dev_idx_a == dev_idx_b) { continue; }
				if(devices[dev_idx_a].dev.ext_oneapi_can_access_peer(devices[dev_idx_b].dev)) {
					devices[dev_idx_a].dev.ext_oneapi_enable_peer_access(devices[dev_idx_b].dev);
				} else {
					return false;
				}
			}
		}
		return true;
#else
		return false;
#endif
	}

} // namespace

executor::executor(int64_t buffer_size, int64_t devices_needed, int64_t queues_per_device)
    : buffer_size(buffer_size), pool(checked_queues_per_device(queues_per_device)) {
	COPYLIB_ENSURE(devices_needed > 0, "Need at least one device");
	COPYLIB_ENSURE(devices_needed <= static_cast<int64_t>(device_id::count), "Too many devices requested: {} (at most {})", devices_needed,
	    static_cast<int>(device_id::count));

	gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
	if(gpu_devices.size() < static_cast<size_t>(devices_needed)) {
		COPYLIB_ERROR("Not enough GPU devices available: {} ({} needed)", gpu_devices.size(), devices_needed);
	} else if(gpu_devices.size() > static_cast<size_t>(devices_needed)) {
		gpu_devices.resize(devices_needed); // don't waste time initializing more devices than needed
	}
	cpu_set_t prior_mask;
	CPU_ZERO(&prior_mask);
	COPYLIB_ENSURE(pthread_getaffinity_np(pthread_self(), sizeof(prior_mask), &prior_mask) == 0, "Failed to get CPU affinity");
	// restores the affinity however the constructor is left, so a failed check cannot leave the caller pinned to one core
	struct affinity_guard {
		cpu_set_t mask;
		~affinity_guard() { pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask); }
	} const restore_affinity{prior_mask};

	// a whole number of alignments, which aligned allocations require on some implementations; reported as the buffer size
	this->buffer_size = (buffer_size + detail::staging_alignment - 1) / detail::staging_alignment * detail::staging_alignment;
	const auto staging_bytes = static_cast<size_t>(this->buffer_size);

	// allocate queues and staging buffers
	int dev_id = 0;
	for(const auto& device : gpu_devices) {
		const sycl::property_list queue_properties = {
		    sycl::property::queue::in_order{},
#ifdef ACPP_EXT_COARSE_GRAINED_EVENTS
		    // minor perf improvement on AdaptiveCPP
		    sycl::property::queue::AdaptiveCpp_coarse_grained_events{},
#endif // ACPP_EXT_COARSE_GRAINED_EVENTS
#ifdef SYCL_EXT_INTEL_QUEUE_IMMEDIATE_COMMAND_LIST
		    // ~ 10% perf improvement on Intel GPUs in strided chunk copy set peak performance
		    sycl::ext::intel::property::queue::immediate_command_list{},
#endif // SYCL_EXT_INTEL_QUEUE_IMMEDIATE_COMMAND_LIST
		};

		std::vector<sycl::queue> queues;
		for(int64_t i = 0; i < queues_per_device; i++) {
			queues.emplace_back(sycl::queue(device, queue_properties));
		}
		// allocated only once the device is listed, so that its destructor frees them if a later check throws
		auto& dev = devices.emplace_back(device, queues);
		auto& q = dev.queues[0];

		dev.staging_buffer = sycl::aligned_alloc_device<std::byte>(detail::staging_alignment, staging_bytes, q);
		COPYLIB_ENSURE(dev.staging_buffer != nullptr, "Failed to allocate device staging buffer");

		cpu_set_t mask_for_device;
		CPU_ZERO(&mask_for_device);
		const auto cpu_id = get_cpu_for_gpu_alloc(dev_id, gpu_devices.size());
		CPU_SET(cpu_id, &mask_for_device);
		COPYLIB_ENSURE(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask_for_device) == 0, "Failed to set CPU affinity");

		dev.host_staging_buffer = sycl::aligned_alloc_host<std::byte>(detail::staging_alignment, staging_bytes, q);
		COPYLIB_ENSURE(dev.host_staging_buffer != nullptr, "Failed to allocate host staging buffer");
		std::memset(dev.host_staging_buffer, 0, staging_bytes); // first touch, placing the pages close to the CPU chosen above

		dev_id++;
	}
	peer_access_available = enable_peer_access(devices);
}

device::device(sycl::device dev, const std::vector<sycl::queue>& queues) : dev(dev), queues(queues) {}

device::~device() {
	for(auto& q : queues) {
		q.wait(); // don't throw in the deconstructor
	}

	auto& q = queues[0];
	sycl::free(staging_buffer, q);
	sycl::free(host_staging_buffer, q);
}

sycl::queue& executor::get_queue(device_id id, int64_t queue_idx) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	auto& queues = devices[static_cast<int>(id)].queues;
	COPYLIB_ENSURE(queue_idx >= 0 && static_cast<size_t>(queue_idx) < queues.size(), "Invalid queue idx: {} ({} queue(s) available)", queue_idx, queues.size());
	return queues[queue_idx];
}

sycl::queue& executor::get_queue(const target& tgt) { return get_queue(tgt.did, tgt.queue_idx); }

std::byte* executor::get_staging_buffer(device_id id) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	return devices[static_cast<int>(id)].staging_buffer;
}

std::byte* executor::get_host_staging_buffer(device_id id) {
	COPYLIB_ENSURE(static_cast<int>(id) >= 0 && static_cast<size_t>(id) < devices.size(), "Invalid device id: {} ({} device(s) available)", id, devices.size());
	return devices[static_cast<int>(id)].host_staging_buffer;
}

int64_t executor::get_buffer_size() const { return buffer_size; }

int64_t executor::get_queues_per_device() const { return devices.front().queues.size(); }

namespace detail {

	step_result execute_copy(executor& exec, const copy_spec& spec, int64_t queue_idx, bool alternate_device, const step_result& last) { return {}; }

	staging_fulfiller::staging_fulfiller(executor& exec) : exec(exec) {}

	void staging_fulfiller::fulfill(data_layout& layout) {
		if(!layout.is_unplaced_staging()) {return;}
		const auto staging_idx = layout.staging.index;
		auto staging_it = staging_buffers.find(staging_idx);
		if(staging_it == staging_buffers.end()) {
			const auto did = layout.staging.did;
			const bool host = layout.staging.on_host;
			COPYLIB_ENSURE(did != device_id::host, "Device id for staging cannot be host");
			staging_info info{
				.size = layout.window_length(),
				.device = did,
				.on_host = host,
			};
			if(host) {
				info.buffer = exec.get_host_staging_buffer(did) + current_host_staging_offsets[static_cast<size_t>(did)];
				current_host_staging_offsets[static_cast<size_t>(did)] += (info.size + staging_alignment - 1) / staging_alignment * staging_alignment;
				COPYLIB_ENSURE(current_host_staging_offsets[static_cast<size_t>(did)] <= exec.get_buffer_size(),
					"Staging buffer overflow on host for device {}", static_cast<int>(did));
			} else {
				info.buffer = exec.get_staging_buffer(did) + current_staging_offsets[static_cast<size_t>(did)];
				current_staging_offsets[static_cast<size_t>(did)] += (info.size + staging_alignment - 1) / staging_alignment * staging_alignment;
				COPYLIB_ENSURE(current_staging_offsets[static_cast<size_t>(did)] <= exec.get_buffer_size(), "Staging buffer overflow for device {}",
					static_cast<int>(did));
			}
			staging_it = staging_buffers.emplace(staging_idx, info).first;
		} else {
			COPYLIB_ENSURE(staging_buffers[staging_idx].size == layout.window_length(), "Staging buffer size mismatch");
			COPYLIB_ENSURE(staging_buffers[staging_idx].device == layout.staging.did, "Staging buffer device mismatch");
			COPYLIB_ENSURE(staging_buffers[staging_idx].on_host == layout.staging.on_host, "Staging buffer host flag mismatch");
		}
		layout.base = reinterpret_cast<intptr_t>(staging_it->second.buffer);
	}

	void staging_fulfiller::fulfill(copy_spec& spec) {
		fulfill(spec.source_layout);
		fulfill(spec.target_layout);
	}

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
