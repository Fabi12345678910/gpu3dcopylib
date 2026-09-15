#pragma once

#include "core.hpp"

#include <sycl/sycl.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace copylib {

struct device {
	sycl::device dev;
	std::vector<sycl::queue> queues;
	std::byte* dev_buffer = nullptr;
	std::byte* staging_buffer = nullptr;
	std::byte* host_buffer = nullptr;
	std::byte* host_staging_buffer = nullptr;

	device(sycl::device dev, const std::vector<sycl::queue>& queues);

	~device();
};

using device_list = std::vector<device>;

class executor {
  public:
	struct target {
		device_id did;
		int64_t queue_idx;

		bool operator==(const target& other) const = default;
		bool operator!=(const target& other) const = default;
	};
	static constexpr target null_target = target{device_id::count, 0};

	executor(int64_t buffer_size);
	executor(int64_t buffer_size, int64_t devices_needed, int64_t queues_per_device = 1);

	sycl::queue& get_queue(device_id id, int64_t queue_idx = 0);
	sycl::queue& get_queue(const target& tgt);

	std::byte* get_buffer(device_id id);
	std::byte* get_staging_buffer(device_id id);
	std::byte* get_host_buffer(device_id id);
	std::byte* get_host_staging_buffer(device_id id);

	[[nodiscard]] int64_t get_buffer_size() const;
	[[nodiscard]] int64_t get_queues_per_device() const;

	[[nodiscard]] std::string get_sycl_impl_name() const;
	[[nodiscard]] bool is_2d_copy_available() const;
	[[nodiscard]] bool is_3d_copy_available() const;
	[[nodiscard]] bool is_device_to_device_copy_available() const;
	[[nodiscard]] bool is_peer_memory_access_available() const;
	[[nodiscard]] int32_t get_preferred_wg_size() const;
	[[nodiscard]] std::string get_info() const;

	enum class possibility {
		possible,
		needs_2d_copy,
		needs_3d_copy,
		needs_d2d_copy,
	};

	[[nodiscard]] possibility can_copy(const copy_spec& spec) const;
	[[nodiscard]] possibility can_copy(const parallel_copy_set& set) const;

	void barrier();

  private:
	mutable device_list devices; // Mutable due to ext_oneapi_can_access_peer not being const; very ugly
	std::vector<sycl::device> gpu_devices;
	int64_t buffer_size;
};

int get_cpu_for_gpu_alloc(int gpu_idx, size_t total_gpu_count);

namespace detail {

	void copy_with_kernel(sycl::queue& q, const copy_spec& spec, int32_t preferred_wg_size);

	class staging_fulfiller {
	  public:
		staging_fulfiller(executor& exec);

		void fulfill(data_layout& layout);
		void fulfill(copy_spec& spec);

	  private:
		struct staging_info {
			int64_t size = 0;
			device_id device = device_id::d0;
			bool on_host = false;
			std::byte* buffer = nullptr;
		};

		executor& exec;
		std::vector<int64_t> current_staging_offsets = std::vector<int64_t>(static_cast<int>(device_id::count), 0);
		std::vector<int64_t> current_host_staging_offsets = std::vector<int64_t>(static_cast<int>(device_id::count), 0);
		static constexpr int64_t staging_alignment = 128;

		std::unordered_map<decltype(staging_id::index), staging_info> staging_buffers;
	};

} // namespace detail

executor::target execute_copy(
    executor& exec, const copy_spec& spec, int64_t queue_idx = 0, bool alternate_device = false, const executor::target last_target = executor::null_target);

void execute_copy(executor& exec, const copy_plan& plan);

void execute_copy(executor& exec, const parallel_copy_set& set);

} // namespace copylib
