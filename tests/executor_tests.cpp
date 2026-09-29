#include "backend_test_utils.hpp"

#include <copylib/backend.hpp>
#include <copylib/utils.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <utility>
#include <vector>

// Layer 12: constructing the executor and what it reports about itself.
//
// The executor creates and owns one in-order queue per device and queue index, on the devices and in the contexts it is
// given, and staging memory of the given size. It allocates no data buffers and does not configure the SYCL system; the
// harness does both (backend_test_utils.hpp).

using namespace copylib;
using namespace copylib_testing;
namespace ref = copylib_testing::reference_box;

using device_contexts = std::vector<std::pair<sycl::device, sycl::context>>;

TEST_CASE("an executor has in-order queues for every requested device and queue index", "[executor]") {
	const auto devices = test_devices(2);
	executor exec(test_staging_bytes, devices, 3);

	CHECK(exec.get_queues_per_device() == 3);
	for(size_t i = 0; i < devices.size(); ++i) {
		const auto did = static_cast<device_id>(i);
		for(int64_t q = 0; q < 3; ++q) {
			CAPTURE(did, q);
			CHECK(exec.get_queue(did, q).is_in_order());
			CHECK(exec.get_queue(did, q).get_device() == devices[i]);
		}
	}
	CHECK(exec.get_queue(device_id::d0, 0) != exec.get_queue(device_id::d0, 1));
}

TEST_CASE("an executor creates its queues in the context given with each device", "[executor]") {
	// one context per device, as Celerity creates them
	const auto devices = test_devices(2);
	const device_contexts contexts{{devices[0], sycl::context(devices[0])}, {devices[1], sycl::context(devices[1])}};
	executor exec(test_staging_bytes, contexts, 2);

	for(size_t i = 0; i < contexts.size(); ++i) {
		const auto did = static_cast<device_id>(i);
		for(int64_t q = 0; q < 2; ++q) {
			CAPTURE(did, q);
			CHECK(exec.get_queue(did, q).get_device() == contexts[i].first);
			CHECK(exec.get_queue(did, q).get_context() == contexts[i].second);
		}
	}

	// a staged copy within one device, on memory and staging buffers of that device's own context
	const prepared_copy copy(exec, on_device(device_id::d1), ref::fields(), on_device(device_id::d1), shapes::six_rows_of_48(0));
	const auto outcome = run_copy(exec, copy, strategy_from_fields(copy_type::staged, copy_properties::use_kernel, d2d_implementation::direct, 64));
	INFO(outcome.describe());
	CHECK(outcome.correct());
}

TEST_CASE("an executor rejects a device paired with a context that does not contain it", "[executor][error]") {
	configure_test_system();
	const auto all = sycl::device::get_devices();
	if(all.size() < 2) { SKIP("needs two distinct devices"); }

	CHECK_THROWS_AS(executor(test_staging_bytes, device_contexts{{all[0], sycl::context(all[1])}}, 1), copylib::error);
	CHECK_NOTHROW(executor(test_staging_bytes, device_contexts{{all[0], sycl::context(all[0])}}, 1));
}

TEST_CASE("the executor's buffer size is its staging memory per device", "[executor]") {
	// rounded up to a whole number of the 128-byte staging alignment
	const auto exec = make_executor(1, 1, 12345 * 64);
	CHECK(exec.get_buffer_size() == 12346 * 64);
}

TEST_CASE("an executor cannot have more devices than device_id can name", "[executor][error]") {
	CHECK_THROWS_AS(executor(test_staging_bytes, test_devices(9), 1), copylib::error);
	CHECK_NOTHROW(executor(test_staging_bytes, test_devices(8), 1));
}

TEST_CASE("the GPU shortcut cannot pick more GPUs than exist", "[executor][error]") {
	configure_test_system();
	// more than any machine running the tests
	CHECK_THROWS_AS(executor(test_staging_bytes, 64, 1), copylib::error);
}

TEST_CASE("an executor needs at least one device and one queue per device", "[executor][error]") {
	CHECK_THROWS_AS(executor(test_staging_bytes, std::vector<sycl::device>{}, 1), copylib::error);
	CHECK_THROWS_AS(executor(test_staging_bytes, test_devices(1), 0), copylib::error);
	CHECK_THROWS_AS(executor(test_staging_bytes, 0, 1), copylib::error);
	CHECK_NOTHROW(executor(test_staging_bytes, test_devices(1), 1));
}

TEST_CASE("an executor describes itself", "[executor]") {
	const auto exec = make_executor();
	CHECK_FALSE(exec.get_sycl_impl_name().empty());
	CHECK_FALSE(exec.get_info().empty());
}

TEST_CASE("failed checks throw copylib::error", "[executor][error]") {
	// COPYLIB_ENSURE is the library's single way of failing; it throws rather than ending the process
	CHECK_THROWS_AS([] { COPYLIB_ENSURE(1 + 1 == 3, "arithmetic is broken: {}", 1 + 1); }(), copylib::error);
	CHECK_NOTHROW([] { COPYLIB_ENSURE(1 + 1 == 2, "arithmetic is broken: {}", 1 + 1); }());
}
