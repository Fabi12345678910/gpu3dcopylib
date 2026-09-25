#include "backend_test_utils.hpp"

#include <copylib/backend.hpp>
#include <copylib/utils.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

// Layer 12: constructing the executor and what it reports about itself.
//
// The executor creates and owns one in-order queue per device and queue index, and staging memory of the given size.
// It allocates no data buffers and does not configure the SYCL system; the harness does both (backend_test_utils.hpp).

using namespace copylib;
using namespace copylib_testing;

TEST_CASE("an executor has in-order queues for every requested device and queue index", "[executor][!mayfail]") {
	auto exec = make_executor(2, 3);

	CHECK(exec.get_queues_per_device() == 3);
	for(const auto did : {device_id::d0, device_id::d1}) {
		for(int64_t q = 0; q < 3; ++q) {
			CAPTURE(did, q);
			CHECK(exec.get_queue(did, q).is_in_order());
		}
	}
	CHECK(exec.get_queue(device_id::d0).get_device() != exec.get_queue(device_id::d1).get_device());
	CHECK(exec.get_queue(device_id::d0, 0) != exec.get_queue(device_id::d0, 1));
}

TEST_CASE("the executor's buffer size is its staging memory per device", "[executor][!mayfail]") {
	const auto exec = make_executor(1, 1, 12345 * 64);
	CHECK(exec.get_buffer_size() == 12345 * 64);
}

TEST_CASE("an executor cannot have more devices than exist", "[executor][error][!mayfail]") {
	configure_test_system();
	// more devices than any machine running the tests, and more than device_id can name
	CHECK_THROWS_AS(executor(test_staging_bytes, 64, 1), copylib::error);
	CHECK_NOTHROW(executor(test_staging_bytes, 1, 1));
}

TEST_CASE("an executor needs at least one device and one queue per device", "[executor][error][!mayfail]") {
	configure_test_system();
	CHECK_THROWS_AS(executor(test_staging_bytes, 0, 1), copylib::error);
	CHECK_THROWS_AS(executor(test_staging_bytes, 1, 0), copylib::error);
	CHECK_NOTHROW(executor(test_staging_bytes, 1, 1));
}

TEST_CASE("an executor describes itself", "[executor][!mayfail]") {
	const auto exec = make_executor();
	CHECK_FALSE(exec.get_sycl_impl_name().empty());
	CHECK_FALSE(exec.get_info().empty());
}

TEST_CASE("failed checks throw copylib::error", "[executor][error]") {
	// COPYLIB_ENSURE is the library's single way of failing; it throws rather than ending the process
	CHECK_THROWS_AS([] { COPYLIB_ENSURE(1 + 1 == 3, "arithmetic is broken: {}", 1 + 1); }(), copylib::error);
	CHECK_NOTHROW([] { COPYLIB_ENSURE(1 + 1 == 2, "arithmetic is broken: {}", 1 + 1); }());
}
