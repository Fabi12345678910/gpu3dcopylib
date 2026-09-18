#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>

// Layer 4: is_equivalent(), the tiling check that decides whether a plan or set implements a spec.
//
// The 2D suite used is_equivalent as the oracle for all planning tests without ever testing its negative cases, so a
// broken is_equivalent would have let every planning test pass. Here each case states the expected verdict twice: once
// for the library, and once for the independent oracle `implements()` from test_utils.hpp, which symbolically executes
// the plan. The second check guards the test itself.

using namespace copylib;
using namespace copylib_testing;
namespace ref = copylib_testing::reference_box;

namespace {

// the part [start, end) of `spec`, copied directly
copy_plan chunk(const copy_spec& spec, int64_t start, int64_t end) {
	return {spec_from_fields(spec.source_device, with_window_fields(spec.source_layout, start, end), spec.target_device,
	    with_window_fields(spec.target_layout, start, end))};
}

// the part [start, end) of `spec`, gathered into a staging buffer on the source device and scattered from there
copy_plan staged_chunk(const copy_spec& spec, staging_id staging, int64_t start, int64_t end) {
	const auto buffer = staging_layout_from_fields(staging, end - start);
	return {
	    spec_from_fields(spec.source_device, with_window_fields(spec.source_layout, start, end), spec.source_device, buffer),
	    spec_from_fields(spec.source_device, buffer, spec.target_device, with_window_fields(spec.target_layout, start, end)),
	};
}

staging_id staging(uint32_t index) { return staging_id_from_fields(false, device_id::d0, index); }

} // namespace

TEST_CASE("a spec is implemented by a plan consisting of itself", "[equivalence][!mayfail]") {
	const auto spec = ref::spec();

	REQUIRE(implements(copy_plan{spec}, spec));
	CHECK(is_equivalent(copy_plan{spec}, spec));
	CHECK(is_equivalent(parallel_copy_set{copy_plan{spec}}, spec));
}

TEST_CASE("an empty plan or set implements nothing", "[equivalence][!mayfail]") {
	const auto spec = ref::spec();

	CHECK(is_equivalent(copy_plan{spec}, spec));
	CHECK_FALSE(is_equivalent(copy_plan{}, spec));
	CHECK_FALSE(is_equivalent(parallel_copy_set{}, spec));
}

TEST_CASE("a staged plan is equivalent only with its steps in order", "[equivalence][staging][!mayfail]") {
	const auto spec = ref::spec();
	const auto good = staged_chunk(spec, staging(0), 0, ref::total_bytes);
	const copy_plan reversed{good[1], good[0]};

	REQUIRE(implements(good, spec));
	REQUIRE_FALSE(implements(reversed, spec));

	CHECK(is_equivalent(good, spec));
	CHECK_FALSE(is_equivalent(reversed, spec));
}

TEST_CASE("a plan must deliver the bytes to the right place", "[equivalence][!mayfail]") {
	const auto spec = ref::spec();
	REQUIRE(implements(copy_plan{spec}, spec));
	CHECK(is_equivalent(copy_plan{spec}, spec));

	SECTION("target window shifted by one row") {
		const copy_plan shifted{spec_from_fields(device_id::d0, with_window_fields(spec.source_layout, 0, 264), device_id::d1,
		    with_window_fields(spec.target_layout, 24, 288))};
		REQUIRE_FALSE(implements(shifted, spec));
		CHECK_FALSE(is_equivalent(shifted, spec));
	}

	SECTION("staged plan scattering to a different allocation") {
		auto misdirected = staged_chunk(spec, staging(0), 0, ref::total_bytes);
		misdirected[1].target_layout.base = 0x90000;
		REQUIRE_FALSE(implements(misdirected, spec));
		CHECK_FALSE(is_equivalent(misdirected, spec));
	}

	SECTION("right allocation, wrong device") {
		auto wrong_device = spec;
		wrong_device.target_device = device_id::d2;
		REQUIRE_FALSE(implements(copy_plan{wrong_device}, spec));
		CHECK_FALSE(is_equivalent(copy_plan{wrong_device}, spec));
	}
}

TEST_CASE("the chunks of a set must tile the window", "[equivalence][chunking][!mayfail]") {
	const auto spec = ref::spec();
	const parallel_copy_set tiled{chunk(spec, 0, 96), chunk(spec, 96, 192), chunk(spec, 192, 288)};

	REQUIRE(implements(tiled, spec));
	CHECK(is_equivalent(tiled, spec));

	SECTION("a missing chunk") {
		const parallel_copy_set gap{chunk(spec, 0, 96), chunk(spec, 192, 288)};
		REQUIRE_FALSE(implements(gap, spec));
		CHECK_FALSE(is_equivalent(gap, spec));
	}

	SECTION("overlapping chunks, even though the resulting bytes are right") {
		const parallel_copy_set overlap{chunk(spec, 0, 120), chunk(spec, 96, 288)};
		REQUIRE_FALSE(implements(overlap, spec));
		CHECK_FALSE(is_equivalent(overlap, spec));
	}

	SECTION("a duplicated chunk") {
		const parallel_copy_set duplicate{chunk(spec, 0, 96), chunk(spec, 96, 192), chunk(spec, 192, 288), chunk(spec, 96, 192)};
		REQUIRE_FALSE(implements(duplicate, spec));
		CHECK_FALSE(is_equivalent(duplicate, spec));
	}

	SECTION("an extra plan writing outside the spec") {
		auto elsewhere = spec;
		elsewhere.target_layout.base = 0x90000;
		auto stray = tiled;
		stray.push_back(copy_plan{elsewhere});
		REQUIRE_FALSE(implements(stray, spec));
		CHECK_FALSE(is_equivalent(stray, spec));
	}
}

TEST_CASE("the order of plans in a set does not matter", "[equivalence][!mayfail]") {
	const auto spec = ref::spec();
	const parallel_copy_set shuffled{chunk(spec, 192, 288), chunk(spec, 0, 96), chunk(spec, 96, 192)};

	REQUIRE(implements(shuffled, spec));
	CHECK(is_equivalent(shuffled, spec));
}

TEST_CASE("concurrent plans may not share a staging buffer", "[equivalence][staging][!mayfail]") {
	const auto spec = ref::spec();
	const parallel_copy_set separate{staged_chunk(spec, staging(0), 0, 144), staged_chunk(spec, staging(1), 144, 288)};
	const parallel_copy_set shared{staged_chunk(spec, staging(0), 0, 144), staged_chunk(spec, staging(0), 144, 288)};

	REQUIRE(implements(separate, spec));
	REQUIRE_FALSE(implements(shared, spec));

	CHECK(is_equivalent(separate, spec));
	CHECK_FALSE(is_equivalent(shared, spec));
}

TEST_CASE("a reshaping copy can be implemented through staging", "[equivalence][staging][!mayfail]") {
	// int[1,1,6] -> int[1,2,3] from design.md
	const auto spec = spec_from_fields(device_id::d0, shapes::one_row_of_six(0x10000), device_id::d1, shapes::two_rows_of_three(0x20000));
	const auto plan = staged_chunk(spec, staging(0), 0, 6 * ref::elem_size);

	REQUIRE(implements(plan, spec));
	CHECK(is_equivalent(plan, spec));
	CHECK(is_equivalent(copy_plan{spec}, spec));
}
