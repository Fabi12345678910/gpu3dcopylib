#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>

// Layer 2: the validity rules for layouts, specs, plans and sets, as listed in docs/design.md.
//
// Every case pairs the rejection of a malformed input with the acceptance of a well-formed one. That matters while
// `is_valid` is a placeholder returning false: without the positive check, a rejection test would pass for the wrong
// reason and report progress that is not there.

using namespace copylib;
using copylib_testing::layout_from_fields;
using copylib_testing::spec_from_fields;
namespace ref = copylib_testing::reference_box;

TEST_CASE("a well-formed layout is valid", "[validation][!mayfail]") {
	CHECK(is_valid(ref::fields()));
}

TEST_CASE("strides must be non-zero", "[validation][!mayfail]") {
	CHECK(is_valid(ref::fields()));

	auto no_d0 = ref::fields();
	no_d0.d0_stride = 0;
	CHECK_FALSE(is_valid(no_d0));

	auto no_d1 = ref::fields();
	no_d1.d1_stride = 0;
	CHECK_FALSE(is_valid(no_d1));
}

TEST_CASE("the plane stride must be a multiple of the row stride", "[validation][!mayfail]") {
	CHECK(is_valid(ref::fields()));

	auto layout = ref::fields();
	layout.d1_stride = ref::d1_stride + 1;
	CHECK_FALSE(is_valid(layout));
}

TEST_CASE("offsets must be aligned to the enclosing stride", "[validation][!mayfail]") {
	CHECK(is_valid(ref::fields()));

	SECTION("d1 offsets are multiples of the row stride") {
		auto layout = ref::fields();
		layout.d1_start_offset = ref::d1_start_offset + 1;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("d2 offsets are multiples of the plane stride") {
		auto layout = ref::fields();
		layout.d2_end_offset = ref::d2_end_offset + 1;
		CHECK_FALSE(is_valid(layout));
	}
}

TEST_CASE("the box must fit within the strides", "[validation][!mayfail]") {
	CHECK(is_valid(ref::fields()));

	SECTION("a row may not be longer than the row stride") {
		auto layout = ref::fields();
		layout.d0_start_offset = 0;
		layout.d0_end_offset = ref::d0_stride + ref::elem_size;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("a plane may not hold more rows than the plane stride") {
		auto layout = ref::fields();
		layout.d1_start_offset = 0;
		layout.d1_end_offset = ref::d1_stride + ref::d0_stride;
		CHECK_FALSE(is_valid(layout));
	}
}

TEST_CASE("the encoding must be canonical", "[validation][!mayfail]") {
	// Otherwise one box has several encodings, which breaks operator==, hashing and plan chaining.
	CHECK(is_valid(ref::fields()));

	SECTION("the row offset is inside one row") {
		auto layout = ref::fields();
		layout.d0_start_offset = ref::d0_stride + ref::d0_start_offset;
		layout.d0_end_offset = ref::d0_stride + ref::d0_end_offset;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("the plane offset is inside one plane") {
		auto layout = ref::fields();
		layout.d1_start_offset = ref::d1_stride + ref::d1_start_offset;
		layout.d1_end_offset = ref::d1_stride + ref::d1_end_offset;
		CHECK_FALSE(is_valid(layout));
	}
}

TEST_CASE("the base must be at least 2-byte aligned", "[validation][!mayfail]") {
	// the lowest byte of the base is what distinguishes a placed layout from a staging placeholder
	CHECK(is_valid(ref::fields()));

	auto layout = ref::fields();
	layout.base = ref::base | 0x1;
	CHECK_FALSE(is_valid(layout));
}

TEST_CASE("the window must lie within the box", "[validation][window][!mayfail]") {
	CHECK(is_valid(ref::fields()));

	SECTION("a window may not extend past the end of the box") {
		auto layout = ref::fields();
		layout.end = ref::total_bytes + 1;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("a window may not be empty") {
		auto layout = ref::fields();
		layout.start = layout.end;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("a window may not start before the box") {
		auto layout = ref::fields();
		layout.start = -1;
		CHECK_FALSE(is_valid(layout));
	}

	SECTION("a sub-window of the box is valid") {
		auto layout = ref::fields();
		layout.start = ref::row_extent;
		layout.end = 2 * ref::row_extent;
		CHECK(is_valid(layout));
	}
}

TEST_CASE("a copy spec requires equal window lengths on both sides", "[validation][!mayfail]") {
	const auto source = ref::fields();

	SECTION("equal windows are valid") {
		auto target = ref::fields();
		target.base = ref::base + 0x10000;
		CHECK(is_valid(spec_from_fields(device_id::d0, source, device_id::d1, target)));
	}

	SECTION("differing window lengths are not") {
		auto target = ref::fields();
		target.base = ref::base + 0x10000;
		target.end = ref::total_bytes - ref::elem_size;
		CHECK_FALSE(is_valid(spec_from_fields(device_id::d0, source, device_id::d1, target)));
	}
}

TEST_CASE("a reshaping copy spec is valid", "[validation][!mayfail]") {
	// design.md: source and target may have different shapes, so int[1,1,6] -> int[1,2,3] is allowed
	constexpr int64_t elem = ref::elem_size;

	// int[1,1,6]: one row of 6 elements
	const auto source = layout_from_fields(0x10000, 6 * elem, 6 * elem, 0, 0, 0, 6 * elem, 6 * elem, 6 * elem, 0, 6 * elem);
	// int[1,2,3]: two rows of 3 elements
	const auto target = layout_from_fields(0x20000, 3 * elem, 6 * elem, 0, 0, 0, 3 * elem, 6 * elem, 6 * elem, 0, 6 * elem);

	CHECK(is_valid(source));
	CHECK(is_valid(target));
	CHECK(is_valid(spec_from_fields(device_id::d0, source, device_id::d1, target)));
}

TEST_CASE("an empty copy plan or set is invalid", "[validation][!mayfail]") {
	CHECK_FALSE(is_valid(copy_plan{}));
	CHECK_FALSE(is_valid(parallel_copy_set{}));

	auto target = ref::fields();
	target.base = ref::base + 0x10000;
	const auto spec = spec_from_fields(device_id::d0, ref::fields(), device_id::d1, target);

	CHECK(is_valid(copy_plan{spec}));
	CHECK(is_valid(parallel_copy_set{copy_plan{spec}}));
}
