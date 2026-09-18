#include "test_utils.hpp"

#include <copylib/core.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <vector>

// Layer 5: apply_chunking() splits a spec into independent single-spec plans of at most chunk_size bytes each.
//
// design.md: chunking keeps the box and narrows only the window; chunk boundaries are rounded to the local alignment
// (the largest power of two up to 64 dividing the row extents and all strides); chunk_size == 0 means no chunking.

using namespace copylib;
using namespace copylib_testing;
namespace ref = copylib_testing::reference_box;

namespace {

// a set's plans are unordered, so compare chunks by where their window starts
std::vector<copy_spec> chunks_by_start(const parallel_copy_set& set) {
	std::vector<copy_spec> chunks;
	for(const auto& plan : set) {
		for(const auto& spec : plan) {
			chunks.push_back(spec);
		}
	}
	std::sort(chunks.begin(), chunks.end(), [](const copy_spec& a, const copy_spec& b) { return a.source_layout.start < b.source_layout.start; });
	return chunks;
}

std::vector<int64_t> chunk_lengths(const parallel_copy_set& set) {
	std::vector<int64_t> lengths;
	for(const auto& chunk : chunks_by_start(set)) {
		lengths.push_back(chunk.source_layout.window_length());
	}
	return lengths;
}

copy_spec reshaping_spec() {
	return spec_from_fields(device_id::d0, shapes::one_row_of_six(0x10000), device_id::d1, shapes::two_rows_of_three(0x20000));
}

} // namespace

TEST_CASE("a chunk size of zero means no chunking", "[chunking][!mayfail]") {
	const auto spec = ref::spec();
	const auto set = apply_chunking(spec, chunking_strategy(0));

	REQUIRE(set.size() == 1);
	REQUIRE(set.front().size() == 1);
	const auto& only = set.front().front();
	CHECK(same_fields(only.source_layout, spec.source_layout));
	CHECK(same_fields(only.target_layout, spec.target_layout));
	CHECK(only.source_device == spec.source_device);
	CHECK(only.target_device == spec.target_device);
}

TEST_CASE("chunks implement the spec", "[chunking][!mayfail]") {
	const int64_t chunk_size = GENERATE(8, 24, 64, 77, 96, 100, 288, 1000);
	CAPTURE(chunk_size);

	const auto spec = ref::spec();
	const auto set = apply_chunking(spec, chunking_strategy(chunk_size));

	CHECK(implements(set, spec));
}

TEST_CASE("every chunk is a single direct copy of at most chunk_size bytes", "[chunking][!mayfail]") {
	const int64_t chunk_size = GENERATE(8, 24, 64, 77, 96, 100);
	CAPTURE(chunk_size);

	const auto spec = ref::spec();
	const auto set = apply_chunking(spec, chunking_strategy(chunk_size));

	REQUIRE_FALSE(set.empty());
	for(const auto& plan : set) {
		REQUIRE(plan.size() == 1);
		CHECK(plan.front().source_layout.window_length() <= chunk_size);
		CHECK(plan.front().target_layout.window_length() == plan.front().source_layout.window_length());
	}
}

TEST_CASE("chunking keeps the box and narrows only the window", "[chunking][window][!mayfail]") {
	const auto spec = ref::spec();
	const auto set = apply_chunking(spec, chunking_strategy(64));

	REQUIRE_FALSE(set.empty());
	for(const auto& chunk : chunks_by_start(set)) {
		CHECK(same_box(chunk.source_layout, spec.source_layout));
		CHECK(same_box(chunk.target_layout, spec.target_layout));
		CHECK(chunk.source_device == spec.source_device);
		CHECK(chunk.target_device == spec.target_device);
	}
}

TEST_CASE("source and target windows advance in lockstep", "[chunking][window][!mayfail]") {
	const auto spec = GENERATE(ref::spec(), reshaping_spec());
	const auto set = apply_chunking(spec, chunking_strategy(8));

	REQUIRE(set.size() > 1);
	for(const auto& chunk : chunks_by_start(set)) {
		CHECK(chunk.source_layout.start - spec.source_layout.start == chunk.target_layout.start - spec.target_layout.start);
		CHECK(chunk.source_layout.window_length() == chunk.target_layout.window_length());
	}
}

TEST_CASE("chunk boundaries are aligned", "[chunking][!mayfail]") {
	// the reference box has row extent 24 and strides 80 and 1280, so its alignment is 8
	const int64_t chunk_size = GENERATE(64, 77, 100);
	CAPTURE(chunk_size);

	const auto set = apply_chunking(ref::spec(), chunking_strategy(chunk_size));

	REQUIRE_FALSE(set.empty());
	for(const auto& chunk : chunks_by_start(set)) {
		CHECK(chunk.source_layout.start % ref::alignment == 0);
		CHECK(chunk.source_layout.end % ref::alignment == 0);
	}
}

TEST_CASE("chunks are as large as alignment permits", "[chunking][!mayfail]") {
	SECTION("perfectly divisible") {
		CHECK(chunk_lengths(apply_chunking(ref::spec(), chunking_strategy(96))) == std::vector<int64_t>{96, 96, 96});
	}

	SECTION("with a remainder, which goes into the last chunk") {
		CHECK(chunk_lengths(apply_chunking(ref::spec(), chunking_strategy(64))) == std::vector<int64_t>{64, 64, 64, 64, 32});
	}

	SECTION("an unaligned chunk size is rounded down to the alignment") {
		// each chunk may be at most 77 bytes, and the largest multiple of 8 below that is 72
		CHECK(chunk_lengths(apply_chunking(ref::spec(), chunking_strategy(77))) == std::vector<int64_t>{72, 72, 72, 72});
	}

	SECTION("a chunk size larger than the window yields a single chunk") {
		CHECK(chunk_lengths(apply_chunking(ref::spec(), chunking_strategy(1000))) == std::vector<int64_t>{288});
	}
}

TEST_CASE("a chunk size below the alignment is raised to it", "[chunking][!mayfail]") {
	// design.md: the reference box has alignment 8, so chunks of 4 or 1 byte become chunks of 8
	const int64_t chunk_size = GENERATE(1, 4, 7);
	CAPTURE(chunk_size);

	const auto spec = ref::spec();
	const auto set = apply_chunking(spec, chunking_strategy(chunk_size));

	CHECK(implements(set, spec));
	CHECK(chunk_lengths(set) == std::vector<int64_t>(ref::total_bytes / ref::alignment, ref::alignment));
}

TEST_CASE("chunk boundaries of an unaligned window sit at absolute aligned offsets", "[chunking][window][!mayfail]") {
	// design.md: boundaries are multiples of the alignment in packed offsets, not relative to the window start, so only
	// the window's own ends can be unaligned
	auto spec = ref::spec();
	const auto window = [&](int64_t start, int64_t end) {
		spec.source_layout = with_window_fields(spec.source_layout, start, end);
		spec.target_layout = with_window_fields(spec.target_layout, start, end);
	};

	SECTION("unaligned start") {
		window(3, 288);
		const auto set = apply_chunking(spec, chunking_strategy(64));
		CHECK(implements(set, spec));
		CHECK(chunk_lengths(set) == std::vector<int64_t>{61, 64, 64, 64, 32}); // [3,64), [64,128), ..., [256,288)
	}

	SECTION("unaligned start and end") {
		window(3, 285);
		const auto set = apply_chunking(spec, chunking_strategy(64));
		CHECK(implements(set, spec));
		CHECK(chunk_lengths(set) == std::vector<int64_t>{61, 64, 64, 64, 29});
	}

	SECTION("every interior boundary is aligned") {
		window(3, 285);
		const auto chunks = chunks_by_start(apply_chunking(spec, chunking_strategy(64)));
		REQUIRE(chunks.size() > 1);
		for(size_t i = 1; i < chunks.size(); ++i) {
			CHECK(chunks[i].source_layout.start % ref::alignment == 0);
		}
	}
}

TEST_CASE("a partial window is chunked within the window", "[chunking][window][!mayfail]") {
	auto spec = ref::spec();
	spec.source_layout = with_window_fields(spec.source_layout, 24, 264);
	spec.target_layout = with_window_fields(spec.target_layout, 24, 264);

	const auto set = apply_chunking(spec, chunking_strategy(64));

	CHECK(implements(set, spec));
	const auto chunks = chunks_by_start(set);
	REQUIRE_FALSE(chunks.empty());
	CHECK(chunks.front().source_layout.start == 24);
	CHECK(chunks.back().source_layout.end == 264);
}

TEST_CASE("a reshaping spec is chunked on both sides", "[chunking][!mayfail]") {
	// source row extent 24, target row extent 12, strides 24 / 24 and 12 / 24: the alignment is 4
	const auto spec = reshaping_spec();
	const auto set = apply_chunking(spec, chunking_strategy(8));

	CHECK(implements(set, spec));
	CHECK(chunk_lengths(set) == std::vector<int64_t>{8, 8, 8});
}
