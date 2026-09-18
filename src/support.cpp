#include <copylib/support.hpp>

namespace std {

size_t hash<copylib::data_layout>::operator()(const copylib::data_layout& layout) const { return {}; }

size_t hash<copylib::copy_properties>::operator()(const copylib::copy_properties& prop) const { return {}; }

size_t hash<copylib::copy_spec>::operator()(const copylib::copy_spec& spec) const { return {}; }

size_t hash<copylib::copy_plan>::operator()(const copylib::copy_plan& plan) const { return {}; }

size_t hash<copylib::copy_type>::operator()(const copylib::copy_type& type) const { return {}; }

size_t hash<copylib::d2d_implementation>::operator()(const copylib::d2d_implementation& impl) const { return {}; }

size_t hash<copylib::copy_strategy>::operator()(const copylib::copy_strategy& strat) const { return {}; }

} // namespace std

#ifdef COPYLIB_USE_FMT
namespace fmt {
#else
namespace std {
#endif

format_context::iterator formatter<copylib::device_id>::format(const copylib::device_id& p, format_context& ctx) const { return ctx.out(); }

format_context::iterator formatter<copylib::staging_id>::format(const copylib::staging_id& p, format_context& ctx) const { return ctx.out(); }

format_context::iterator formatter<copylib::data_layout>::format(const copylib::data_layout& p, format_context& ctx) const { return ctx.out(); }

format_context::iterator formatter<copylib::copy_properties>::format(const copylib::copy_properties& p, format_context& ctx) const { return ctx.out(); }

format_context::iterator formatter<copylib::copy_spec>::format(const copylib::copy_spec& p, format_context& ctx) const { return ctx.out(); }

format_context::iterator formatter<copylib::copy_type>::format(const copylib::copy_type& p, format_context& ctx) const { return ctx.out(); }

format_context::iterator formatter<copylib::d2d_implementation>::format(const copylib::d2d_implementation& p, format_context& ctx) const { return ctx.out(); }

format_context::iterator formatter<copylib::copy_strategy>::format(const copylib::copy_strategy& p, format_context& ctx) const { return ctx.out(); }

format_context::iterator formatter<copylib::copy_plan>::format(const copylib::copy_plan& p, format_context& ctx) const { return ctx.out(); }

format_context::iterator formatter<copylib::parallel_copy_set>::format(const copylib::parallel_copy_set& p, format_context& ctx) const { return ctx.out(); }

} // namespace fmt // std

namespace std {

#define COPYLIB_OSTREAM_FOR(__type)                                                                                                                            \
	std::ostream& operator<<(std::ostream& os, const copylib::__type& p) { return os; }

COPYLIB_OSTREAM_FOR(device_id)
COPYLIB_OSTREAM_FOR(staging_id)
COPYLIB_OSTREAM_FOR(data_layout)
COPYLIB_OSTREAM_FOR(copy_properties)
COPYLIB_OSTREAM_FOR(copy_spec)
COPYLIB_OSTREAM_FOR(copy_type)
COPYLIB_OSTREAM_FOR(d2d_implementation)
COPYLIB_OSTREAM_FOR(copy_strategy)
COPYLIB_OSTREAM_FOR(copy_plan)
COPYLIB_OSTREAM_FOR(parallel_copy_set)

#undef COPYLIB_OSTREAM_FOR

} // namespace std
