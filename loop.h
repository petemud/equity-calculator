#include <array>

template<int Size, int Times, typename Function, typename ...Indexes>
std::enable_if_t<(Times > 0)> nested_loops(Function f, Indexes... indexes) {
	for (int i = 0; i < Size; ++i) {
		nested_loops<Size, Times - 1>(f, indexes..., i);
	}
}

template<int Size, int Times, typename Function, typename ...Indexes>
std::enable_if_t<Times == 0> nested_loops(Function f, Indexes... indexes) {
	f(std::array{indexes...});
}
