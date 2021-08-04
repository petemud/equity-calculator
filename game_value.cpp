#include "game_value.h"

template void game_value<2, double>(
	const parameter_t<double, 2> &X,
	const double &P,
	value_t<double, 2> *value,
	derivative_t<double, 2> *derivative,
	jacobian_t<double, 2> *jacobian,
	const double &e
);
