#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

#include "game_value.h"


#include <random>

int main()
{
	std::cout.precision(std::numeric_limits<long double>::max_digits10);

	using Scalar = double;
	parameter_t<Scalar, 2> X;
	std::default_random_engine generator{std::random_device{}()};
	std::uniform_real_distribution<Scalar> distribution;
	X = decltype(X)::NullaryExpr([&]() {
		return distribution(generator);
	});
	Scalar P = 2e4;
	X.segment(0,168).fill(0);
	X(168) = 1;
	X.segment(169,168).fill(0);
	X(169*2-1) = 1;
	X.segment(169*2, 169).fill(1);

	value_t<Scalar, 2> E;
	derivative_t<Scalar, 2> f;
	jacobian_t<Scalar, 2> J(vars<2>, vars<2>);
	
	game_value(X, Scalar(1e9), (std::array<parameter_t<Scalar, 2>, 2>*)nullptr, Scalar(1e-9), &E, &f, &J);
	std::cout << "E = " << E << "\n";
	std::cout << "f(0) = " << f(0) << "\n";
	std::cout << "J(0, 1) = " << J(0 * ranges + 0, 1 * ranges + 1) << "\n";
	Eigen::HouseholderQR<decltype(J)> solver;
	Scalar e = 1e2;
	std::array<decltype(X), 2> prev{X, X};
	prev[0](P_pos<2>) = P - 1;
	prev[1](P_pos<2>) = P;
	X(P_pos<2>) = P - 2;
	Scalar delta = 1;
	while (X(P_pos<2>) > 15) {
		auto saved = X;
		X = prev[0] - prev[1];
		X.normalize();
		X = prev[0] + delta * X;
		int iteration = 8;
		while (iteration--) {
			game_value(X, e, &prev, delta, &E, &f, &J);
			if (f.norm() < 1e-9) break;
		//	std::cout << "X(0) = " << X(0) << "\n";
		//	std::cout << "E1 = " << E << "\n";
			solver.compute(J);
			X -= solver.solve(f);
		}
		if (f.norm() > 1e-9 || (X - prev[0]).norm() > std::max(delta * 1.05, 1e-3/e)) {
			std::cout << "norm = " << f.norm() << "\n";
			std::cout << "delta = " << delta << "\n";
			X = saved;
			delta /= 2;
		} else {
			std::cout << "P = " << X(P_pos<2>) << "\n";
			std::cout << "E1 = " << E(0) << "\n";
			if (iteration > 4) {
				delta *= 1.5;
			} else {
				delta = std::min(delta * 1.2, .1);
			}
			std::swap(prev[1], prev[0]);
			prev[0] = X;
		}
	}
	
	for (int i = 0; i < X.size(); ++i) {
		auto range = i % ranges;
		auto node = i / ranges;
		auto rank1 = range / 13;
		auto rank2 = range % 13;
		auto suit = rank1 < rank2 ? "o" : "s";
		std::cout << "node " << node << ", range " << range << " = " << X(i) << " " << prev[1](i) << "\n";
	}
	game_value(X, Scalar(1e9), (std::array<parameter_t<Scalar, 2>, 2>*)nullptr, Scalar(1e-9), &E, (derivative_t<Scalar>*)nullptr, (jacobian_t<Scalar>*)nullptr);
	game_value(X, Scalar(1e9), (std::array<parameter_t<Scalar, 2>, 2>*)nullptr, Scalar(1e-9), &E, &f, &J);
	std::cout << E << "\n";
}
