#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

#include "game_value.h"


#include <random>

int main()
{
	std::cout.precision(std::numeric_limits<long double>::max_digits10);

	using Scalar = double;
	parameter_t<Scalar, 2> X;
	value_t<Scalar, 2> E;
	derivative_t<Scalar, 2> f;
	jacobian_t<Scalar, 2> J(vars<2>, vars<2>);

	std::default_random_engine generator{std::random_device{}()};
	std::uniform_real_distribution<Scalar> distribution;
	X = decltype(X)::NullaryExpr([&]() {
		return distribution(generator);
	});
	Scalar P = 8;
//	X.segment(0,168).fill(0);
//	X(168) = 1;
//	X.segment(169,168).fill(0);
//	X(169*2-1) = 1;
//	X.segment(169*2, 169).fill(1);
	X(P_pos<2>) = P;
	
	game_value(X, Scalar(1e9), (std::array<parameter_t<Scalar, 2>, 2>*)nullptr, Scalar(1e-9), &E, &f, &J);
	std::cout << "E = " << E << "\n";
	std::cout << "f(0) = " << f(0) << "\n";
	std::cout << "J(0, 1) = " << J(0 * ranges + 0, 1 * ranges + 1) << "\n";
	Eigen::ColPivHouseholderQR<decltype(J)> solver;
	Scalar e = 1e-4;
	Scalar times = 10;
	Scalar last_e = e;
	while (e < 1e9) {
		auto saved = X;
		// std::cout << "P guess = " << X(P_pos<2>) << "\n";
		int iterations = 20;
		while (--iterations) {
			game_value(X, e, (std::array<parameter_t<Scalar, 2>, 2>*)nullptr, Scalar(0), &E, &f, &J);
			if (f.norm() < 1e-6) break;
		//	std::cout << "X(0) = " << X(0) << "\n";
		//	std::cout << "E1 = " << E << "\n";
			solver.compute(J);
			X -= solver.solve(f);
		}
		if (f.norm() >= 1e-6) {
			std::cout << "norm = " << f.norm() << "\n";
			std::cout << "e = " << e << "\n";
			X = saved;
			if (times * .9 <= 1) break;
			e = last_e;
			times *= .9;
			e *= times;
		} else {
			std::cout << "norm = " << f.norm() << "\n";
			std::cout << X(P_pos<2>) << ", " << E(0) << "\n";
			last_e = e;
			e *= times;
		}
	}
	
	for (int i = 0; i < X.size(); ++i) {
		auto range = i % ranges;
		auto node = i / ranges;
		auto rank1 = range / 13;
		auto rank2 = range % 13;
		auto suit = rank1 < rank2 ? "o" : "s";
		std::cout << "node " << node << ", range " << range << " = " << X(i) << "\n";
	}
	game_value(X, Scalar(1e9), (std::array<parameter_t<Scalar, 2>, 2>*)nullptr, Scalar(1e-9), &E, (derivative_t<Scalar>*)nullptr, (jacobian_t<Scalar>*)nullptr);
	game_value(X, Scalar(1e9), (std::array<parameter_t<Scalar, 2>, 2>*)nullptr, Scalar(1e-9), &E, &f, &J);
	std::cout << E << "\n";
}
