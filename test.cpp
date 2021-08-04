#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

#include "game_value.h"


#include <random>

int main()
{
//	Eigen::Matrix<Scalar, 2, 2> A;
//	Eigen::Matrix<Scalar, 2, 1> b; b << 3, 3;
//	Eigen::FullPivLU<decltype(A)> fullPivLu;
//	fullPivLu.setThreshold(1e-100);
//	auto x2 = fullPivLu.compute(A).solve(b);
	std::cout.precision(std::numeric_limits<long double>::max_digits10);

	using Scalar = double;
	Eigen::Matrix<Scalar, vars(2), 1> X;
	std::default_random_engine generator{std::random_device{}()};
	std::uniform_real_distribution<Scalar> distribution;
	X = decltype(X)::NullaryExpr([&]() {
		return distribution(generator);
	});
	Scalar P = 1e6;
//	X.segment(0,168).fill(0);
//	X(168) = 1;
//	X.segment(169,168).fill(0);
//	X(169*2-1) = 1;
//	X.segment(169*2, 169).fill(1);
	Eigen::Matrix<Scalar, 2, 1> E;
	Eigen::Matrix<Scalar, vars(2), 1> f;
	jacobian_t<Scalar> J;
	game_value(X, P, &E, &f, &J);
	std::cout << "E = " << E << "\n";
	std::cout << "f(last) = " << f(X.size() - 1) << "\n";
	std::cout << "J(last, last) = " << J(0 * ranges + 0, 1 * ranges + 1) << "\n";
	struct functor {
		Scalar P, e;
		int operator()(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x,
				Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &fvec) const {
			parameter_t<Scalar> X = x;
			derivative_t<Scalar> vec;
			game_value(X, P, (value_t<Scalar>*)nullptr, &vec, (jacobian_t<Scalar>*)nullptr, e);
			fvec = vec;
			return 0;
		}

		int df(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x,
				Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &fjac) {
			parameter_t<Scalar> X = x;
			jacobian_t<Scalar> jac;
			game_value(X, P, (value_t<Scalar>*)nullptr, (derivative_t<Scalar>*)nullptr, &jac, e);
			fjac = jac;
			return 0;
		}

		int values() const {
			return vars(2);
		}
	};
	auto func = functor{P, 1e9};
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Y = X;
	{
		Eigen::HybridNonLinearSolver<functor, Scalar> nonlinsolver{func};
		auto info = nonlinsolver.hybrj1(Y);
		std::cout << info << "\n";
	}
	X = Y;
	func = functor{P-1, 1e9};
	{
		Eigen::HybridNonLinearSolver<functor, Scalar> nonlinsolver{func};
		auto info = nonlinsolver.hybrj1(Y);
		std::cout << info << "\n";
	}
	X -= Y;
//	auto J = jacobian(X, P);
//	Eigen::SparseLU<decltype(J)> solver;
//	solver.analyzePattern(J);
//	for (Scalar e = 1e1; e < 1e9;) {
//		do {
//			if (e > 1e9) break;
//			e += e/10;
//		} while (regularized(X, P, e).norm() < 1e-2);
//		while (true) {
//			auto f = regularized(X, P, e);
//			auto norm = f.norm();
//			auto E = game_value<0>(X, P);
//			std::cout << "e = " << e << "\n";
//			std::cout << "X(0) = " << X(0) << "\n";
//			std::cout << "norm = " << norm << "\n";
//			std::cout << "E1 = " << E << "\n";
//			if (norm < 1e-8) {
//				break;
//			}
//			auto J = jacobian(X, P, e);
//			solver.factorize(J);
//			if (solver.info() != Eigen::Success) {
//				std::cout << "decomposition failed\n";
//				return 0;
//			}
//			X -= solver.solve(f);
//			if (solver.info() != Eigen::Success) {
//				std::cout << "solving failed\n";
//				return 0;
//			}
//		}
//	}
	for (int i = 0; i < X.size(); ++i) {
		auto range = i % ranges;
		auto node = i / ranges;
		auto rank1 = range / 13;
		auto rank2 = range % 13;
		auto suit = rank1 < rank2 ? "o" : "s";
		std::cout << "node " << node << ", range " << range << " = " << X(i) << "\n";
	}
	game_value(X, P, &E, (derivative_t<Scalar>*)nullptr, (jacobian_t<Scalar>*)nullptr);
	std::cout << E << "\n";
//	X.segment(169, 169) = decltype(X)::Constant(1);
//	auto s = system(X, P);
//	for (int i = 0; i < s.size(); ++i)
//		std::cout << i << " " << s(i) << "\n";
//	std::cout << game_value<0>(X, P) << "\n";
//	std::cout << game_value<1>(X, P) << "\n";
}

