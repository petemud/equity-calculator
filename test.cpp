#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "loop.h"

using Scalar = float;

constexpr int ranges = 169;

template<int players, typename Enable = std::enable_if_t<players == 2>>
float equity(int nbetting, std::array<int, players> player_range) {
	static float table[ranges][ranges];
	static bool initialized = false;
	if (!initialized) {
		auto filename = "equity2.bin";
		std::fstream file(filename, std::ios::in | std::ios::binary);
		if (file.fail()) {
			std::cerr << "Couldn't open file " << filename << "\n";
			exit(1);
		}
		using char_type = std::fstream::char_type;
		file.read(reinterpret_cast<char_type*>(table), sizeof(table) / sizeof(char_type));
		initialized = true;
	}
	return table[player_range[0]][player_range[1]];
}

template<int players, typename Enable = std::enable_if_t<players == 2>>
uint16_t times(std::array<int, players> player_range) {
	static uint16_t table[ranges][ranges];
	static bool initialized = false;
	if (!initialized) {
		auto filename = "times2.bin";
		std::fstream file(filename, std::ios::in | std::ios::binary);
		if (file.fail()) {
			std::cerr << "Couldn't open file " << filename << "\n";
			exit(1);
		}
		using char_type = std::fstream::char_type;
		file.read(reinterpret_cast<char_type*>(table), sizeof(table) / sizeof(char_type));
		initialized = true;
	}
	return table[player_range[0]][player_range[1]];
}

// Number of game tree nodes for a given number of players
constexpr int nodes(int players) {
	return (1 << players) - 1;
}

// Number of frequency variables x for a given number of players
constexpr int vars(int players) {
	return nodes(players) * ranges;
}

template<int cur = 0, int players = 2, typename Scalar>
Scalar game_value(const Eigen::Matrix<Scalar, vars(players), 1> &X, const Scalar &P) {
	constexpr Scalar SMALL_BLIND = 1;
	constexpr Scalar BIG_BLIND = SMALL_BLIND*2;
	// The stack of a single player (must be >= BB)
	auto stack = P + BIG_BLIND;
	Scalar result = 0;
	nested_loops<ranges, players>([&](auto player_range) {
		Scalar times_val = times<players>(player_range);
		if (times_val == 0)
			return;
		for (int mask = 1, max_mask = 1 << players; mask < max_mask; ++mask) {
			Scalar prob = times_val;
			Scalar pot = 0;
			std::array<int, players> betting, folding;
			int nbetting = 0, nfolding = 0;
			for (int node = 0, player = 0; player < players; ++player) {
				auto x = X(node * ranges + player_range[player]);
				if ((mask >> player) & 1) {
					node = node * 2 + 1;
					prob *= x;
					pot += stack;
					betting[nbetting++] = player;
				} else {
					node = node * 2 + 2;
					prob *= 1 - x;
					if (player == players - 1)
						pot += BIG_BLIND;
					if (player == players - 2)
						pot += SMALL_BLIND;
					folding[nfolding++] = player;
				}
			}
			if ((mask >> cur) & 1) {
				if (nbetting == 1) {
					result += prob * (pot - stack);
				} else {
					std::array<int, players> equity_index;
					int i = 0;
					equity_index[i++] = player_range[cur];
					int ibetting = nbetting, ifolding = nfolding;
					// First go betting players
					while (ibetting--) {
						if (betting[ibetting] == cur)
							continue;
						equity_index[i++] = player_range[betting[ibetting]];
					}
					// And then folding ones
					while (ifolding--) {
						equity_index[i++] = player_range[folding[ifolding]];
					}
					Scalar equity_val = equity<players>(nbetting, equity_index);
					result += prob * (pot * equity_val - stack);
				}
			} else {
				if (cur == players - 1)
					result -= prob * BIG_BLIND;
				if (cur == players - 2)
					result -= prob * SMALL_BLIND;
			}
		}
	});
	return result / ((52*51/2) * (50*49/2));
}

template<int players = 2, typename Scalar>
Eigen::Matrix<Scalar, vars(players), 1> regularized(const Eigen::Matrix<Scalar, vars(players), 1> &X, const Scalar &P) {
	constexpr Scalar SMALL_BLIND = 1;
	constexpr Scalar BIG_BLIND = SMALL_BLIND*2;
	// The stack of a single player (must be >= BB)
	auto stack = P + BIG_BLIND;
	Eigen::Matrix<Scalar, vars(players), 1> f = decltype(f)::Zero();
	nested_loops<ranges, players>([&](auto player_range) {
		Scalar times_val = times<players>(player_range);
		if (times_val == 0)
			return;
		for (int mask = 1, max_mask = 1 << players; mask < max_mask; ++mask) {
			Scalar prob = times_val;
			Scalar pot = 0;
			std::array<int, players> betting, folding;
			int nbetting = 0, nfolding = 0;
			for (int node = 0, player = 0; player < players; ++player) {
				auto x = X(node * ranges + player_range[player]);
				if ((mask >> player) & 1) {
					node = node * 2 + 1;
					prob *= x;
					pot += stack;
					betting[nbetting++] = player;
				} else {
					node = node * 2 + 2;
					prob *= 1 - x;
					if (player == players - 1)
						pot += BIG_BLIND;
					if (player == players - 2)
						pot += SMALL_BLIND;
					folding[nfolding++] = player;
				}
			}
			for (int node = 0, player = 0; player < players; ++player) {
				auto range = player_range[player];
				auto x = X(node * ranges + range);
				auto &result = f(node * ranges + range);
				if ((mask >> player) & 1) {
					node = node * 2 + 1;
					if (nbetting == 1) {
						result += prob / x * (pot - stack);
					} else {
						std::array<int, players> equity_index;
						int i = 0;
						equity_index[i++] = range;
						int ibetting = nbetting, ifolding = nfolding;
						// First go betting players
						while (ibetting--) {
							if (betting[ibetting] == player)
								continue;
							equity_index[i++] = player_range[betting[ibetting]];
						}
						// And then folding ones
						while (ifolding--) {
							equity_index[i++] = player_range[folding[ifolding]];
						}
						Scalar equity_val = equity<players>(nbetting, equity_index);
						result += prob / x * (pot * equity_val - stack);
					}
				} else {
					node = node * 2 + 2;
					if (player == players - 1)
						result -= prob / (1 - x) * BIG_BLIND;
					if (player == players - 2)
						result -= prob / (1 - x) * SMALL_BLIND;
				}
			}
		}
	});
	f /= ((52*51/2) * (50*49/2));
//	f *= 1e9;
	return atan(f.array()) / EIGEN_PI + .5;
}

#include <random>

int main()
{
//	Eigen::Matrix<Scalar, 2, 2> A;
//	Eigen::Matrix<Scalar, 2, 1> b; b << 3, 3;
//	Eigen::FullPivLU<decltype(A)> fullPivLu;
//	fullPivLu.setThreshold(1e-100);
//	auto x2 = fullPivLu.compute(A).solve(b);
	std::cout.precision(std::numeric_limits<long double>::max_digits10);

	Eigen::Matrix<Scalar, vars(2), 1> X;
	std::default_random_engine generator{std::random_device{}()};
	std::uniform_real_distribution<Scalar> distribution;
	X = decltype(X)::NullaryExpr([&]() {
		return distribution(generator);
	});
	Scalar P = 20;
	std::cout << "E1 = " << game_value<0>(X, P) << "\n";
	std::cout << "E2 = " << game_value<1>(X, P) << "\n";
	std::cout << "f = " << regularized(X, P) << "\n";
	auto f = regularized(X, P);
	std::cout << "f(last) = " << f(f.size() - 1) << "\n";
}

