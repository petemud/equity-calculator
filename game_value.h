#include <iostream>
#include <fstream>
#include <memory>
#include <array>
#include <Eigen/Dense>
#include "loop.h"

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

template<typename Scalar, int players = 2>
using parameter_t = Eigen::Matrix<Scalar, vars(players), 1>;

template<typename Scalar, int players = 2>
using value_t = Eigen::Matrix<Scalar, players, 1>;

template<typename Scalar, int players = 2>
using derivative_t = Eigen::Matrix<Scalar, vars(players), 1>;

template<typename Scalar, int players = 2>
using jacobian_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template<int players = 2, typename Scalar>
void game_value(
	const parameter_t<Scalar, players> &X,
	const Scalar &P,
	value_t<Scalar, players> *value,
	derivative_t<Scalar, players> *derivative,
	jacobian_t<Scalar, players> *jacobian,
	const Scalar &e = 1
) {
	constexpr Scalar SMALL_BLIND = 1;
	constexpr Scalar BIG_BLIND = SMALL_BLIND*2;
	Scalar prob;
	std::array<Scalar, players> dprob;
	std::array<decltype(dprob), players> ddprob;
	// The stack of a single player (must be >= BB)
	auto stack = P + BIG_BLIND;
	std::unique_ptr<derivative_t<Scalar, players>> temp_derivative;
	if (value)
		value->setConstant(0);
	if (derivative) {
		temp_derivative = std::make_unique<derivative_t<Scalar, players>>();
		temp_derivative->setConstant(0);
	}
	if (jacobian)
		jacobian->setConstant(0);
	nested_loops<ranges, players>([&](auto player_range) {
		// Scalar times_val = times<players>(player_range) * (1 / Scalar((52*51/2) * (50*49/2)));
		Scalar times_val = times<players>(player_range);
		if (times_val == 0)
			return;
		for (int mask = 1, max_mask = 1 << players; mask < max_mask; ++mask) {
			prob = times_val;
			dprob.fill(times_val);
			ddprob.fill(dprob);
			Scalar pot = 0;
			std::array<int, players> betting, folding;
			int nbetting = 0, nfolding = 0;
			std::array<int, players> player_node;
			for (int node = 0, player = 0; player < players; ++player) {
				auto x = X(node * ranges + player_range[player]);
				player_node[player] = node;
				if ((mask >> player) & 1) {
					node = node * 2 + 1;
					prob *= x;
					for (int p1 = 0; p1 < players; ++p1) {
						if (p1 == player) continue;
						dprob[p1] *= x;
						for (int p2 = 0; p2 < players; ++p2) {
							if (p2 == player || p2 == p1) continue;
							ddprob[p1][p2] *= x;
						}
					}
					pot += stack;
					betting[nbetting++] = player;
				} else {
					node = node * 2 + 2;
					prob *= 1 - x;
					dprob[player] = -dprob[player];
					for (int p1 = 0; p1 < players; ++p1) {
						if (p1 == player) continue;
						dprob[p1] *= 1 - x;
						ddprob[player][p1] = -ddprob[player][p1];
						ddprob[p1][player] = -ddprob[p1][player];
						for (int p2 = 0; p2 < players; ++p2) {
							if (p2 == player || p2 == p1) continue;
							ddprob[p1][p2] *= 1 - x;
						}
					}
					if (player == players - 1)
						pot += BIG_BLIND;
					if (player == players - 2)
						pot += SMALL_BLIND;
					folding[nfolding++] = player;
				}
			}
			for (int player = 0; player < players; ++player) {
				auto id = player_node[player] * ranges + player_range[player];
				Scalar stack_delta = 0;
				if ((mask >> player) & 1) {
					if (nbetting == 1) {
						stack_delta = pot - stack;
					} else {
						std::array<int, players> equity_index;
						int i = 0;
						equity_index[i++] = player_range[player];
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
						stack_delta = pot * equity_val - stack;
					}
				} else {
					if (player == players - 1)
						stack_delta = -BIG_BLIND;
					if (player == players - 2)
						stack_delta = -SMALL_BLIND;
				}
				if (stack_delta == 0) continue;
				if (value)
					value->coeffRef(player) += prob * stack_delta;
				if (derivative || jacobian)
					temp_derivative->coeffRef(id) += dprob[player] * stack_delta;
				if (jacobian)
					for (int player2 = 0; player2 < players; ++player2) {
						if (player2 == player) continue;
						auto id2 = player_node[player2] * ranges + player_range[player2];
						jacobian->coeffRef(id, id2) += ddprob[player][player2] * stack_delta;
					}
			}
		}
	});
	if (derivative || jacobian)
		*temp_derivative *= e;
	if (derivative)
		*derivative = temp_derivative->array().atan() * Scalar(1 / EIGEN_PI) + Scalar(.5) - X.array();
	if (jacobian) {	
		*temp_derivative = (temp_derivative->array().square() + 1).inverse() * Scalar(e / EIGEN_PI);
		for (int i = 0; i < vars(players); ++i) {
			jacobian->row(i) *= temp_derivative->coeffRef(i);
			jacobian->coeffRef(i, i) = -1;
		}
	}
}

extern template void game_value<2, double>(
	const parameter_t<double, 2> &X,
	const double &P,
	value_t<double, 2> *value,
	derivative_t<double, 2> *derivative,
	jacobian_t<double, 2> *jacobian,
	const double &e
);
