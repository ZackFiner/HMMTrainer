#include "ProbInit.h"
#include <random>
#include <chrono>
#include <iostream>
#include <math.h>

#define EPSILON 1e-10f

DefaultProbInit::DefaultProbInit(float _variance) {
	this->variance = _variance; // this should be > 0 and < 1
}

float** DefaultProbInit::AInit(unsigned int N) {
	float** new_A = new float* [N];
	for (unsigned int i = 0; i < N; i++)
		new_A[i] = new float[N];

	for (unsigned int i = 0; i < N; i++) {
		this->initializeStochasticRow(new_A[i], N);
	}

	return new_A;
}

float** DefaultProbInit::BInit(unsigned int N, unsigned int M) {
	float** new_B = new float* [N];
	for (unsigned int i = 0; i < N; i++)
		new_B[i] = new float[M];

	for (unsigned int i = 0; i < N; i++) {
		this->initializeStochasticRow(new_B[i], M);
	}

	return new_B;
}

float* DefaultProbInit::PiIinit(unsigned int N) {
	float* new_Pi = new float[N];
	this->initializeStochasticRow(new_Pi, N);
	return new_Pi;
}

void DefaultProbInit::initializeStochasticRow(float* row, unsigned int size) {
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();

	float mean = 1.0f / size;
	float true_variance = this->variance * mean;
	std::default_random_engine gen;
	gen.seed(seed);
	std::uniform_real_distribution<float> random_nums(-true_variance, true_variance);
	float accum = 0;
	for (unsigned int i = 0; i < size; i++) {
		float random_value = random_nums(gen);
		row[i] = mean + random_value;
		accum += random_value; // to obey the stochastic property, this should fluctuate around 0
	}

	if (accum > EPSILON || accum < -EPSILON) { // if our accumulator is not roughly 0 (meaning that our row doesn't obey the stochastic property)
		std::vector<unsigned int> targets;
		if (accum > EPSILON) { // if we're too high
			for (unsigned int i = 0; i < size; i++)
				if (row[i] > mean) // find all values that are greater than the mean
					targets.push_back(i);
		}
		else { // if we're too low
			for (unsigned int i = 0; i < size; i++)
				if (row[i] < mean) // find all values that are less than the mean
					targets.push_back(i);
		}

		float correction = accum / targets.size();

		for (unsigned int index : targets)
			row[index] -= correction; // add some fraction of the amount that we're over to maintain the stochastic property 
	}
}