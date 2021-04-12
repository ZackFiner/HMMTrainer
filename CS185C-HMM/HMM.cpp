#include "HMM.h"

HMM::HMM() {

}

HMM::HMM(unsigned int N, unsigned int M) {

}

HMM::~HMM() {

}


void HMM::alphaPass(unsigned int* obs, unsigned int size) {
	alpha = new float* [size];
	for (unsigned int i = 0; i < size; i++)
		alpha[i] = new float[N];

	for (unsigned int i = 0; i < N; i++)
		alpha[0][i] = Pi[i] * B[obs[0]][i]; // B has been transposed to improve spatial locality

	for (unsigned int t = 1; t < size; t++) {
		for (unsigned int i = 0; i < N; i++) {
			float sum = 0;
			for (unsigned int j = 0; j < N; j++) {
				sum += alpha[t - 1][j] * A[j][i]; // probability that we'd see the previous hidden state * the probability we'd transition to this new hidden state
			}
			alpha[t][i] = sum * B[obs[t]][i];

		}
	}
	
}

void HMM::betaPass(unsigned int* obs, unsigned int size) {
	beta = new float* [size];
	for (unsigned int i = 0; i < size; i++)
		beta[i] = new float[N];

	for (unsigned int i = 0; i < N; i++)
		beta[size - 1][i] = 1;

	for (int t = size-2; t >= 0; t--) {
		for (unsigned int i = 0; i < N; i++) {
			float sum = 0;
			for (unsigned int j = 0; j < N; j++)
				sum += A[i][j] * B[obs[t + 1]][j] * beta[t + 1][j];

			beta[t][i] = sum;
		}
	}

}
