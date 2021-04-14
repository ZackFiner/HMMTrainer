#include "HMM.h"
#include "ProbInit.h"
#include "MatUtil.h"

HMM::HMM() {

}

HMM::HMM(float** _A, float** _B, float* _Pi, unsigned int _N, unsigned int _M) {
	this->A = _A;
	this->B = transpose(_B, _N, _M);
	delete_array(_B, _N, _M);
	this->Pi = _Pi;
	
	this->N = _N;
	this->M = _M;
}

HMM::HMM(unsigned int _N, unsigned int _M, ProbInit* initializer) {
	DefaultProbInit def_init;
	ProbInit* init = initializer ? initializer : &def_init;
	this->A = init->AInit(_N);
	this->B = init->BInit(_N, _M);
	
	float** T_B = transpose(this->B, _N, _M); // transpose our B array for simplified processing
	delete_array(this->B, _N, _M);
	this->B = T_B;
	
	this->Pi = init->PiIinit(_N);
	this->N = _N;
	this->M = _M;

}

HMM::~HMM() {
	delete_array(this->A, N, N);
	delete_array(this->B, M, N); // remember, B is transposed
	delete[] this->Pi;
	if (this->alpha)
		delete_array(this->alpha, T, N);
	if (this->beta)
		delete_array(this->beta, T, N);

	// TODO: delete the gammas, coeffs, and di-gammas
}

int HMM::getStateAtT(unsigned int t) {
	float max = -1.0f;
	int max_idx = -1;
	for (unsigned int i = 0; i < N; i++) {
		float state_prob = this->gamma[t][i];
		if (max < state_prob) {
			max_idx = i;
			max = state_prob;
		}
	}
	return max_idx;
}

int* HMM::getIdealStateSequence(unsigned int* obs, unsigned int size) {
	alphaPass(obs, size); // calculate alpha
	betaPass(obs, size); // calculate beta
	calcSeqProb(); // calculate P(O | lm)
	calcGamma(obs, size); // calculate the gammas and di-gammas

	print_matrix(gamma, T, N, true);

	int* r_array = new int[size];
	for (unsigned int t = 0; t < size; t++) {
		r_array[t] = getStateAtT(t);
	}

	return r_array;
}

void HMM::alphaPass(unsigned int* obs, unsigned int size) {
	alpha = new float* [size];
	T = size;
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

void HMM::calcSeqProb() {
	if (this->alpha) {
		float sum = 0;
		for (unsigned int i = 0; i < N; i++)
			sum += this->alpha[T - 1][i]; // sum of all hidden state probabilities at at the last state
		this->seqProb = sum;
	}
}

void HMM::betaPass(unsigned int* obs, unsigned int size) {
	beta = new float* [size];
	T = size;
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

void HMM::calcGamma(unsigned int* obs, unsigned int size) {
	if (!this->gamma) {
		this->gamma = new float*[T];
		for (unsigned int i = 0; i < T; i++)
			this->gamma[i] = new float[N];
	}

	if (!this->digamma) {
		this->digamma = new float** [T-1];
		for (unsigned int i = 0; i < T-1; i++) {
			this->digamma[i] = new float* [N];
			for (unsigned int j = 0; j < N; j++)
				this->digamma[i][j] = new float[N];
		}
			
			
	}

	float div = 1.0f/this->seqProb;// 1/P(O | lm)
	for (unsigned int t = 0; t < T; t++) {
		for (unsigned int i = 0; i < N; i++) {
			this->gamma[t][i] = this->alpha[t][i] * this->beta[t][i] * div;
		}
	}

	for (unsigned int t = 0; t < T-1; t++) {
		for (unsigned int i = 0; i < N; i++) {
			// float sum = 0;
			for (unsigned int j = 0; j < N; j++) {
				float val = this->alpha[t][i] * this->A[i][j] * this->B[obs[t]][j] * this->beta[t][j] * div;
				this->digamma[t][i][j] = val;
				// sum += val;
			}
			//this->gamma[t][i] = sum;
		} // NOTE: we could calculate gamma using digamma by taking the sum over j
	}
}
