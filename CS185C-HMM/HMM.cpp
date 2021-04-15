#include "HMM.h"
#include "ProbInit.h"
#include "MatUtil.h"
#include <iostream>

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
}

int HMM::getStateAtT(float** gamma, unsigned int size, unsigned int t) {
	float max = -1.0f;
	int max_idx = -1;
	for (unsigned int i = 0; i < N; i++) {
		float state_prob = gamma[t][i];
		if (max < state_prob) {
			max_idx = i;
			max = state_prob;
		}
	}
	return max_idx;
}

int* HMM::getIdealStateSequence(unsigned int* obs, unsigned int size) {
	float** alpha = alloc_mat(size, N);
	float** beta = alloc_mat(size, N);
	float** gamma = alloc_mat(size, N);
	float*** digamma = alloc_mat3(size, N, N);
	float* coeffs = alloc_vec(size);


	alphaPass(obs, size, alpha, coeffs); // calculate alpha
	betaPass(obs, size, beta, coeffs); // calculate beta

	calcGamma(obs, size, alpha, beta, gamma); // calculate the gammas and di-gammas
	calcDigamma(obs, size, alpha, beta, digamma);

	print_matrix(gamma, size, N, true);

	int* r_array = new int[size];
	for (unsigned int t = 0; t < size; t++) {
		r_array[t] = getStateAtT(gamma, size, t);
	}

	delete_array(alpha, size, N);
	delete_array(beta, size, N);
	delete_array(gamma, size, N);
	delete_array3(digamma, size, N, N);
	delete[] coeffs;

	return r_array;
}

void HMM::alphaPass(unsigned int* obs, unsigned int size, float** alpha, float* coeffs) {

	float val;
	coeffs[0] = 0.0f;
	for (unsigned int i = 0; i < N; i++) {
		val = Pi[i] * B[obs[0]][i];
		alpha[0][i] = val; // B has been transposed to improve spatial locality
		coeffs[0] += val;
	}
	
	float div = 1.0f / coeffs[0];
	coeffs[0] = div;
	for (unsigned int i = 0; i < N; i++) // scale the values s.t. alpha[0][0] + alpha[0][1] + ... = 1
		alpha[0][i] *= div;

	for (unsigned int t = 1; t < size; t++) {
		coeffs[t] = 0.0f;
		for (unsigned int i = 0; i < N; i++) {
			float sum = 0;
			for (unsigned int j = 0; j < N; j++) {
				sum += alpha[t - 1][j] * A[j][i]; // probability that we'd see the previous hidden state * the probability we'd transition to this new hidden state
			}
			val = sum * B[obs[t]][i];
			alpha[t][i] = val;
			coeffs[t] += val;
			 
		}

		div = 1.0f / coeffs[t];
		coeffs[t] = div;
		for (unsigned int i = 0; i < N; i++)  // scale the values s.t. alpha[t][0] + alpha[t][1] + ... = 1
			alpha[t][i] *= div;

	}
	
}

float HMM::calcSeqProb(float** alpha, unsigned int size) {
	float seqProb = 0.0f;
	if (alpha) {
		float sum = 0;
		for (unsigned int i = 0; i < N; i++)
			sum += alpha[size - 1][i]; // sum of all hidden state probabilities at at the last state
		seqProb = sum;
	}
	return seqProb;
}

void HMM::betaPass(unsigned int* obs, unsigned int size, float** beta, float* coeffs) {
	for (unsigned int i = 0; i < N; i++)
		beta[size - 1][i] = coeffs[size-1];

	for (int t = size-2; t >= 0; t--) {
		float ct = coeffs[t];
		for (unsigned int i = 0; i < N; i++) {
			float sum = 0;
			for (unsigned int j = 0; j < N; j++)
				sum += A[i][j] * B[obs[t + 1]][j] * beta[t + 1][j];

			beta[t][i] = sum*ct;
		}
	}

}

void HMM::calcGamma(unsigned int* obs, unsigned int size, float** alpha, float** beta, float** gamma) {
	float seqProb = this->calcSeqProb(alpha, size);
	float div = 1.0f/seqProb;// 1/P(O | lm)
	float val;
	for (unsigned int t = 0; t < size; t++) {
		float scale = 1e-10f;
		for (unsigned int i = 0; i < N; i++) {
			val = alpha[t][i] * beta[t][i] * div;
			gamma[t][i] = val;
			scale += val;
		}

		scale = 1.0f / scale;
		for (unsigned int i = 0; i < N; i++)
			gamma[t][i] *= scale;
	}
}

void HMM::calcDigamma(unsigned int* obs, unsigned int size, float** alpha, float** beta, float*** digamma) {
	float seqProb = this->calcSeqProb(alpha, size);
	float div = 1.0f / seqProb;// 1/P(O | lm)

	for (unsigned int t = 0; t < size - 1; t++) {
		for (unsigned int i = 0; i < N; i++) {
			// float sum = 0;
			float scale = 1e-10f;
			for (unsigned int j = 0; j < N; j++) {
				float val = alpha[t][i] * this->A[i][j] * this->B[obs[t]][j] * beta[t][j] * div;
				digamma[t][i][j] = val;
				// sum += val;
			}
			//this->gamma[t][i] = sum;
		} // NOTE: we could calculate gamma using digamma by taking the sum over j
	}
}

void HMM::applyAdjust(unsigned int* obs, unsigned int size, float** gamma, float*** digamma) {
	for (unsigned int i = 0; i < N; i++) {
		this->Pi[i] = gamma[0][i]; // use our calculated initial probability from gamma
	}

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			float digamma_sum = 0.0f;
			float gamma_sum = 0.0f;
			for (unsigned int t = 0; t < size-1; t++) {
				digamma_sum += digamma[t][i][j]; // ouch, cache hurty
				gamma_sum += gamma[t][i];
			}
			this->A[i][j] = digamma_sum / gamma_sum; // assign our new transition probability for Aij
		}
	}

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int k = 0; k < M; k++) {
			float gamma_obs_sum = 0.0f;
			float gamma_total_sum = 0.0f;
			for (unsigned int t = 0; t < size; t++) {
				float cur_gamma = gamma[t][i];
				gamma_total_sum += cur_gamma;
				if (obs[t] == i)
					gamma_obs_sum += cur_gamma;
			}

			this->B[k][i] = gamma_obs_sum/gamma_total_sum; //re-estimate our Bik

		}
	}
}