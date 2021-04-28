#include "HMM.h"
#include "ProbInit.h"
#include "MatUtil.h"
#include "DataSet.h"
#include <math.h>
#include <thread>
#include <immintrin.h>
#include <xmmintrin.h>

HMM::HMM() {

}

HMM::HMM(float* _A, float* _B, float* _Pi, unsigned int _N, unsigned int _M) {
	this->A = _A;
	this->A_T = transpose(this->A, _N, _N);
	this->B = transpose(_B, _N, _M);
	delete_array(_B, _N, _M);
	this->Pi = _Pi;
	
	this->N = _N;
	this->M = _M;
}

HMM::HMM(const HMM& o): native_symbolmap(o.native_symbolmap) {
	this->N = o.N;
	this->M = o.M;
	this->A = alloc_mat(o.A, N, N);
	this->A_T = alloc_mat(o.A_T, N, N);
	this->B = alloc_mat(o.B, M, N);
	this->Pi = alloc_vec(o.Pi, N);
}

HMM& HMM::operator=(const HMM& o) {
	delete_array(A, N, N);
	delete_array(A_T, N, N);
	delete_array(B, M, N);
	delete[] Pi;

	native_symbolmap = o.native_symbolmap;
	this->N = o.N;
	this->M = o.M;
	this->A = alloc_mat(o.A, N, N);
	this->A_T = alloc_mat(o.A_T, N, N);
	this->B = alloc_mat(o.B, M, N);
	this->Pi = alloc_vec(o.Pi, N);
	return *this;
}

HMM::HMM(unsigned int _N, unsigned int _M, ProbInit* initializer) {
	DefaultProbInit def_init;
	ProbInit* init = initializer ? initializer : &def_init;
	this->A = init->AInit(_N);
	this->B = init->BInit(_N, _M);
	this->A_T = transpose(this->A, _N, _N);
	float* T_B = transpose(this->B, _N, _M); // transpose our B array for simplified processing
	delete_array(this->B, _N, _M);
	this->B = T_B;
	
	this->Pi = init->PiIinit(_N);
	this->N = _N;
	this->M = _M;

}

HMM::~HMM() {
	delete_array(this->A, N, N);
	delete_array(this->B, M, N); // remember, B is transposed
	delete_array(this->A_T, N, N);
	delete[] this->Pi;
}

unsigned int HMM::getM() { return M; }
unsigned int HMM::getN() { return N; }

void HMM::reset(ProbInit* initializer) {
	delete_array(this->A, N, N);
	delete_array(this->B, M, N); // remember, B is transposed
	delete_array(this->A_T, N, N);
	delete[] this->Pi;

	DefaultProbInit def_init;
	ProbInit* init = initializer ? initializer : &def_init;
	this->A = init->AInit(N);
	this->B = init->BInit(N, M);
	this->A_T = transpose(this->A, N, N);
	float* T_B = transpose(this->B, N, M); // transpose our B array for simplified processing
	delete_array(this->B, N, M);
	this->B = T_B;

	this->Pi = init->PiIinit(N);
}

int HMM::getStateAtT(float* gamma, unsigned int size, unsigned int t) {
	float max = -1.0f;
	int max_idx = -1;
	for (unsigned int i = 0; i < N; i++) {
		float state_prob = gamma[t*N + i];
		if (max < state_prob) {
			max_idx = i;
			max = state_prob;
		}
	}
	return max_idx;
}

void HMM::setDataMapper(const DataMapper& o) {
	native_symbolmap = o;
}

int* HMM::getIdealStateSequence(unsigned int* obs, unsigned int size) {
	float* alpha = alloc_mat(size, N);
	float* beta = alloc_mat(size, N);
	float* gamma = alloc_mat(size, N);
	float* digamma = alloc_mat3(size, N, N);
	float* coeffs = alloc_vec(size);


	alphaPass(obs, size, alpha, coeffs); // calculate alpha
	betaPass(obs, size, beta, coeffs); // calculate beta

	calcGamma(obs, size, alpha, beta, gamma, digamma); // calculate the gammas and di-gammas

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

inline float fastsum8_m256(__m256 vals) {
	__m128 hi = _mm256_extractf128_ps(vals, 1); // take out the top 4 floats
	__m128 lo = _mm256_castps256_ps128(vals); // take out the bottom 4 floats
	hi = _mm_add_ps(hi, lo); // sum the top with the bottom, assign to hi
	// fold 2
	lo = _mm_unpacklo_ps(hi, hi); // lo = [h1, h1, h2, h2]
	hi = _mm_unpackhi_ps(hi, hi); // hi = [h3, h3, h4, h4]
	hi = _mm_add_ps(hi, lo); // hi = [h1+h3, h1+h3, h2+h4, h2+h4]
	// fold 3
	lo = _mm_unpacklo_ps(hi, hi); // lo = [h1+h3, h1+h3, h1+h3, h1+h3]
	hi = _mm_unpackhi_ps(hi, hi); // hi = [h2+h4, h2+h4, h2+h4, h2+h4]
	hi = _mm_add_ps(hi, lo); // all 4 entires contain the sum
	// to obtain the sum of all elements, we simply need to extract the first and second element of hi and add them:
	return _mm_cvtss_f32(hi);
}

inline float fastsum8(float* arr) {
	__m256 vals = _mm256_loadu_ps(arr);
	return fastsum8_m256(vals);
}

void HMM::alphaPass(unsigned int* obs, unsigned int size, float* alpha, float* coeffs) const {

	float val;
	coeffs[0] = 0.0f;
	unsigned int obs_idx = obs[0] * N;
	unsigned int i = 0;
	if (N > 8)
	for (; i < N-8; i+=8) {
		__m256 pi_v = _mm256_loadu_ps(&Pi[i]);
		__m256 b_v = _mm256_loadu_ps(&B[obs_idx + i]);
		__m256 prod = _mm256_mul_ps(pi_v, b_v);
		_mm256_storeu_ps(&alpha[i], prod);
		coeffs[0] += fastsum8_m256(prod);
	}
	for (; i < N; i++) {
		val = Pi[i] * B[obs_idx + i];
		alpha[/* 0*N+ */i] = val; // B has been transposed to improve spatial locality
		coeffs[0] += val;
	}
	
	float div = 1.0f / coeffs[0];
	coeffs[0] = div;
	__m256 div_v = _mm256_broadcast_ss(&div);
	i = 0;
	if (N > 8)
	for (; i < N - 8; i += 8) { // scale the values s.t. alpha[0][0] + alpha[0][1] + ... = 1
		__m256 prod = _mm256_loadu_ps(&alpha[i]);
		prod = _mm256_mul_ps(prod, div_v); // alpha[i...i+7] = alpha[i...i+7] * div
		_mm256_storeu_ps(&alpha[i], prod);
	}
	for (; i < N; i++)
		alpha[i] *= div;

	for (unsigned int t = 1; t < size; t++) {
		coeffs[t] = 0.0f;
		unsigned int last_alpha_idx = (t - 1) * N;
		unsigned int cur_alpha_idx = t * N;
		obs_idx = obs[t] * N;
		for (i = 0; i < N; i++) {
			float sum = 0;
			unsigned int j = 0;
			if (N > 8)
			for (; j < N-8; j+=8) {
				__m256 alpha_v = _mm256_loadu_ps(&alpha[last_alpha_idx + j]);
				__m256 a_v = _mm256_loadu_ps(&A_T[i * N + j]);
				__m256 prod = _mm256_mul_ps(alpha_v, a_v);
				sum += fastsum8_m256(prod);
			}
			for(; j < N; j++)
				sum += alpha[last_alpha_idx + j] * A_T[i*N + j]; // probability that we'd see the previous hidden state * the probability we'd transition to this new hidden state

			val = sum * B[obs_idx + i];
			alpha[cur_alpha_idx + i] = val;
			coeffs[t] += val;
			 
		}

		div = 1.0f / coeffs[t];
		coeffs[t] = div;
		div_v = _mm256_broadcast_ss(&div);
		i = 0;
		if (N > 8)
		for (; i < N - 8; i += 8) {
			__m256 prod = _mm256_loadu_ps(&alpha[cur_alpha_idx + i]);
			prod = _mm256_mul_ps(prod, div_v); // alpha[i...i+7] = alpha[i...i+7] * div
			_mm256_storeu_ps(&alpha[cur_alpha_idx + i], prod);
		}

		for (; i < N; i++)  // scale the values s.t. alpha[t][0] + alpha[t][1] + ... = 1
			alpha[cur_alpha_idx + i] *= div;


	}
	
}

float HMM::calcSeqProb(float* alpha, unsigned int size) {
	float seqProb = 0.0f;
	if (alpha) {
		float sum = 0;
		for (unsigned int i = 0; i < N; i++)
			sum += alpha[(size - 1) * N + i]; // sum of all hidden state probabilities at at the last state
		seqProb = sum;
	}
	return seqProb;
}

void HMM::betaPass(unsigned int* obs, unsigned int size, float* beta, float* coeffs) const {
	unsigned int beta_end = (size - 1) * N;
	for (unsigned int i = 0; i < N; i++)
		beta[beta_end + i] = coeffs[size-1];

	for (int t = size-2; t >= 0; t--) {
		unsigned int obs_idx = obs[t + 1] * N;
		unsigned int cur_beta_idx = t * N;
		unsigned int beta_idx = (t + 1) * N;
		float ct = coeffs[t];
		for (unsigned int i = 0; i < N; i++) {
			float sum = 0.0f;
			unsigned int row_idx = i * N;
			unsigned int j = 0;
			if (N > 8)
			for (; j < N - 8; j += 8) {
				__m256 a_v = _mm256_loadu_ps(&A[row_idx + j]);
				__m256 b_v = _mm256_loadu_ps(&B[obs_idx + j]);
				__m256 beta_v = _mm256_loadu_ps(&beta[beta_idx + j]);
				__m256 prod = _mm256_mul_ps(a_v, b_v);
				prod = _mm256_mul_ps(prod, beta_v);

				sum += fastsum8_m256(prod);
			}
			for (; j < N; j++) {
				sum += A[row_idx + j] * B[obs_idx + j] * beta[beta_idx + j];
			}
			beta[cur_beta_idx + i] = sum*ct;
		}
	}

}

void HMM::calcGamma(unsigned int* obs, unsigned int size, float* alpha, float* beta, float* gamma, float* digamma) {
	float div = 1e-10f;
	float val;
	for (unsigned int t = 0; t < size-1; t++) {
		div = 1e-10f;
		for (unsigned int i = 0; i < N; i++) {
			for (unsigned int j = 0; j < N; j++) {
				div += alpha[t*N + i] * A[i*N + j] * B[obs[t + 1]*N + j] * beta[(t + 1)*N + j];
			}
		}
		div = 1.0f / div;
		for (unsigned int i = 0; i < N; i++) {
			gamma[t*N + i] = 0.0f;
			for (unsigned int j = 0; j < N; j++) {
				val = alpha[t*N + i] * A[i*N + j] * B[obs[t + 1]*N + j] * beta[(t + 1)*N + j] * div;
				digamma[t*N*N + i*N + j] = val;
				gamma[t*N + i] += val;
			}
		}
	}
	div = 1e-10f;
	for (unsigned int i = 0; i < N; i++)
		div += alpha[(size - 1)*N + i];
	div = 1.0f / div;

	for (unsigned int i = 0; i < N; i++)
		gamma[(size - 1)*N + i] = alpha[(size - 1)*N + i] * div;
}


void HMM::calcGamma(unsigned int* obs, unsigned int size, unsigned int t, float* alpha, float* beta, float* gamma, float* digamma) {
	float div = 1e-10f;
	float val;
	if (t < size-1) {
		for (unsigned int i = 0; i < N; i++) {
			__m256 alpha_v = _mm256_broadcast_ss(&alpha[t * N + i]);
			unsigned int j = 0;
			if (N > 8)
			for (j = 0; j < N - 8; j+=8) {

				// compute the un-normalized digamma values and assign them to digamma
				__m256 a_v = _mm256_loadu_ps(&A[i * N + j]);
				__m256 b_v = _mm256_loadu_ps(&B[obs[t + 1] * N + j]);
				__m256 beta_v = _mm256_loadu_ps(&beta[(t + 1) * N + j]);
				__m256 prod1 = _mm256_mul_ps(alpha_v, a_v);
				__m256 prod2 = _mm256_mul_ps(b_v, beta_v);
				prod1 = _mm256_mul_ps(prod1, prod2);

				_mm256_storeu_ps(&digamma[i * N + j], prod1); // update digamma
				div += fastsum8_m256(prod1);
			}
			for (; j < N; j++)
				div += alpha[t * N + i] * A[i * N + j] * B[obs[t + 1] * N + j] * beta[(t + 1) * N + j];
		}
		div = 1.0f / div;

		__m256 div_v = _mm256_broadcast_ss(&div);
		for (unsigned int i = 0; i < N; i++) {
			gamma[i] = 0.0f;
			unsigned int j = 0;
			if (N > 8)
			for (j = 0; j < N-8; j+=8) {
				// digamma calculation for 8 elements
				__m256 prod1 = _mm256_loadu_ps(&digamma[i*N + j]);
				prod1 = _mm256_mul_ps(prod1, div_v); // normalize the elements
				_mm256_storeu_ps(&digamma[i * N + j], prod1); // update digamma
				gamma[i] += fastsum8_m256(prod1); // add the 8 elements we just processed to gamma
			}
			for (; j < N; j++) { // cleanup left overs
				val = alpha[t * N + i] * A[i * N + j] * B[obs[t + 1] * N + j] * beta[(t + 1) * N + j] * div;
				digamma[i * N + j] = val;
				gamma[i] += val;
			}
		}
	}
	else if (t == size - 1) {
		for (unsigned int i = 0; i < N; i++)
			div += alpha[(size - 1)*N + i];
		div = 1.0f / div;

		for (unsigned int i = 0; i < N; i++)
			gamma[i] = alpha[(size - 1)*N + i] * div;
	}
}

// THIS ONLY WORKS FOR 1 SEQUENCE
void HMM::applyAdjust(unsigned int* obs, unsigned int size, float* gamma, float* digamma) {
	for (unsigned int i = 0; i < N; i++) {
		this->Pi[i] = gamma[i]; // use our calculated initial probability from gamma
	}

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			float digamma_sum = 0.0f;
			float gamma_sum = 0.0f;
			for (unsigned int t = 0; t < size-1; t++) {
				digamma_sum += digamma[t*N*N + i*N + j]; // ouch, cache hurty
				gamma_sum += gamma[t*N + i];
			}
			this->A[i*N + j] = digamma_sum / gamma_sum; // assign our new transition probability for Aij
		}
	}

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int k = 0; k < M; k++) {
			float gamma_obs_sum = 0.0f;
			float gamma_total_sum = 0.0f;
			for (unsigned int t = 0; t < size; t++) {
				float cur_gamma = gamma[t*N + i];
				gamma_total_sum += cur_gamma;
				if (obs[t] == k)
					gamma_obs_sum += cur_gamma;
			}

			this->B[k*N + i] = gamma_obs_sum/gamma_total_sum; //re-estimate our Bik

		}
	}
}

void HMM::applyAdjust(const AdjustmentAccumulator& accum) {
	for (unsigned int i = 0; i < N; i++) {
		this->Pi[i] = accum.pi_accum[i]; // use our calculated initial probability from gamma
	}

	for (unsigned int i = 0; i < N; i++) {
		unsigned int row_ind = i * N;
		unsigned int j = 0;
		if (N > 8)
		for (j = 0; j < N-8; j+=8) {
			__m256 digamma_v = _mm256_loadu_ps(&accum.A_digamma_accum[row_ind + j]);
			__m256 gamma_v = _mm256_loadu_ps(&accum.A_gamma_accum[row_ind + j]);
			_mm256_storeu_ps(&this->A[row_ind + j], _mm256_div_ps(digamma_v, gamma_v));
		}

		for (; j < N; j++) {
			unsigned int col_ind = row_ind + j;
			this->A[col_ind] = accum.A_digamma_accum[col_ind] / accum.A_gamma_accum[col_ind];// assign our new transition probability for Aij
		}
	}

	for (unsigned int i = 0; i < N; i++) {
		unsigned int row_ind = i * M;
		unsigned int k = 0;
		if (M > 8)
		for (k = 0; k < M-8; k+=8) {
			__m256 obs_v = _mm256_loadu_ps(&accum.B_obs_accum[row_ind + k]);
			__m256 gamma_v = _mm256_loadu_ps(&accum.B_gamma_accum[row_ind + k]);
			_mm256_storeu_ps(&this->B[row_ind + k], _mm256_div_ps(obs_v, gamma_v));
		}
		for (; k < M; k++) {
			unsigned int col_ind = row_ind + k;
			this->B[col_ind] = accum.B_obs_accum[col_ind] / accum.B_gamma_accum[col_ind]; //re-estimate our Bik
		}
	}

	transpose_emplace(this->A, N, N, this->A_T);
}

HMM::AdjustmentAccumulator::~AdjustmentAccumulator() {
	delete_array(this->A_digamma_accum, N, N);
	delete_array(this->A_gamma_accum, N, N);
	delete_array(this->B_gamma_accum, M, N);
	delete_array(this->B_obs_accum, M, N);
	delete[] this->pi_accum;
	this->initialized = false;
}

void HMM::AdjustmentAccumulator::initialize(unsigned int N, unsigned int M) {
	this->N = N;
	this->M = M;
	this->accumLogProb = 0.0f;
	this->accumLogProb_v = 0.0f;
	this->A_digamma_accum = alloc_mat(N, N);
	this->A_gamma_accum = alloc_mat(N, N);
	this->B_gamma_accum = alloc_mat(M, N);
	this->B_obs_accum = alloc_mat(M, N);
	this->pi_accum = new float[N];
	this->initialized = true;
}

void HMM::AdjustmentAccumulator::reset() {
	if (!initialized)
		std::cout << "ERROR, attempting to reset an accumulator before it has been initialized" << std::endl;
	for (unsigned int i = 0; i < N; i++) {
		this->pi_accum[i] = 0.0f;
		for (unsigned int j = 0; j < N; j++) {
			this->A_digamma_accum[i*N + j] = 0.0f;
			this->A_gamma_accum[i*N + j] = 0.0f;
		}
		for (unsigned int j = 0; j < M; j++) {
			this->B_gamma_accum[j*N + i] = 0.0f;
			this->B_obs_accum[j*N + i] = 0.0f;
		}
	}
	this->accumLogProb = 0.0f;
	this->accumLogProb_v = 0.0f;
	this->count = 0;

}

void HMM::accumAdjust(unsigned int* obs, unsigned int size, float* gamma, float* digamma, AdjustmentAccumulator& accum) {
	for (unsigned int i = 0; i < N; i++) {
		accum.pi_accum[i] += gamma[i]; // use our calculated initial probability from gamma
	}

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			float digamma_sum = 0.0f;
			float gamma_sum = 0.0f;
			for (unsigned int t = 0; t < size - 1; t++) {
				digamma_sum += digamma[t*N*N + i*N + j]; // ouch, cache hurty
				gamma_sum += gamma[t*N + i];
			}

			accum.A_digamma_accum[i*N + j] += digamma_sum;
			accum.A_gamma_accum[i*N + j] += gamma_sum;

		}
	}

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int k = 0; k < M; k++) {
			float gamma_obs_sum = 0.0f;
			float gamma_total_sum = 0.0f;
			for (unsigned int t = 0; t < size - 1; t++) {
				float cur_gamma = gamma[t*N + i];
				gamma_total_sum += cur_gamma;
				if (obs[t] == k)
					gamma_obs_sum += cur_gamma;
			}

			accum.B_gamma_accum[k*N + i] += gamma_total_sum;
			accum.B_obs_accum[k*N + i] += gamma_obs_sum;

		}
	}
	accum.count += 1;
}

void HMM::print_mats() const {
	print_vector(Pi, N, 5U);
	print_matrix(A, N, N, false, 5U);
	print_matrix(B, M, N, false, 5U);
}

void HMM::accumAdjust(unsigned int* obs, unsigned int size, unsigned int t, float* gamma, float* digamma, AdjustmentAccumulator& accum) {
	if (t == 0) {
		unsigned int i = 0;
		if (N > 8)
		for (; i < N-8; i+=8) {
			__m256 pi_v = _mm256_loadu_ps(&accum.pi_accum[i]);
			__m256 gamma_v = _mm256_loadu_ps(&gamma[i]);
			_mm256_storeu_ps(&accum.pi_accum[i], _mm256_add_ps(pi_v, gamma_v));
		}
		for (; i < N; i++) {
			accum.pi_accum[i] += gamma[i]; // use our calculated initial probability from gamma
		}
	}
	if (t < size - 1) {
		for (unsigned int i = 0; i < N; i++) {
			unsigned int row_ind = i * N;
			__m256 gamma_v = _mm256_broadcast_ss(&gamma[i]);
			unsigned int j = 0;
			if (N > 8)
			for (; j < N-8; j+=8) {
				unsigned int col_ind = row_ind + j;
				__m256 numer_o = _mm256_loadu_ps(&accum.A_digamma_accum[col_ind]);
				__m256 numer_n = _mm256_loadu_ps(&digamma[col_ind]);
				__m256 denom_o = _mm256_loadu_ps(&accum.A_gamma_accum[col_ind]);

				_mm256_storeu_ps(&accum.A_digamma_accum[col_ind], _mm256_add_ps(numer_o, numer_n));
				_mm256_storeu_ps(&accum.A_gamma_accum[col_ind], _mm256_add_ps(denom_o, gamma_v));
			}
			for (; j < N; j++) {
				unsigned int col_ind = row_ind + j;
				accum.A_digamma_accum[col_ind] += digamma[col_ind];
				accum.A_gamma_accum[col_ind] += gamma[i];

			}
		}
		for (unsigned int k = 0; k < M; k++) {
			unsigned int row_ind = k * N;
			float mult = obs[t] == k ? 1.0f : 0.0f;
			unsigned int i = 0;
			/*for (; i < N-8; i+=8) { // for some reason, this seems to hinder training performance, idk if it's due to rounding issues or what
				unsigned int col_ind = row_ind + i;
				__m256 denom_o = _mm256_loadu_ps(&accum.B_gamma_accum[col_ind]);
				__m256 numer_o = _mm256_loadu_ps(&accum.B_obs_accum[col_ind]);
				__m256 denom_n = _mm256_broadcast_ss(&gamma[i]);
				__m256 numer_n = obs[t] == k ? _mm256_add_ps(numer_o, denom_n) : numer_o;

				_mm256_storeu_ps(&accum.B_gamma_accum[col_ind], _mm256_add_ps(denom_o, denom_n));
				_mm256_storeu_ps(&accum.B_obs_accum[col_ind], numer_n);
			}*/
			for (; i < N; i++) {
				unsigned int col_ind = row_ind + i;
				float cur_gamma = gamma[i];
				accum.B_gamma_accum[col_ind] += cur_gamma;
				accum.B_obs_accum[col_ind] += cur_gamma*mult;

			}
		}
	}
}


void HMM::trainModel(const HMMDataSet& dataset, unsigned int iterations, unsigned int n_folds, unsigned int fold_index, bool early_stop, unsigned int patience) {

	NFoldIterator iter = dataset.getIter(n_folds);
	AdjustmentAccumulator accum;
	accum.initialize(N, M);
	unsigned int max_length = dataset.getMaxLength();
	
	NFoldTrainingManager trainer;
	trainer.data = dataset.getDataPtr();
	trainer.lengths = dataset.getLengthsPtr();
	trainer.case_count = dataset.getSize();


	trainer.multi_thread_fold(n_folds, N, M, this, max_length);

	float last_training_score, last_validation_score;
	trainer.score_fold(0, &accum);
	last_training_score = accum.accumLogProb;
	last_validation_score = accum.accumLogProb_v;
	unsigned int osc = 0;

	for (unsigned int epoch = 0; epoch < iterations; epoch++) {
		std::cout << "----------------------------------------- EPOCH " << epoch << " -----------------------------------------" << std::endl;

		trainer.train_fold(fold_index, &accum);


		applyAdjust(accum);

		// calculate the score improvement
		trainer.score_fold(fold_index, &accum);
		float training_score_gain = accum.accumLogProb - last_training_score;
		float validation_score_gain = accum.accumLogProb_v - last_validation_score;

		last_validation_score = accum.accumLogProb_v;
		last_training_score = accum.accumLogProb;
		print_vector(Pi, N);
		print_matrix(A, N, N, false);
		//print_matrix(B, M, N, false);
		std::cout << "Training Score: " << accum.accumLogProb << std::endl;
		std::cout << "Validation Score: " << accum.accumLogProb_v << std::endl;
		std::cout << "Training Improvement: " << training_score_gain << std::endl;
		std::cout << "Validation Improvement: " << validation_score_gain << std::endl;

		// early stopping
		osc = validation_score_gain > 0 ? 0 : osc + (last_training_score > 0);
		if (early_stop && patience < osc)
			return;

	}
	std::cout << accum.accumLogProb_v << std::endl;
}

void HMM::NFoldTrainingManager::multi_thread_fold(unsigned int nfold, unsigned int _N, unsigned int _M, HMM* _hmm, unsigned int max_sequence_length) {
	std::vector<unsigned int**> fold_regions;
	std::vector<unsigned int*> fold_region_lengths;
	std::vector<unsigned int> fold_region_sizes;
	N = _N;
	M = _M;
	hmm = _hmm;
	fold_count = nfold;
	unsigned int fold_size = case_count / nfold;
	unsigned int** cur_region = data;
	unsigned int* cur_lengths = lengths;

	fold_regions.push_back(cur_region);
	fold_region_lengths.push_back(cur_lengths);
	fold_region_sizes.push_back(fold_size);
	unsigned int case_accum = fold_size;

	for (unsigned int i = 1; i < nfold-1; i++) {
		cur_region += fold_size;
		cur_lengths += fold_size;

		case_accum += fold_size;
		fold_regions.push_back(cur_region);
		fold_region_lengths.push_back(cur_lengths);
		fold_region_sizes.push_back(fold_size);
	}

	cur_region += fold_size;
	cur_lengths += fold_size;
	fold_regions.push_back(cur_region);
	fold_region_lengths.push_back(cur_lengths);
	fold_region_sizes.push_back(case_count - case_accum);
	
	accumulators = new AdjustmentAccumulator[nfold];
	workers = new TrainingWorker[nfold];
	for (unsigned int i = 0; i < fold_regions.size(); i++) {
		accumulators[i].initialize(N, M);
		workers[i].initialize(
			fold_regions[i],
			fold_region_lengths[i],
			fold_region_sizes[i],
			max_sequence_length,
			N,
			M,
			hmm,
			(accumulators+i)
			);
	}
}

HMM::NFoldTrainingManager::~NFoldTrainingManager() {
	delete[] accumulators;
	delete[] workers;
}

void HMM::NFoldTrainingManager::score_fold(unsigned int fold_index, AdjustmentAccumulator* master) {
	std::vector<std::thread> threads;
	for (unsigned int i = 0; i < fold_count; i++) {
		accumulators[i].reset();
		threads.emplace_back(std::thread(&(HMM::TrainingWorker::valid_work), (HMM::TrainingWorker*)&workers[i]));
	}

	for (auto& th : threads) {
		th.join();
	}

	float div = 1.0f / (float)(case_count - accumulators[fold_index].count);
	// combine all acumulations in the master accumulator
	master->reset();
	for (unsigned int i = 0; i < fold_count; i++) {
		AdjustmentAccumulator* cur = &accumulators[i];

		if (i != fold_index) {
			master->count += cur->count;
			master->accumLogProb += cur->accumLogProb_v * div;
		}
		else {
			master->accumLogProb_v = cur->accumLogProb_v / ((float)cur->count + 1e-6f);
		}
	}
}

void HMM::NFoldTrainingManager::train_fold(unsigned int fold_index, AdjustmentAccumulator* master) {
	std::vector<std::thread> threads;
	for (unsigned int i = 0; i < fold_count; i++) {
		accumulators[i].reset();
		if (i != fold_index) {
			threads.emplace_back(std::thread(&(HMM::TrainingWorker::train_work), (HMM::TrainingWorker*)&workers[i]));
		}
		else {
			threads.emplace_back(std::thread(&(HMM::TrainingWorker::valid_work), (HMM::TrainingWorker*)&workers[i]));
		}
	}

	for (auto& th : threads) {
		th.join();
	}

	float div = 1.0f / (float)(case_count-accumulators[fold_index].count);
	// combine all acumulations in the master accumulator
	master->reset();
	for (unsigned int i = 0; i < fold_count; i++) {
		AdjustmentAccumulator* cur = &accumulators[i];
		
		if (i != fold_index) {
			master->count += cur->count;
			master->accumLogProb += cur->accumLogProb*div;
			for (unsigned int i = 0; i < N; i++) {
				master->pi_accum[i] += cur->pi_accum[i]*div;
				unsigned int row_index = i * N;
				for (unsigned int j = 0; j < N; j++) {
					unsigned int location = row_index + j;
					master->A_digamma_accum[location] += cur->A_digamma_accum[location]*div;
					master->A_gamma_accum[location] += cur->A_gamma_accum[location]*div;
				}

				for (unsigned int j = 0; j < M; j++) {
					unsigned int location = j * N + i;
					master->B_gamma_accum[location] += cur->B_gamma_accum[location]*div;
					master->B_obs_accum[location] += cur->B_obs_accum[location]*div;
				}
			}
		}
		else {
			master->accumLogProb_v = cur->accumLogProb_v / ((float)cur->count + 1e-6f);
		}
	}
}

void HMM::TrainingWorker::initialize(
	unsigned int** _case_data,
	unsigned int* _case_lengths,
	unsigned int _case_count,
	unsigned int _sequence_count,
	unsigned int _N,
	unsigned int _M,
	HMM* _hmm,
	AdjustmentAccumulator* _accumulator
) {
	case_data = _case_data;
	case_lengths = _case_lengths;
	case_count = _case_count;
	sequence_count = _sequence_count;
	N = _N;
	M = _M;
	hmm = _hmm;
	accumulator = _accumulator;


	// note: These are too big, the sequence count can be > 20,000 in some cases: you need to find a way to use less memory
	// Solution: don't pre-compute digamma and gamma along t, just compute it for one t at a time, store it, and then accumulate each computation
	alpha = alloc_mat(sequence_count, N); // allocate enough space for the largest observation sequence
	beta = alloc_mat(sequence_count, N);
	gamma = alloc_vec(N);
	digamma = alloc_mat(N, N);
	coeffs = alloc_vec(sequence_count);
}

HMM::TrainingWorker::~TrainingWorker() {
	delete_array(alpha, sequence_count, N); // allocate enough space for the largest observation sequence
	delete_array(beta, sequence_count, N);
	delete_array(digamma, N, N);
	if (coeffs)
		delete[] coeffs;

	if (gamma)
		delete[] gamma;
}

void HMM::TrainingWorker::train_work() {
	for (unsigned int i = 0; i < case_count; i++) {
		unsigned int* obs = case_data[i];
		unsigned int length = case_lengths[i];
		hmm->alphaPass(obs, length, alpha, coeffs);
		hmm->betaPass(obs, length, beta, coeffs); // calculate beta
		for (unsigned int i = 0; i < length; i++) {
			hmm->calcGamma(obs, length, i, alpha, beta, gamma, digamma); // calculate the gammas and di-gammas
			hmm->accumAdjust(obs, length, i, gamma, digamma, *accumulator); // add our adjustments to the accumulator
		}
		float logProb = 0.0f;
		for (unsigned int i = 0; i < length; i++)
			logProb += log(coeffs[i]);

		accumulator->accumLogProb += -logProb*(1.0f/length);
	}
}

void HMM::TrainingWorker::valid_work() {
	for (unsigned int i = 0; i < case_count; i++) {
		unsigned int* obs = case_data[i];
		unsigned int length = case_lengths[i];
		hmm->alphaPass(obs, length, alpha, coeffs);
		float logProb = 0.0f;
		for (unsigned int i = 0; i < length; i++)
			logProb += log(coeffs[i]);

		accumulator->accumLogProb_v += -logProb * (1.0f / length);
		accumulator->count++;
	}
}


void HMM::testClassifier(const HMMDataSet& positives, const HMMDataSet& negatives, float thresh) const {
	HMMDataSet remapped_pos = positives.getRemapped(native_symbolmap);
	unsigned int** pos_data = remapped_pos.getDataPtr();
	unsigned int* pos_l = remapped_pos.getLengthsPtr();
	HMMDataSet remapped_neg = negatives.getRemapped(native_symbolmap);
	unsigned int** neg_data = remapped_neg.getDataPtr();
	unsigned int* neg_l = remapped_neg.getLengthsPtr();
	unsigned int  tp=0, tn=0, fp=0, fn=0;

	unsigned int pos_size = remapped_pos.getSize();
	unsigned int neg_size = remapped_neg.getSize();
	
	unsigned int max_t_size = std::max(remapped_pos.getMaxLength(), remapped_neg.getMaxLength());
	float* alpha = alloc_mat(max_t_size, N);
	float* coeffs = alloc_vec(max_t_size);
	
	
	for (unsigned int i = 0; i < pos_size; i++) {
		unsigned int length = pos_l[i];
		alphaPass(pos_data[i], length, alpha, coeffs);

		float rating = 0.0f;
		for (unsigned int j = 0; j < length; j++)
			rating += log(coeffs[j]);
		tp += -rating * (1.0f / length) >= thresh;
		fn += -rating * (1.0f / length) < thresh;
	}

	for (unsigned int i = 0; i < neg_size; i++) {
		unsigned int length = neg_l[i];
		alphaPass(neg_data[i], length, alpha, coeffs);

		float rating = 0.0f;
		for (unsigned int j = 0; j < length; j++)
			rating += log(coeffs[j]);
		tn += -rating * (1.0f / length) < thresh;
		fp += -rating * (1.0f / length) >= thresh;
	}

	std::cout << "TPR: " << ((float)tp / (float)(tp + fn)) << std::endl;
	std::cout << "FPR: " << ((float)fp / (float)(tn + fp)) << std::endl;

	delete_array(alpha, max_t_size, N);
	delete[] coeffs;
}

void HMM::generateROC(const HMMDataSet& positives, const HMMDataSet& negatives, float * dest) const {

	HMMDataSet remapped_pos = positives.getRemapped(native_symbolmap);
	unsigned int** pos_data = remapped_pos.getDataPtr();
	unsigned int* pos_l = remapped_pos.getLengthsPtr();
	HMMDataSet remapped_neg = negatives.getRemapped(native_symbolmap);
	unsigned int** neg_data = remapped_neg.getDataPtr();
	unsigned int* neg_l = remapped_neg.getLengthsPtr();

	unsigned int pos_size = remapped_pos.getSize();
	unsigned int neg_size = remapped_neg.getSize();

	unsigned int max_t_size = std::max(remapped_pos.getMaxLength(), remapped_neg.getMaxLength());
	float* alpha = alloc_mat(max_t_size, N);
	float* coeffs = alloc_vec(max_t_size);

	std::vector<float> positive_probs;
	std::vector<float> negative_probs;
	// store a sorted array of 

	for (unsigned int i = 0; i < pos_size; i++) {
		unsigned int length = pos_l[i];
		alphaPass(pos_data[i], length, alpha, coeffs);

		float rating = 0.0f;
		for (unsigned int j = 0; j < length; j++)
			rating += log(coeffs[j]) * (1.0f / length);

		positive_probs.push_back(-rating);

	}

	for (unsigned int i = 0; i < neg_size; i++) {
		unsigned int length = neg_l[i];
		alphaPass(neg_data[i], length, alpha, coeffs);

		float rating = 0.0f;
		for (unsigned int j = 0; j < length; j++)
			rating += log(coeffs[j]) * (1.0f / length);

		negative_probs.push_back(-rating);

	}

	std::sort(positive_probs.begin(), positive_probs.end());
	for (auto debug : positive_probs)
		std::cout << debug << std::endl;
	std::sort(negative_probs.begin(), negative_probs.end());
	unsigned int index = 0;
	unsigned int j = 0;
	for (unsigned int i = 0; i < pos_size; i++) {
		float current_thresh = positive_probs[i];
		while (negative_probs[j] < current_thresh) j++;
		unsigned int TP = pos_size - i;
		unsigned int FN = i;
		unsigned int FP = neg_size - j;
		unsigned int TN = j;
		
		dest[index] = (float)TP / (float)(TP + FN); // TPR
		dest[index + 1] = (float)FP / (float)(TN + FP); // FPR
		index += 2;
	}
	
	j = 0;
	for (unsigned int i = 0; i < neg_size; i++) {
		float current_thresh = negative_probs[i];

		while (positive_probs[j] < current_thresh) j++;
		unsigned int TP = pos_size - j;
		unsigned int FN = j;
		unsigned int FP = neg_size - i;
		unsigned int TN = i;

		dest[index] = (float)TP / (float)(TP + FN); // TPR
		dest[index + 1] = (float)FP / (float)(TN + FP); // FPR
		index += 2;
	}

	delete_array(alpha, max_t_size, N);
	delete[] coeffs;
}