#include "HMM.h"
#include "ProbInit.h"
#include "MatUtil.h"
#include "DataSet.h"
#include <math.h>
#include <thread>

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

void HMM::calcGamma(unsigned int* obs, unsigned int size, float** alpha, float** beta, float** gamma, float*** digamma) {
	float div = 1e-10f;
	float val;
	for (unsigned int t = 0; t < size-1; t++) {
		div = 1e-10f;
		for (unsigned int i = 0; i < N; i++) {
			for (unsigned int j = 0; j < N; j++) {
				div += alpha[t][i] * A[i][j] * B[obs[t + 1]][j] * beta[t + 1][j];
			}
		}
		div = 1.0f / div;
		for (unsigned int i = 0; i < N; i++) {
			gamma[t][i] = 0.0f;
			for (unsigned int j = 0; j < N; j++) {
				val = alpha[t][i] * A[i][j] * B[obs[t + 1]][j] * beta[t + 1][j] * div;
				digamma[t][i][j] = val;
				gamma[t][i] += val;
			}
		}
	}
	div = 1e-10f;
	for (unsigned int i = 0; i < N; i++)
		div += alpha[size - 1][i];
	div = 1.0f / div;

	for (unsigned int i = 0; i < N; i++)
		gamma[size - 1][i] = alpha[size - 1][i] * div;



}

// THIS ONLY WORKS FOR 1 SEQUENCE
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

void HMM::applyAdjust(const AdjustmentAccumulator& accum) {
	float div = 1.0f / accum.count;
	for (unsigned int i = 0; i < N; i++) {
		this->Pi[i] = accum.pi_accum[i]*div; // use our calculated initial probability from gamma
	}

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			this->A[i][j] = accum.A_digamma_accum[i][j] / accum.A_gamma_accum[i][j];// assign our new transition probability for Aij
		}
	}

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int k = 0; k < M; k++) {
			this->B[k][i] = accum.B_obs_accum[k][i] / accum.B_gamma_accum[k][i]; //re-estimate our Bik
		}
	}
}

HMM::AdjustmentAccumulator::~AdjustmentAccumulator() {
	delete_array(this->A_digamma_accum, N, N);
	delete_array(this->A_gamma_accum, N, N);
	delete_array(this->B_gamma_accum, M, N);
	delete_array(this->B_obs_accum, M, N);
	delete[] this->pi_accum;
}

void HMM::AdjustmentAccumulator::initialize(unsigned int N, unsigned int M) {
	this->N = N;
	this->M = M;

	this->A_digamma_accum = alloc_mat(N, N);
	this->A_gamma_accum = alloc_mat(N, N);
	this->B_gamma_accum = alloc_mat(M, N);
	this->B_obs_accum = alloc_mat(M, N);
	this->pi_accum = new float[N];
}

void HMM::AdjustmentAccumulator::reset() {
	for (unsigned int i = 0; i < N; i++) {
		this->pi_accum[i] = 0.0f;
		for (unsigned int j = 0; j < N; j++) {
			this->A_digamma_accum[i][j] = 0.0f;
			this->A_gamma_accum[i][j] = 0.0f;
		}
		for (unsigned int j = 0; j < M; j++) {
			this->B_gamma_accum[i][j] = 0.0f;
			this->B_obs_accum[i][j] = 0.0f;
		}
	}
	this->accumLogProb = 0.0f;
	this->accumLogProb_v = 0.0f;
	this->count = 0;

}

void HMM::accumAdjust(unsigned int* obs, unsigned int size, float** gamma, float*** digamma, AdjustmentAccumulator& accum) {
	for (unsigned int i = 0; i < N; i++) {
		accum.pi_accum[i] += gamma[0][i]; // use our calculated initial probability from gamma
	}

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			float digamma_sum = 0.0f;
			float gamma_sum = 0.0f;
			for (unsigned int t = 0; t < size - 1; t++) {
				digamma_sum += digamma[t][i][j]; // ouch, cache hurty
				gamma_sum += gamma[t][i];
			}

			accum.A_digamma_accum[i][j] += digamma_sum;
			accum.A_gamma_accum[i][j] += gamma_sum;

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

			accum.B_gamma_accum[k][i] += gamma_total_sum;
			accum.B_obs_accum[k][i] += gamma_obs_sum;

		}
	}
	accum.count += 1;
}

void HMM::trainModel(const HMMDataSet& dataset, unsigned int iterations, unsigned int n_folds) {

	NFoldIterator iter = dataset.getIter(n_folds);
	AdjustmentAccumulator accum;
	accum.initialize(N, M);
	unsigned int max_length = dataset.getMaxLength();
	float** alpha = alloc_mat(max_length, N); // allocate enough space for the largest observation sequence
	float** beta = alloc_mat(max_length, N);
	float** gamma = alloc_mat(max_length, N);
	float*** digamma = alloc_mat3(max_length, N, N);
	float* coeffs = alloc_vec(max_length);

	float avgTrainLogProb = -1e11f, avgValidLogProb = -1e11f;
	float old_valid_score = -1e11f, old_train_score = -1e11f;

	for (unsigned int epoch = 0; epoch < iterations; epoch++) {
		do {
			accum.reset();
			unsigned int* obs;
			unsigned int length;
			avgTrainLogProb = 0.0f;
			iter.nextTrain(&obs, &length);
			while (obs) {
				alphaPass(obs, length, alpha, coeffs);
				betaPass(obs, length, beta, coeffs); // calculate beta

				calcGamma(obs, length, alpha, beta, gamma, digamma); // calculate the gammas and di-gammas
				accumAdjust(obs, length, gamma, digamma, accum); // add our adjustments to the accumulator

				for (unsigned int i = 0; i < length; i++)
					avgTrainLogProb -= log(coeffs[i]);

				iter.nextTrain(&obs, &length);
			}
			avgTrainLogProb /= (float)accum.count + 1e-10f;

			avgValidLogProb = 0.0f;
			unsigned int validation_count = 0;
			iter.nextValid(&obs, &length);
			while (obs) {
				alphaPass(obs, length, alpha, coeffs);
				for (unsigned int i = 0; i < length; i++)
					avgValidLogProb -= log(coeffs[i]);

				validation_count++;
				iter.nextValid(&obs, &length);
			}
			std::cout << validation_count << std::endl;
			avgValidLogProb /= (float)validation_count + 1e-10f;
			if (old_valid_score >= avgValidLogProb) // stop when our vaidation score decreases
				return;

			old_valid_score = avgValidLogProb;
			old_train_score = avgTrainLogProb;

			applyAdjust(accum); // make the adjustments to the weights
			accum.reset();

			std::cout << "fold average validation log probability: " << avgValidLogProb << std::endl;
		
		} while (iter.nextFold());
		std::cout << avgValidLogProb << std::endl;
	}
	std::cout << avgValidLogProb << std::endl;

	delete_array(alpha, max_length, N); // allocate enough space for the largest observation sequence
	delete_array(beta, max_length, N);
	delete_array(gamma, max_length, N);
	delete_array3(digamma, max_length, N, N);
	delete[] coeffs;
}

void HMM::NFoldTrainingManager::multi_thread_fold(unsigned int nfold, unsigned int _N, unsigned int _M, HMM* _hmm, unsigned int max_sequence_length) {
	std::vector<unsigned int**> fold_regions;
	std::vector<unsigned int*> fold_region_lengths;
	std::vector<unsigned int> fold_region_sizes;
	N = _N;
	M = _M;
	hmm = _hmm;
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

	for (unsigned int i = 0; i < fold_regions.size(); i++) {
		accumulators[i].initialize(N, M);
		workers.emplace_back(TrainingWorker(
			fold_regions[i],
			fold_region_lengths[i],
			fold_region_sizes[i],
			max_sequence_length,
			N,
			M,
			hmm,
			&accumulators[i]
			));
	}
}

HMM::NFoldTrainingManager::~NFoldTrainingManager() {
	delete[] accumulators;
}

void HMM::NFoldTrainingManager::train_fold(unsigned int fold_index, AdjustmentAccumulator* master) {
	std::vector<std::thread> threads;
	for (unsigned int i = 0; i < workers.size(); i++) {
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

	// combine all acumulations in the master accumulator
	master->reset();
	for (unsigned int i = 0; i < workers.size(); i++) {
		AdjustmentAccumulator* cur = &accumulators[i];
		
		if (i != fold_index) {
			master->count += cur->count;
			master->accumLogProb -= cur->accumLogProb;
			for (unsigned int i = 0; i < N; i++) {
				master->pi_accum[i] += cur->pi_accum[i];
				for (unsigned int j = 0; j < N; j++) {
					master->A_digamma_accum[i][j] += cur->A_digamma_accum[i][j];
					master->A_gamma_accum[i][j] += cur->A_gamma_accum[i][j];
				}

				for (unsigned int j = 0; j < M; j++) {
					master->B_gamma_accum[j][i] += cur->B_gamma_accum[j][i];
					master->B_obs_accum[j][i] += cur->B_obs_accum[j][i];
				}
			}
		}
		else {
			master->accumLogProb_v -= cur->accumLogProb_v;
			master->accumLogProb_v /= (float)cur->count + 1e-6f;
		}
	}

	master->accumLogProb /= (float)master->count + 1e-6f;
}

HMM::TrainingWorker::TrainingWorker(
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
	case_count = case_count;
	sequence_count = _sequence_count;
	N = _N;
	M = _M;
	hmm = _hmm;
	accumulator = _accumulator;

	alpha = alloc_mat(sequence_count, N); // allocate enough space for the largest observation sequence
	beta = alloc_mat(sequence_count, N);
	gamma = alloc_mat(sequence_count, N);
	digamma = alloc_mat3(sequence_count, N, N);
	coeffs = alloc_vec(sequence_count);
}

HMM::TrainingWorker::~TrainingWorker() {
	delete_array(alpha, sequence_count, N); // allocate enough space for the largest observation sequence
	delete_array(beta, sequence_count, N);
	delete_array(gamma, sequence_count, N);
	delete_array3(digamma, sequence_count, N, N);
	delete[] coeffs;
}

void HMM::TrainingWorker::train_work() {
	for (unsigned int i = 0; i < case_count; i++) {
		unsigned int* obs = case_data[i];
		unsigned int length = case_lengths[i];
		hmm->alphaPass(obs, length, alpha, coeffs);
		hmm->betaPass(obs, length, beta, coeffs); // calculate beta

		hmm->calcGamma(obs, length, alpha, beta, gamma, digamma); // calculate the gammas and di-gammas
		hmm->accumAdjust(obs, length, gamma, digamma, *accumulator); // add our adjustments to the accumulator

		for (unsigned int i = 0; i < length; i++)
			accumulator->accumLogProb -= log(coeffs[i]);
	}
}

void HMM::TrainingWorker::valid_work() {
	for (unsigned int i = 0; i < case_count; i++) {
		unsigned int* obs = case_data[i];
		unsigned int length = case_lengths[i];
		hmm->alphaPass(obs, length, alpha, coeffs);

		for (unsigned int i = 0; i < length; i++)
			accumulator->accumLogProb_v -= log(coeffs[i]);
	}
}