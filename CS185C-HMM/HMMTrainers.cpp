#include "HMMTrainers.h"
#include "HMM.h"
#include "MatUtil.h"
#include <thread>

#define NUM_THREADS 10
#define RANDOM_RESTARTS 10
#define MAX_TRAIN_ITERATIONS 100

//#define EARLY_STOPPING
#define EARLY_STOP_IMPROVEMENT 1e-7
#define EARLY_STOP_TOLERANCE 3

inline void getSortedColumnVec(float* B, unsigned int N, unsigned int M, float* vec) {
	std::vector<int> new_indx;
	for (unsigned int i = 0; i < N; i++)
		new_indx.push_back(i);

	std::sort(new_indx.begin(), new_indx.end(), [B, M, N](int a, int b) {
		//lexical comparison of two columns
		for (unsigned int i = 0; i < M; i++) {
			float v1 = B[i * N + a]; // ith observation probability for hidden state a
			float v2 = B[i * N + b]; // ith observation probability for hidden state b
			if (v1 != v2) // if the observation probabilities are not equal
				return v1 > v2; // return the larger of the two
			// otherwise continue searching until we find one which is different
		}
		return true; // if everything is equal, then just return a is greater than b
	});

	// new_indx should now be sorted according to the column values of B
	// we can now read the columns in the order that new_indx specifies to get the sorted column vector

	unsigned int vec_idx = 0;
	for (unsigned int i = 0; i < N; i++)
	{
		unsigned int col_idx = new_indx[i];
		for (unsigned int j = 0; j < M; j++) {
			vec[vec_idx] = B[j * N + col_idx];
			vec_idx++;
		}
	}
}

inline float getLogProb(float* coeffs, unsigned int size) {
	float log_prob = 0.0f;
	for (unsigned int i = 0; i < size; i++)
		log_prob += log(coeffs[i]);
	return -log_prob;
}

void calcWordEmbeddings(
	unsigned int** data, 
	unsigned int* length, 
	unsigned int size, 
	unsigned int N, 
	unsigned int M, 
	float* results, 
	unsigned int max_length, 
	const DataMapper& map
) {
	float* alpha = allocMat(max_length, N);
	float* beta = allocMat(max_length, N);
	float* coeffs = allocVec(max_length);
	float* digamma = allocMat(N, N);
	float* gamma = allocVec(N);
	
	HMM::AdjustmentAccumulator acc;
	acc.initialize(N, M);
	float* current_result = results;
	for (unsigned int i = 0; i < size; i++) {
		unsigned int* seq = data[i];
		unsigned int seq_l = length[i];


		/*
		* 	// Dr. Stamp's paper outlines how the number of random restarts needed depends on the length of the sequence:
			// 30k+ symbols = 10 restarts
			// 10-30k symbols = 30 restarts
			// 5k-10k symbols = 100 restarts
			// fewer than 500 = 500 restarts
		* Chandak, A., Lee, W., & Stamp, M. (2021). A Comparison of Word2Vec, HMM2Vec, and PCA2Vec for Malware Classification. CoRR, abs/2103.05763.
		*/
		std::vector<HMM> hmms;
		for (unsigned int j = 0; j < RANDOM_RESTARTS; j++) {
			HMM hmm(N, M); // new HMM
			hmm.setDataMapper(map); // make sure our symbol map is consistent with the dataset
			
			hmm.alphaPass(seq, seq_l, alpha, coeffs); // initial alpha pass and coeffs for comparison

			float old_log_prob = getLogProb(coeffs, seq_l);

			unsigned int hiccups = 0;
			
			for (unsigned int k = 0; k < MAX_TRAIN_ITERATIONS; k++) { // train the HMM
				acc.reset();
				for (unsigned int l = 0; l < seq_l; l++) {
					hmm.betaPass(seq, seq_l, beta, coeffs);
					hmm.calcGamma(seq, seq_l, l, alpha, beta, gamma, digamma);
					hmm.accumAdjust(seq, seq_l, l, gamma, digamma, acc);
				}
				hmm.applyAdjust(acc);
				hmm.alphaPass(seq, seq_l, alpha, coeffs); // calc alpha and coeffs for next round
				
				// below is the code for early stopping
#ifdef EARLY_STOPPING
				float new_log_prob = getLogProb(coeffs, seq_l);
				float delta_log_prob = new_log_prob - old_log_prob;
				old_log_prob = new_log_prob;

				if (delta_log_prob > 0.0f && delta_log_prob < EARLY_STOP_IMPROVEMENT) { // early stop if we're moving too slow
					if (hiccups > EARLY_STOP_TOLERANCE)
						break;
					hiccups++;
				}
				else {
					hiccups = 0;
				}
#endif
			}
			hmms.push_back(hmm);
		}
		hmms[0].alphaPass(seq, seq_l, alpha, coeffs);
		float best_log_prob = getLogProb(coeffs, seq_l);
		unsigned int best_hmm_index = 0;

		for (unsigned int j = 1; j < RANDOM_RESTARTS; j++) {
			hmms[j].alphaPass(seq, seq_l, alpha, coeffs);
			float log_prob = getLogProb(coeffs, seq_l);
			if (log_prob > best_log_prob)
				best_hmm_index = j;
		}

		getSortedColumnVec(hmms[best_hmm_index].B, N, M, current_result);
		current_result = current_result + N * M;
		
	}

	delArray(alpha, max_length, N);
	delArray(beta, max_length, N);
	delArray(digamma, N, N);
	delete[] coeffs;
	delete[] gamma;
}


void generateEmbeddings(const HMMDataSet& positives, const HMMDataSet& negatives, const DataMapper& map, unsigned int N, unsigned int M, float* results) {
	/*
	* For each example (sequence):
	*	1. train several models for the sequence and select the one with the highest probabilitiy (random restarts)
	*	2. inspect the resulting B matrix of the best model, sort it's columns by some criteria (probability of a popular instruction has been used)
	*	3. stack the columns of the B matrix to get a resulting embedding vector, and place it in the corresponding entry in the results array
	* 
	* Parallelism:
	*	In this function, we need to train several models for each example sequence, so the parallelism we used for training the models before
	*	isn't as applicable. Instead, we can train a lot of these models in parallel. For now, we will create several worker threads, each
	*	with their own set of HMMs to train, and just let them work sequentially. This will require us to partition our datasets.
	*/


	auto remap_pos = positives.getRemapped(map);
	auto remap_neg = negatives.getRemapped(map);
	auto pos_partitions = remap_pos.getPartitions(NUM_THREADS);
	auto neg_partitions = remap_neg.getPartitions(NUM_THREADS);

	std::vector<std::thread> threads;
	float* cur_results = results;
	unsigned int max_l = positives.getMaxLength();
	for (unsigned int i = 0; i < NUM_THREADS; i++) {
		unsigned int** part = pos_partitions[i].first;
		unsigned int length = pos_partitions[i].second.first;
		unsigned int* lengths = pos_partitions[i].second.second;
		unsigned int max_l = positives.getMaxLength();
		threads.emplace_back(
			std::thread(calcWordEmbeddings, part, lengths, length, N, M, cur_results, max_l, map)
		);

		cur_results = cur_results + length;
	}

	max_l = negatives.getMaxLength();
	for (unsigned int i = 0; i < NUM_THREADS; i++) {
		unsigned int** part = neg_partitions[i].first;
		unsigned int length = neg_partitions[i].second.first;
		unsigned int* lengths = neg_partitions[i].second.second;
		threads.emplace_back(
			std::thread(calcWordEmbeddings, part, lengths, length, N, M, cur_results, max_l, map)
		);

		cur_results = cur_results + length;
	}


	for (auto& thread : threads)
		thread.join();
	// we're done...
}