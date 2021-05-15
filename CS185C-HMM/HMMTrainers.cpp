#include "HMMTrainers.h"
#include "HMM.h"
#include "MatUtil.h"
#include <thread>

#define NUM_THREADS 10
#define RANDOM_RESTARTS 10
#define MAX_TRAIN_ITERATIONS 100
#define EARLY_STOP_IMPROVEMENT 1e-7

void calcWordEmbeddings(
	unsigned int** data, 
	unsigned int* length, 
	unsigned int size, 
	unsigned int N, 
	unsigned int M, 
	float* results, 
	unsigned int max_length, 
	const DataMapper& map) {
	float* alpha = allocMat(max_length, N);
	float* beta = allocMat(max_length, N);
	float* coeffs = allocVec(max_length);
	float* digamma = allocMat(N, N);
	float* gamma = allocVec(N);
	
	HMM::AdjustmentAccumulator acc;
	acc.initialize(N, M);

	for (unsigned int i = 0; i < size; i++) {
		unsigned int* seq = data[i];
		unsigned int seq_l = length[i];
		std::vector<HMM> hmms;
		for (unsigned int j = 0; j < RANDOM_RESTARTS; j++) {
			HMM hmm(N, M); // new HMM
			hmm.setDataMapper(map); // make sure our symbol map is consistent with the dataset
			
			for (unsigned int k = 0; k < MAX_TRAIN_ITERATIONS; k++) { // train the HMM
				acc.reset();
				for (unsigned int l = 0; l < seq_l; l++) {
					hmm.alphaPass(seq, seq_l, alpha, coeffs);
					hmm.betaPass(seq, seq_l, beta, coeffs);
					hmm.calcGamma(seq, seq_l, l, alpha, beta, gamma, digamma);

					hmm.accumAdjust(seq, seq_l, l, gamma, digamma, acc);
				}

				hmm.applyAdjust(acc);

				// TODO: add early stop code
			}
			hmms.push_back(hmm);
		}

		// TODO: select the best hmm model using seq probability as the metric
		// TODO: sort the columns of the best B matrix
		// TODO: stack the B matrix columns, and save the resulting vector in results[i]
		
	}



	deleteArray(alpha, max_length, N);
	deleteArray(beta, max_length, N);
	deleteArray(digamma, N, N);
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
			std::thread(calcWordEmbeddings, part, lengths, length, cur_results, max_l, map)
		);

		cur_results = cur_results + length;
	}

	max_l = negatives.getMaxLength();
	for (unsigned int i = 0; i < NUM_THREADS; i++) {
		unsigned int** part = neg_partitions[i].first;
		unsigned int length = neg_partitions[i].second.first;
		unsigned int* lengths = neg_partitions[i].second.second;
		threads.emplace_back(
			std::thread(calcWordEmbeddings, part, lengths, length, cur_results, max_l, map)
		);

		cur_results = cur_results + length;
	}


	for (auto& thread : threads)
		thread.join();
	// we're done...
}