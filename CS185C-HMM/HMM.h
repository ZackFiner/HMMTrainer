#pragma once
#include <vector>
#include <string>
#include "DataSet.h"

class ProbInit;


class HMM
{
public:
	HMM();
	HMM(float* _A, float* _B, float* _Pi, unsigned int _N, unsigned int _M);
	HMM(unsigned int _N, unsigned int _M, ProbInit* initializer = nullptr);
	HMM(const HMM&);
	HMM& operator=(const HMM&);
	~HMM();
	int* getIdealStateSequence(unsigned int* obs, unsigned int size);
	void trainModel(const HMMDataSet& dataset, unsigned int iterations = 10, unsigned int n_folds = 10, unsigned int fold_index = 0, bool early_stop = true, unsigned int patience = 1);
	void print_mats() const;
	void reset(ProbInit* initializer);
	friend void pickle_hmm(HMM* hmm, std::string fpath);
	friend void initialize_hmm(HMM* hmm, std::string fpath);
	friend HMM load_hmm(std::string fpath);
	void setDataMapper(const DataMapper& o);
	void testClassifier(const HMMDataSet& positives, const HMMDataSet& negatives, float thresh) const;
	void generateROC(const HMMDataSet& positives, const HMMDataSet& negatives, float* dest, unsigned int eval_size = 0) const;
	unsigned int getM();
	unsigned int getN();

private:
	struct AdjustmentAccumulator {
		float* A_gamma_accum;
		float* A_digamma_accum;
		float* B_gamma_accum;
		float* B_obs_accum;
		float* pi_accum;
		unsigned int N, M, count;
		float accumLogProb;
		float accumLogProb_v;
		void initialize(unsigned int N, unsigned int M);
		void reset();
		bool initialized = false;
		~AdjustmentAccumulator();

	};
	struct TrainingWorker {
		void initialize(
			unsigned int** _case_data,
			unsigned int* _case_lengths,
			unsigned int _case_count,
			unsigned int _sequence_count,
			unsigned int _N,
			unsigned int _M,
			HMM* _hmm,
			AdjustmentAccumulator* _accumulator
			);

		unsigned int** case_data;
		unsigned int* case_lengths;
		unsigned int case_count;

		float *coeffs=nullptr, *alpha=nullptr, *beta=nullptr, *gamma=nullptr, * digamma=nullptr;
		unsigned int sequence_count, N, M;

		HMM* hmm;
		AdjustmentAccumulator* accumulator;
		void train_work();
		void valid_work();
		~TrainingWorker();
	};

	struct NFoldTrainingManager {
		AdjustmentAccumulator* accumulators;
		unsigned int** data;
		unsigned int* lengths;
		unsigned int case_count;
		unsigned int N;
		unsigned int M;
		unsigned int fold_count;
		HMM* hmm;
		TrainingWorker* workers;
		void multi_thread_fold(unsigned int nfold, unsigned int _N, unsigned int _M, HMM* _hmm, unsigned int max_sequence_length);
		void train_fold(unsigned int fold_index, AdjustmentAccumulator* master);
		void score_fold(unsigned int fold_index, AdjustmentAccumulator* master);
		~NFoldTrainingManager();
	};



	int getStateAtT(float* gamma, unsigned int size, unsigned int t);
	void alphaPass(unsigned int* obs, unsigned int size, float* alpha, float* coeffs) const;
	void betaPass(unsigned int* obs, unsigned int size, float* beta, float* coeffs) const;
	void calcGamma(unsigned int* obs, unsigned int size, float* alpha, float *beta, float* gamma, float* digamma);
	void calcGamma(unsigned int* obs, unsigned int size, unsigned int t, float* alpha, float* beta, float* gamma, float* digamma);

	void applyAdjust(unsigned int* obs, unsigned int size, float* gamma, float* digamma);
	void applyAdjust(const AdjustmentAccumulator& accum);
	void accumAdjust(unsigned int* obs, unsigned int size, float* gamma, float* digamma, AdjustmentAccumulator& accum);
	void accumAdjust(unsigned int* obs, unsigned int size, unsigned int t, float* gamma, float* digamma, AdjustmentAccumulator& accum);
	float calcSeqProb(float* alpha, unsigned int size);


	DataMapper native_symbolmap;
	float *A, *B, *A_T;
	float *Pi;
	unsigned int N, M;

};
