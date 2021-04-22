#pragma once
#include <vector>
#include <string>
class HMMDataSet;
class ProbInit;


class HMM
{
public:
	HMM();
	HMM(float* _A, float* _B, float* _Pi, unsigned int _N, unsigned int _M);
	HMM(unsigned int _N, unsigned int _M, ProbInit* initializer = nullptr);
	~HMM();
	int* getIdealStateSequence(unsigned int* obs, unsigned int size);
	void trainModel(const HMMDataSet& dataset, unsigned int iterations = 10, unsigned int n_folds = 10);
	void print_mats() const;
	friend void pickle_hmm(HMM* hmm, std::string fpath);
	friend void initialize_hmm(HMM* hmm, std::string fpath);

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
		~NFoldTrainingManager();
	};



	int getStateAtT(float* gamma, unsigned int size, unsigned int t);
	void alphaPass(unsigned int* obs, unsigned int size, float* alpha, float* coeffs);
	void betaPass(unsigned int* obs, unsigned int size, float* beta, float* coeffs);
	void calcGamma(unsigned int* obs, unsigned int size, float* alpha, float *beta, float* gamma, float* digamma);
	void calcGamma(unsigned int* obs, unsigned int size, unsigned int t, float* alpha, float* beta, float* gamma, float* digamma);

	void applyAdjust(unsigned int* obs, unsigned int size, float* gamma, float* digamma);
	void applyAdjust(const AdjustmentAccumulator& accum);
	void accumAdjust(unsigned int* obs, unsigned int size, float* gamma, float* digamma, AdjustmentAccumulator& accum);
	void accumAdjust(unsigned int* obs, unsigned int size, unsigned int t, float* gamma, float* digamma, AdjustmentAccumulator& accum);
	float calcSeqProb(float* alpha, unsigned int size);


	float *A, *B;
	float *Pi;
	unsigned int N, M;

};
