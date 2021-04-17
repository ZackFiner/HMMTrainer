#pragma once
#include <vector>
template<class T> class HMMDataSet;
class ProbInit;
class HMM
{
public:
	HMM();
	HMM(float** _A, float** _B, float* _Pi, unsigned int _N, unsigned int _M);
	HMM(unsigned int _N, unsigned int _M, ProbInit* initializer = nullptr);
	~HMM();
	int* getIdealStateSequence(unsigned int* obs, unsigned int size);
	template<class T>
	void trainModel(const HMMDataSet<T>& dataset, unsigned int iterations = 10, unsigned int n_folds = 10);
private:
	struct AdjustmentAccumulator {
		float** A_gamma_accum;
		float** A_digamma_accum;
		float** B_gamma_accum;
		float** B_obs_accum;
		float* pi_accum;
		unsigned int N, M;
		void initialize(unsigned int N, unsigned int M);
		void reset();
		~AdjustmentAccumulator();

	};
	int getStateAtT(float** gamma, unsigned int size, unsigned int t);
	void alphaPass(unsigned int* obs, unsigned int size, float** alpha, float* coeffs);
	void betaPass(unsigned int* obs, unsigned int size, float** beta, float* coeffs);
	void calcGamma(unsigned int* obs, unsigned int size, float** alpha, float **beta, float** gamma, float*** digamma);

	void applyAdjust(unsigned int* obs, unsigned int size, float** gamma, float*** digamma);
	void accumAdjust(unsigned int* obs, unsigned int size, float** gamma, float*** digamma, const AdjustmentAccumulator& accum);
	float calcSeqProb(float** alpha, unsigned int size);


	float **A, **B;
	float *Pi;
	unsigned int N, M;

};

