#pragma once
#include <vector>
class ProbInit;
class HMM
{
public:
	HMM();
	HMM(float** _A, float** _B, float* _Pi, unsigned int _N, unsigned int _M);
	HMM(unsigned int _N, unsigned int _M, ProbInit* initializer = nullptr);
	~HMM();
	int* getIdealStateSequence(unsigned int* obs, unsigned int size);
	void trainModel(const std::vector<std::vector<unsigned int>>& dataset, unsigned int iterations = 10, unsigned int n_folds = 10);
private:
	int getStateAtT(float** gamma, unsigned int size, unsigned int t);
	void alphaPass(unsigned int* obs, unsigned int size, float** alpha, float* coeffs);
	void betaPass(unsigned int* obs, unsigned int size, float** beta, float* coeffs);
	void calcGamma(unsigned int* obs, unsigned int size, float** alpha, float **beta, float** gamma, float*** digamma);

	void applyAdjust(unsigned int* obs, unsigned int size, float** gamma, float*** digamma);

	float calcSeqProb(float** alpha, unsigned int size);

	float **A, **B;
	float *Pi;
	unsigned int N, M;

};

