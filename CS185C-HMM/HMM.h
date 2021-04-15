#pragma once

class ProbInit;
class HMM
{
public:
	HMM();
	HMM(float** _A, float** _B, float* _Pi, unsigned int _N, unsigned int _M);
	HMM(unsigned int _N, unsigned int _M, ProbInit* initializer = nullptr);
	~HMM();
	int* getIdealStateSequence(unsigned int* obs, unsigned int size);
private:
	int getStateAtT(float** gamma, unsigned int size, unsigned int t);
	void alphaPass(unsigned int* obs, unsigned int size, float** alpha, float* coeffs);
	void betaPass(unsigned int* obs, unsigned int size, float** beta, float* coeffs);
	void calcGamma(unsigned int* obs, unsigned int size, float** alpha, float **beta, float** gamma); // merge this with digamma calculation
	void calcDigamma(unsigned int* obs, unsigned int size, float** alpha, float** beta, float*** digamma); // add scaling

	void applyAdjust(unsigned int* obs, unsigned int size, float** gamma, float*** digamma);

	float calcSeqProb(float** alpha, unsigned int size);

	float **A, **B;
	float *Pi;
	unsigned int N, M;

};

