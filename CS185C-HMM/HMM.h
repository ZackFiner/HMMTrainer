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
	void alphaPass(unsigned int* obs, unsigned int size, float** alpha); // TODO: add scaling
	void betaPass(unsigned int* obs, unsigned int size, float** beta); // TODO: add scaling
	void calcGamma(unsigned int* obs, unsigned int size, float** alpha, float **beta, float** gamma);
	void calcDigamma(unsigned int* obs, unsigned int size, float** alpha, float** beta, float*** digamma);

	void applyAdjust(unsigned int* obs, unsigned int size, float** gamma, float*** digamma);

	void calcCoeffs(float** alpha, unsigned int size);
	float calcSeqProb(float** alpha, unsigned int size);

	float **A, **B;
	float *Pi;
	unsigned int N, M;

};

