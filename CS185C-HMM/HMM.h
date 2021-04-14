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
	int getStateAtT(unsigned int t);
	void alphaPass(unsigned int* obs, unsigned int size);
	void betaPass(unsigned int* obs, unsigned int size);
	void calcGamma(unsigned int* obs, unsigned int size);
	void calcSeqProb();

	float seqProb;
	float **alpha=nullptr, **beta=nullptr, **coeffs=nullptr, **gamma=nullptr, ***digamma=nullptr, **A, **B;
	float *Pi;
	unsigned int N, M;
	unsigned int T;

};

