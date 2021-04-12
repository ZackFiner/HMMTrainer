#pragma once

class HMM
{
public:
	HMM();
	HMM(unsigned int N, unsigned int M);
	~HMM();

private:
	void alphaPass(unsigned int* obs, unsigned int size);
	void betaPass(unsigned int* obs, unsigned int size);
	void calcGamma();

	float **alpha, **beta, **coeffs, **gamma, **digamma, **A, **B;
	float *Pi;
	unsigned int N, M;

};

