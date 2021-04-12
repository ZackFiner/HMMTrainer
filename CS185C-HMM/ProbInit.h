#pragma once
class ProbInit
{
public:
	virtual float** AInit(unsigned int N) = 0;
	virtual float** BInit(unsigned int N, unsigned int M) = 0;
	virtual float* PiIinit(unsigned int N) = 0;
};

class DefaultProbInit : public ProbInit {
public:
	DefaultProbInit(float _variance=0.1);
	float** AInit(unsigned int N);
	float** BInit(unsigned int N, unsigned int M);
	float* PiIinit(unsigned int N);
private:
	void initializeStochasticRow(float* row, unsigned int size);
	float variance;

};