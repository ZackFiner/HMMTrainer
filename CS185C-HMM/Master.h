#pragma once

#ifdef COMPILING_DLL
#define HMMLIB_API __declspec(dllexport)
#else
#define HMMLIB_API __declspec(dllimport)
#endif

extern "C" HMMLIB_API int twotimes(int x);

extern "C" HMMLIB_API void getRoc(
	float* arr, 
	unsigned int size, 
	char* dset1, 
	char* dset2, 
	char* hmm_fp, 
	unsigned int eval_size);

extern "C" HMMLIB_API void scoreModelFolds(
	float* arr,
	unsigned int size,
	char* dset1,
	char* dset2,
	char* hmm_fp,
	char* test_series,
	char* abrv,
	unsigned int M,
	unsigned int N,
	unsigned int eval_size);

extern "C" HMMLIB_API void evaluateModelFolds(
	float* arr,
	unsigned int size,
	char* dset1,
	char* dset2,
	char* hmm_fp,
	char* test_series,
	char* abrv,
	unsigned int M,
	unsigned int N,
	unsigned int eval_size);