#pragma once

#ifdef COMPILING_DLL
#define HMMLIB_API __declspec(dllexport)
#else
#define HMMLIB_API __declspec(dllimport)
#endif

extern "C" HMMLIB_API int twotimes(int x);

extern "C" HMMLIB_API void getRoc(float* arr, unsigned int size, char* dset1, size_t len1, char* dset2, size_t len2, char* hmm_fp, size_t len3);