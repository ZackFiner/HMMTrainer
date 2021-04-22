#pragma once

#ifdef COMPILING_DLL
#define HMMLIB_API __declspec(dllexport)
#else
#define HMMLIB_API __declspec(dllimport)
#endif

extern "C" HMMLIB_API int twotimes(int x);