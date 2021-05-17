#pragma once
#include "DataSet.h"

inline void getSortedColumnVec(float* B, unsigned int N, unsigned int M, float* vec);
void calcWordEmbeddings(
	unsigned int** data,
	unsigned int* length,
	unsigned int size,
	unsigned int N,
	unsigned int M,
	float* results,
	unsigned int max_length,
	const DataMapper& map
);
void generateEmbeddings(const HMMDataSet& dataset, const DataMapper& map, unsigned int N, unsigned int M, float* results);
