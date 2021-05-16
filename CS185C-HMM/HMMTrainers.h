#pragma once
#include "DataSet.h"

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
void generateEmbeddings(const HMMDataSet& positives, const HMMDataSet& negatives, const DataMapper& map, unsigned int N, unsigned int M, float* results);

inline void getSortedColumnVec(float* B, unsigned int N, unsigned int M, float* vec);