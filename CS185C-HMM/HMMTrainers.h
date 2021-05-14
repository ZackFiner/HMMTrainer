#pragma once
#include "DataSet.h"

void generateEmbeddings(const HMMDataSet& positives, const HMMDataSet& negatives, unsigned int N, unsigned int M, float* results);