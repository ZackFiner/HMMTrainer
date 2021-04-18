// Zackary C. Finer - CS 185C SJSU

#include <iostream>
#include "HMM.h"
#include "DataSet.h"
#include "MatUtil.h"
#include "ProbInit.h"

int main() {
	NewLineSeperatedLoader loader = NewLineSeperatedLoader("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zbot");
	DataMapper winwebsec_mapper = generateDataMapFromStats("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zbot_stats.csv", 1000);
	HMMDataSet winwebsec_dataset = HMMDataSet(&loader, winwebsec_mapper);
	winwebsec_dataset.printExample(0);
	HMM hmm = HMM(10, winwebsec_dataset.getSymbolCount());
	hmm.trainModel(winwebsec_dataset, 10, 10);

	return 0;
}