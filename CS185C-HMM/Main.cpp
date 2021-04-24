// Zackary C. Finer - CS 185C SJSU

#include <iostream>
#include "HMM.h"
#include "DataSet.h"
#include "MatUtil.h"
#include "ProbInit.h"
#include "HmmUtil.h"

int main() {

	NewLineSeperatedLoader loader = NewLineSeperatedLoader("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zeroaccess");
	DataMapper winwebsec_mapper = generateDataMapFromStats("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zeroaccess_stats.csv", 20);
	HMMDataSet winwebsec_dataset = HMMDataSet(&loader, winwebsec_mapper);
	winwebsec_dataset.printExample(0);

	
	DefaultProbInit initializer(0.8f);
	HMM hmm = HMM(5, winwebsec_dataset.getSymbolCount(), &initializer);
	hmm.trainModel(winwebsec_dataset, 5, 10);
	pickle_hmm(&hmm, "K:\\GitHub\\CS185C-HMM\\hmm_5_20_za.hmm");
	
	
	/*
	HMM hmm(5, 20);
	initialize_hmm(&hmm, "K:\\GitHub\\CS185C-HMM\\hmm_5_20_za.hmm");
	hmm.setDataMapper(winwebsec_mapper);
	hmm.print_mats();
	NewLineSeperatedLoader loader2 = NewLineSeperatedLoader("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zbot");
	DataMapper zbot_mapper = generateDataMapFromStats("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zbot_stats.csv", 5000);
	HMMDataSet zbot_dataset = HMMDataSet(&loader2, zbot_mapper);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -1.82f);
	*/
	
	/*
	* // Pickling tests
	HMM hmm(20, 300);
	hmm.print_mats();
	pickle_hmm(&hmm, "K:\\GitHub\\CS185C-HMM\\pickledexample.hmm");
	HMM hmm2(20, 300);
	initialize_hmm(&hmm2, "K:\\GitHub\\CS185C-HMM\\pickledexample.hmm");
	hmm2.print_mats();
	*/
	return 0;
}