// Zackary C. Finer - CS 185C SJSU

#include <iostream>
#include "HMM.h"
#include "DataSet.h"
#include "MatUtil.h"
#include "ProbInit.h"
#include "HmmUtil.h"

int main() {

	NewLineSeperatedLoader loader = NewLineSeperatedLoader("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zeroaccess");
	DataMapper winwebsec_mapper = generateDataMapFromStats("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zeroaccess_stats.csv", 30);
	HMMDataSet winwebsec_dataset = HMMDataSet(&loader, winwebsec_mapper);
	winwebsec_dataset.printExample(0);

	/*
	DefaultProbInit initializer(0.99f);
	HMM hmm = HMM(20, winwebsec_dataset.getSymbolCount(), &initializer);
	hmm.trainModel(winwebsec_dataset, 100, 10);
	pickle_hmm(&hmm, "K:\\GitHub\\CS185C-HMM\\hmm_20_30_za.hmm");
	*/
	
	
	HMM hmm(50, 30);
	initialize_hmm(&hmm, "K:\\GitHub\\CS185C-HMM\\hmm_50_30_za.hmm");
	hmm.setDataMapper(winwebsec_mapper);
	hmm.print_mats();
	NewLineSeperatedLoader loader2 = NewLineSeperatedLoader("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zbot");
	DataMapper zbot_mapper = generateDataMapFromStats("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zbot_stats.csv", 5000);
	HMMDataSet zbot_dataset = HMMDataSet(&loader2, zbot_mapper);

	
	
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -1.85f);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -1.90f);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -1.95f);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -2.00f);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -2.05f);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -2.10f);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -2.15f);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -2.20f);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -2.25f);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -2.30f);
	hmm.testClassifier(winwebsec_dataset, zbot_dataset, -2.35f);
	
	
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