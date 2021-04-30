// Zackary C. Finer - CS 185C SJSU

#include <iostream>
#include <sstream>
#include "HMM.h"
#include "DataSet.h"
#include "MatUtil.h"
#include "ProbInit.h"
#include "HmmUtil.h"
#include "Master.h"

int main() {

#define N 10
#define M 20

	NewLineSeperatedLoader loader = NewLineSeperatedLoader("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zbot");
	DataMapper winwebsec_mapper = generateDataMapFromStats("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zbot_stats.csv", M);
	HMMDataSet winwebsec_dataset = HMMDataSet(&loader, winwebsec_mapper);
	winwebsec_dataset.printExample(0);

	DefaultProbInit initializer(0.99f);
	HMM hmm = HMM(N, winwebsec_dataset.getSymbolCount(), &initializer);
	for (unsigned int i = 0; i < 10; i++) {
		std::cout << i << "'th fold" << std::endl;
		hmm.trainModel(winwebsec_dataset, 100, 10, i, true, 1);
		std::ostringstream s;
		s.str("");
		s << "K:\\GitHub\\CS185C-HMM\\hmms\\zbot\\N_series\\hmm_" << N << "_"<< M << "_zb_fold"<< i << ".hmm";
		pickle_hmm(&hmm, s.str());
		hmm.reset(&initializer);
	}
	
	/*
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