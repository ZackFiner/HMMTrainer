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

#define N 6
#define M 14
	float arr[11330];
	char* dset1 = (char*)"K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zeroaccess";
	size_t len1 = strlen(dset1);
	char* dset2 = (char*)"K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\winwebsec";
	size_t len2 = strlen(dset2);
	char* hmm_fp = (char*)"K:\\GitHub\\CS185C-HMM\\hmms\\zeroaccess\\N_series\\hmm_3_10_za_fold0.hmm";
	size_t len3 = strlen(hmm_fp);

	getRoc(arr, 11330, dset1, len1, dset2, len2, hmm_fp, len3);

	/*NewLineSeperatedLoader loader = NewLineSeperatedLoader("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zeroaccess");
	DataMapper winwebsec_mapper = generateDataMapFromStats("K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\zeroaccess_stats.csv", M);
	HMMDataSet winwebsec_dataset = HMMDataSet(&loader, winwebsec_mapper);
	winwebsec_dataset.printExample(0);

	DefaultProbInit initializer(0.99f);
	HMM hmm = HMM(N, winwebsec_dataset.getSymbolCount(), &initializer);
	for (unsigned int i = 0; i < 10; i++) {
		std::cout << i << "'th fold" << std::endl;
		hmm.trainModel(winwebsec_dataset, 100, 10, i, true, 1);
		std::ostringstream s;
		s.str("");
		s << "K:\\GitHub\\CS185C-HMM\\hmms\\zeroaccess\\M_series\\hmm_" << N << "_"<< M << "_za_fold"<< i << ".hmm";
		pickle_hmm(&hmm, s.str());
		hmm.reset(&initializer);
	}
	*/
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