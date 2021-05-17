#include "Master.h"
#include "HMM.h"
#include "HmmUtil.h"
#include "MatUtil.h"
#include "DataSet.h"
#include "HMMTrainers.h"
#include <sstream>

int twotimes(int x) {
	return x << 1;
}

void getRoc(
	float* arr, 
	unsigned int size, 
	char* dset1, 
	char* dset2, 
	char* hmm_fp, 
	unsigned int eval_size
) {
	std::ostringstream factory;
	factory << std::string(dset1) << "_stats.csv"; 
	std::string mapper_file1 = factory.str();
	factory.str("");
	factory << std::string(dset2) << "_stats.csv"; 
	std::string mapper_file2 = factory.str();

	std::string dataset_file1 = std::string(dset1);
	std::string dataset_file2 = std::string(dset2);
	std::string hmm_file = std::string(hmm_fp);
	HMM tester = loadHmm(hmm_file);

	DataMapper mapper1 = generateDataMapFromStats(mapper_file1, tester.getM());
	tester.setDataMapper(mapper1);
	NewLineSeperatedLoader loader1 = NewLineSeperatedLoader(dataset_file1);

	DataMapper mapper2 = generateDataMapFromStats(mapper_file2, 0xffffffffU);
	NewLineSeperatedLoader loader2 = NewLineSeperatedLoader(dataset_file2);

	HMMDataSet positive_dataset(&loader1, mapper1);
	HMMDataSet negative_dataset(&loader2, mapper2);
	unsigned int total_size = (positive_dataset.getSize() + negative_dataset.getSize()) * 2;
	
	if (size < total_size)
		return;
	tester.generateROC(positive_dataset, negative_dataset, arr, eval_size);
	//DataMapper set1_mapper = generateDataMapFromStats(mapper_file1, )

}

void scoreModelFolds(float* arr, 
	unsigned int size, 
	char* dset1, 
	char* dset2, 
	char* hmm_fp, 
	char* test_series, 
	char* abrv,
	unsigned int M, 
	unsigned int N,  
	unsigned int eval_size
) {
	std::ostringstream factory;
	factory << std::string(dset1) << "_stats.csv";
	std::string mapper_file1 = factory.str();
	factory.str("");
	factory << std::string(dset2) << "_stats.csv";
	std::string mapper_file2 = factory.str();
	factory.str("");

	std::string dataset_file1 = std::string(dset1);
	std::string dataset_file2 = std::string(dset2);
	std::string hmm_file = std::string(hmm_fp);
	std::string abrv_s = std::string(abrv);
	std::string test_series_s = std::string(test_series);

	HMM tester = HMM(N, M);
	DataMapper mapper1 = generateDataMapFromStats(mapper_file1, M);
	tester.setDataMapper(mapper1);
	NewLineSeperatedLoader loader1 = NewLineSeperatedLoader(dataset_file1);

	DataMapper mapper2 = generateDataMapFromStats(mapper_file2, 0xffffffffU);
	NewLineSeperatedLoader loader2 = NewLineSeperatedLoader(dataset_file2);

	HMMDataSet positive_dataset(&loader1, mapper1);
	HMMDataSet negative_dataset(&loader2, mapper2);
	unsigned int total_size = (positive_dataset.getSize() + negative_dataset.getSize()) * 2 * 10;
	unsigned int fold_size = (positive_dataset.getSize() + negative_dataset.getSize()) * 2;

	if (total_size > size)
		return;

	float* fold_head = arr;

	for (unsigned int i = 0; i < 10; i++) {
		factory << hmm_file << "\\" << test_series_s << "\\hmm_" << N << "_" << M << "_" << abrv_s << "_fold" << i << ".hmm";
		std::string hmm_file_path = factory.str();
		factory.str("");
		initializeHmm(&tester, hmm_file_path);
		tester.generateROC(positive_dataset, negative_dataset, fold_head, eval_size);
		fold_head += fold_size;
	}
}

void evaluateModelFolds(float* arr,
	unsigned int size,
	char* dset1,
	char* dset2,
	char* hmm_fp,
	char* test_series,
	char* abrv,
	unsigned int M,
	unsigned int N,
	unsigned int eval_size
) {
	std::ostringstream factory;
	factory << std::string(dset1) << "_stats.csv";
	std::string mapper_file1 = factory.str();
	factory.str("");
	factory << std::string(dset2) << "_stats.csv";
	std::string mapper_file2 = factory.str();
	factory.str("");

	std::string dataset_file1 = std::string(dset1);
	std::string dataset_file2 = std::string(dset2);
	std::string hmm_file = std::string(hmm_fp);
	std::string abrv_s = std::string(abrv);
	std::string test_series_s = std::string(test_series);

	HMM tester = HMM(N, M);
	DataMapper mapper1 = generateDataMapFromStats(mapper_file1, M);
	tester.setDataMapper(mapper1);
	NewLineSeperatedLoader loader1 = NewLineSeperatedLoader(dataset_file1);

	DataMapper mapper2 = generateDataMapFromStats(mapper_file2, 0xffffffffU);
	NewLineSeperatedLoader loader2 = NewLineSeperatedLoader(dataset_file2);

	HMMDataSet positive_dataset(&loader1, mapper1);
	HMMDataSet negative_dataset(&loader2, mapper2);
	unsigned int total_size = (positive_dataset.getSize() + negative_dataset.getSize()) * 10;
	unsigned int fold_size = (positive_dataset.getSize() + negative_dataset.getSize());

	if (total_size > size)
		return;

	float* fold_head = arr;

	for (unsigned int i = 0; i < 10; i++) {
		factory << hmm_file << "\\" << test_series_s << "\\hmm_" << N << "_" << M << "_" << abrv_s << "_fold" << i << ".hmm";
		std::string hmm_file_path = factory.str();
		factory.str("");
		initializeHmm(&tester, hmm_file_path);
		tester.evaluateModel(positive_dataset, negative_dataset, fold_head, eval_size);
		fold_head += fold_size;
	}
}

void generatHMMEmbedding(
	float* arr,
	unsigned int size,
	char* dset,
	char* dmap,
	unsigned int M,
	unsigned int N
) {
	std::string dataset_fpath(dset);
	std::string symbolmap_fpath(dmap);

	NewLineSeperatedLoader loader(dataset_fpath);
	DataMapper map = generateDataMapFromStats(symbolmap_fpath, M);
	HMMDataSet dataset(&loader, map);


	if (size < M * N * dataset.getSize())
		return;

	generateEmbeddings(dataset, map, N, M, arr);

}