#include "Master.h"
#include "HMM.h"
#include "HmmUtil.h"
#include "MatUtil.h"
#include "DataSet.h"
#include <sstream>

int twotimes(int x) {
	return x << 1;
}

void getRoc(float* arr, unsigned int size, char* dset1, size_t len1, char* dset2, size_t len2, char* hmm_fp, size_t len3) {
	std::ostringstream factory;
	factory << std::string(dset1) << "_stats.csv"; 
	std::string mapper_file1 = factory.str();
	factory.str("");
	factory << std::string(dset2) << "_stats.csv"; 
	std::string mapper_file2 = factory.str();

	std::string dataset_file1 = std::string(dset1);
	std::string dataset_file2 = std::string(dset2);
	std::string hmm_file = std::string(hmm_fp);
	HMM tester = load_hmm(hmm_file);

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
	tester.generateROC(positive_dataset, negative_dataset, arr);
	//DataMapper set1_mapper = generateDataMapFromStats(mapper_file1, )

}