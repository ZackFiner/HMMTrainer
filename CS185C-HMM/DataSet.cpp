#include "DataSet.h"


namespace std_fs = std::filesystem;

DataMapper::DataMapper():
	size(0)
{}

DataMapper::DataMapper(const std::unordered_map<std::string, unsigned int>& mapper):
	raw_mapper(mapper)
{
	size = mapper.size();
}

DataMapper::DataMapper(const DataMapper& o):
	raw_mapper(o.raw_mapper),
	size(o.size)
{}

unsigned int DataMapper::getVal(const std::string& v) const
{
	auto loc = this->raw_mapper.find(v);
	return loc != raw_mapper.end() ? loc->second : size;
}

NewLineSeperatedLoader::NewLineSeperatedLoader(std::string _fpath):
	fpath(_fpath),
	file_iterator(_fpath),
	end()
{}

std::vector<std::string> NewLineSeperatedLoader::nextRecord() {
	// read all lines from the file and return it
	auto& cur_file = *(this->file_iterator);
	std::vector<std::string> r_val;
	if (cur_file.is_regular_file()) {

		std::ifstream file(cur_file.path());
		std::string cur;
		
		while(std::getline(file, cur))
			r_val.push_back(cur);
		
		file.close(); // close the file stream after any errors occur
	
	}
	this->file_iterator++;
	return r_val;
}

bool NewLineSeperatedLoader::hasNext() {
	// determine whether we are done reading files
	return this->file_iterator != end;
}

HMMDataSet::HMMDataSet(): 
	data(nullptr), 
	size(0),
	max_length(0)
{
}

HMMDataSet::HMMDataSet(DataLoader* loader, const DataMapper& mapper) :
	symbol_map(mapper),// use deep copy constructor to initialize
	data(nullptr),
	size(0)
{
	bufferData(loader);
}

HMMDataSet::~HMMDataSet() {
	for (unsigned int i = 0; i < size; i++) {
		delete[] data[i];
	}
	delete[] data;
}

void HMMDataSet::bufferData(DataLoader* loader) {
	std::vector<std::vector<std::string>> cache;
	while (loader->hasNext()) {
		cache.push_back(loader->nextRecord());
	}
	size = cache.size();
	data = new unsigned int* [size];
	lengths = new unsigned int[size];
	max_length = 0;
	for (unsigned int i = 0; i < size; i++) {
		std::vector<std::string>& record = cache[i];
		
		unsigned int record_size = record.size();
		lengths[i] = record_size;
		if (record_size > max_length)
			max_length = record_size;

		data[i] = new unsigned int[record_size];
		for (unsigned int j = 0; j < record_size; j++) {
			data[i][j] = symbol_map.getVal(record[i]);
		}
	}
}

int HMMDataSet::getSize() const {
	return size;
}

unsigned int HMMDataSet::getMaxLength() const {
	return max_length;
}


NFoldIterator HMMDataSet::getIter(unsigned int nfolds) const {
	return NFoldIterator(*this, nfolds);
}

NFoldIterator::NFoldIterator(const HMMDataSet& _loader, unsigned int _nfolds): loader(_loader) {
	nfolds = _nfolds;

	int l_size = loader.getSize();

	fold_size = l_size / nfolds;
	fold_index = 0;

	fold_start = loader.data;
	fold_end = std::min(loader.data + fold_size, end);
	cur_validate_ptr = loader.data;
	cur_validate_length_ptr = loader.lengths;

	
	cur_train_ptr = loader.data + fold_size;
	cur_train_length_ptr = loader.lengths + fold_size;
	end = loader.data + l_size;

}

void NFoldIterator::nextTrain(unsigned int** obs, unsigned int* length) {
	if (cur_train_ptr < end) {
		*obs = *cur_train_ptr;
		*length = *cur_train_length_ptr;

		cur_train_ptr++;
		cur_train_length_ptr++;

		int addr = cur_train_ptr == fold_start ? fold_size : 0;
		cur_train_ptr += addr; // skip our fold region if we need to
		cur_train_length_ptr += addr;
	}
	else {
		*obs = nullptr;
	}
}

void NFoldIterator::nextValid(unsigned int** obs, unsigned int* length) {
	if (cur_validate_ptr < fold_end) {
		*obs = *cur_validate_ptr;
		*length = *cur_validate_length_ptr;
		
		cur_validate_ptr++;
		cur_validate_length_ptr++;
	}
	else {
		*obs = nullptr;
	}
}

bool NFoldIterator::nextFold() {
	fold_index++;
	if (fold_index > nfolds) {
		fold_index = 0;

		int l_size = loader.getSize();

		fold_start = loader.data;
		fold_end = loader.data + fold_size;

		cur_validate_ptr = fold_start;
		cur_validate_length_ptr = loader.lengths;

		cur_train_ptr = fold_end;
		cur_train_length_ptr = loader.lengths + fold_size;

		return false;
	}
	else {
		fold_start += fold_size;
		fold_end = std::min(fold_end + fold_size, end);

		cur_validate_ptr += fold_size;
		cur_validate_length_ptr += fold_size;

		cur_train_ptr = loader.data;
		cur_train_length_ptr = loader.lengths;

		return true;
	}
}