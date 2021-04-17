#include "DataSet.h"


namespace std_fs = std::filesystem;

template<class T>
DataMapper<T>::DataMapper():
	size(0)
{}

template<class T>
DataMapper<T>::DataMapper(const std::unordered_map<T, unsigned int>& mapper):
	raw_mapper(mapper)
{
	size = mapper.size();
}

template<class T>
DataMapper<T>::DataMapper(const DataMapper<T>& o):
	raw_mapper(o.raw_mapper),
	size(o.size)
{}

template<class T>
unsigned int DataMapper<T>::getVal(const T& v)
{
	auto loc = this->raw_mapper.find(v);
	return loc != raw_mapper.end() ? loc->second : size;
}

template<>
NewLineSeperatedLoader<std::string>::NewLineSeperatedLoader(std::string _fpath):
	fpath(_fpath),
	file_iterator(_fpath),
	end()
{}

template<>
std::vector<std::string> NewLineSeperatedLoader<std::string>::nextRecord() {
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

template<>
bool NewLineSeperatedLoader<std::string>::hasNext() {
	// determine whether we are done reading files
	return this->file_iterator != end;
}


template<class T>
HMMDataSet<T>::HMMDataSet(): 
	data(nullptr), 
	size(0) 
{
}

template<class T>
HMMDataSet<T>::HMMDataSet(const DataLoader<T>& loader, const DataMapper<T>& mapper) :
	symbol_map(mapper),// use deep copy constructor to initialize
	data(nullptr),
	size(0)
{
	bufferData(loader);
}

template<class T>
HMMDataSet<T>::~HMMDataSet() {
	for (unsigned int i = 0; i < size; i++) {
		delete[] data[i];
	}
	delete[] data;
}

template<class T>
void HMMDataSet<T>::bufferData(const DataLoader<T>& loader) {
	std::vector<std::vector<T>> cache;
	while (loader.hasNext()) {
		cache.push_back(loader.nextRecord());
	}
	size = cache.size();
	data = new unsigned int* [size];
	lengths = new unsigned int[size];
	for (unsigned int i = 0; i < size; i++) {
		std::vector<T>& record = cache[i];
		
		unsigned int record_size = record.size();
		lengths[i] = record_size;

		data[i] = new unsigned int[record_size];
		for (unsigned int j = 0; j < record_size; j++) {
			data[i][j] = symbol_map.getVal(record[i]);
		}
	}
}

template<class T>
int HMMDataSet<T>::getSize() {
	return size;
}

template<class T>
NFoldIterator<T> HMMDataSet<T>::getIter(unsigned int nfolds) {
	return NFoldIterator<T>(*this, nfolds);
}

template<class T>
NFoldIterator<T>::NFoldIterator(const HMMDataSet<T>& _loader, unsigned int _nfolds) {
	loader = _loader;
	nfolds = _nfolds;

	int l_size = loader.getSize();

	fold_size = l_size / nfolds;
	fold_index = 0;

	fold_start = loader.data;
	fold_end = min(loader.data + fold_size, end);
	cur_validate_ptr = loader.data;
	cur_validate_length_ptr = loader.lengths;

	
	cur_train_ptr = loader.data + fold_size;
	cur_train_length_ptr = loader.lengths + fold_size;
	end = loader.data + l_size;
	lengths_end = loader.lengths + l_size;

}

template<class T>
void NFoldIterator<T>::nextTrain(unsigned int** obs, unsigned int* length) {
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

template<class T>
void NFoldIterator<T>::nextValid(unsigned int** obs, unsigned int* length) {
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

template<class T>
bool NFoldIterator<T>::nextFold() {
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
		fold_end = min(fold_end + fold_size, end);

		cur_validate_ptr += fold_size;
		cur_validate_length_ptr += fold_size;

		cur_train_ptr = loader.data;
		cur_train_length_ptr = loader.lengths;

		return true;
	}
}