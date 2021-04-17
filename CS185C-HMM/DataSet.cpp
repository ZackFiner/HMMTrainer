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