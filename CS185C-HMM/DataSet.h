#pragma once
#include <unordered_map>
#include <string>
#include <fstream>
#include <filesystem>

template <class T>
class DataMapper {
public:
	DataMapper();
	DataMapper(const std::unordered_map<T, unsigned int>& mapper);
	DataMapper(const DataMapper<T>& o);
	unsigned int getVal(const T& v);
private:
	std::unordered_map<T, unsigned int> raw_mapper; // this should be 1:1 for now
	unsigned int size;
};

template <class T>
class DataLoader {
public:
	virtual std::vector<T> nextRecord() = 0;
	virtual bool hasNext() = 0;
};

template <class T>
class NewLineSeperatedLoader : public DataLoader<T> {
public:
	NewLineSeperatedLoader(std::string _fpath);
	std::vector<T> nextRecord();
	bool hasNext();
private:
	std::string fpath;
	std::filesystem::directory_iterator file_iterator;
	std::filesystem::directory_iterator end;
	unsigned int cur = 0;
};

template<class T> class NFoldIterator;

template <class T>
class HMMDataSet
{
public:
	HMMDataSet();
	HMMDataSet(const DataLoader<T>& loader, const DataMapper<T>& mapper);
	~HMMDataSet();
	NFoldIterator<T> getIter(unsigned int nfolds);
	int getSize();
private:
	void bufferData(const DataLoader<T>& loader);
	DataMapper<T> symbol_map;
	unsigned int** data; // raw 2d array of byte sequences
	unsigned int* lengths; // array of record lengths
	int size;
	friend class NFoldIterator<T>;
};

template <class T>
class NFoldIterator {
public:
	NFoldIterator(const HMMDataSet<T>& _loader, unsigned int _nfolds);
	void nextTrain(unsigned int** obs, unsigned int* length);
	void nextValid(unsigned int** obs, unsigned int* length);
	bool nextFold();
private:
	const DataLoader<T>& loader;
	unsigned int nfolds;
	unsigned int fold_index;
	unsigned int fold_size;

	unsigned int** cur_train_ptr;
	unsigned int** cur_validate_ptr;
	unsigned int* cur_validate_length_ptr;
	unsigned int* cur_train_length_ptr;

	unsigned int** fold_start;
	unsigned int** fold_end;

	unsigned int** end;
};
