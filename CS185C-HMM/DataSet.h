#pragma once
#include <unordered_map>
#include <string>
#include <fstream>
#include <filesystem>

class DataMapper {
public:
	DataMapper();
	DataMapper(const std::unordered_map<std::string, unsigned int>& mapper);
	DataMapper(const DataMapper& o);
	unsigned int getVal(const std::string& v) const;
	unsigned int getSymbolCount() const;
	DataMapper& operator=(const DataMapper& o);
	std::unordered_map<unsigned int, std::string> getReverseMap() const;
private:
	std::unordered_map<std::string, unsigned int> raw_mapper; // this should be 1:1 for now
	unsigned int size;
};


class DataLoader {
public:
	virtual std::vector<std::string> nextRecord() = 0;
	virtual bool hasNext() = 0;
};


class NewLineSeperatedLoader : public DataLoader {
public:
	NewLineSeperatedLoader(std::string _fpath);
	std::vector<std::string> nextRecord();
	bool hasNext();
private:
	std::string fpath;
	std::filesystem::directory_iterator file_iterator;
	std::filesystem::directory_iterator end;
	unsigned int cur = 0;
};

class NFoldIterator;

class HMMDataSet
{
public:
	HMMDataSet();
	HMMDataSet(DataLoader* loader, const DataMapper& mapper);
	HMMDataSet(const HMMDataSet& o);
	~HMMDataSet();
	NFoldIterator getIter(unsigned int nfolds) const;
	int getSize() const;
	unsigned int getMaxLength() const;
	void printExample(unsigned int i) const;
	unsigned int getSymbolCount() const;
	unsigned int** getDataPtr() const;
	unsigned int* getLengthsPtr() const;
	HMMDataSet& operator=(const HMMDataSet& o);
	HMMDataSet getRemapped(const DataMapper& other) const;
private:
	void bufferData(DataLoader* loader);
	DataMapper symbol_map;
	unsigned int** data = nullptr; // raw 2d array of byte sequences
	unsigned int* lengths = nullptr; // array of record lengths
	unsigned int max_length; // i don't want to re-allocate every time i load a new example
	int size;
	friend class NFoldIterator;
};

class NFoldIterator {
public:
	NFoldIterator(const HMMDataSet& _loader, unsigned int _nfolds);
	void nextTrain(unsigned int** obs, unsigned int* length);
	void nextValid(unsigned int** obs, unsigned int* length);
	bool nextFold();
private:
	const HMMDataSet& loader;
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
