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

template <class T>
class HMMDataSet
{
public:
	HMMDataSet();
	HMMDataSet(const DataLoader<T>& loader, const DataMapper<T>& mapper);
	~HMMDataSet();
private:
	void bufferData(const DataLoader<T>& loader);
	DataMapper<T> symbol_map;
	unsigned int** data; // raw 2d array of byte sequences
	unsigned int* lengths; // array of record lengths
	int size;
};

