#pragma once
#include <iostream>
#include <string>
float** transpose(float** mat, unsigned int N, unsigned int M);
void delete_array(float** mat, unsigned int N, unsigned int M);
void delete_array3(float*** arr, unsigned int N, unsigned int M, unsigned int R);
float** alloc_mat(unsigned int N, unsigned int M);
float get_max_abs(float** mat, unsigned int N, unsigned int M);
float** alloc_mat(float** init, unsigned int N, unsigned int M);
float* alloc_vec(unsigned int N);
float* alloc_vec(float* init, unsigned int N);
float*** alloc_mat3(unsigned int N, unsigned int M, unsigned int R);
void print_matrix(float** mat, unsigned int N, unsigned int M, bool transpose=false);

template <class T>
class DataMapper;

DataMapper<std::string> generateDataMapFromStats(const std::string& fpath, unsigned int symbolcount = 0);