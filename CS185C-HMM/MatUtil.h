#pragma once
#include <iostream>
#include <string>
float* transpose(float* mat, unsigned int N, unsigned int M);
void transposeEmplace(float* mat, unsigned int N, unsigned int M, float* dest);
void deleteArray(float* mat, unsigned int N, unsigned int M);
void deleteArray3(float* arr, unsigned int N, unsigned int M, unsigned int R);
float* allocMat(unsigned int N, unsigned int M);
float getMaxAbs(float* mat, unsigned int N, unsigned int M);
float* allocMat(float* init, unsigned int N, unsigned int M);
float* allocVec(unsigned int N);
float* allocVec(float* init, unsigned int N);
float* allocMat3(unsigned int N, unsigned int M, unsigned int R);
void printMatrix(float* mat, unsigned int N, unsigned int M, bool transpose=false, unsigned int prec = 2);
void printVector(float* vec, unsigned int N, unsigned int prec = 2);
class DataMapper;

DataMapper generateDataMapFromStats(const std::string& fpath, unsigned int symbolcount = 0);