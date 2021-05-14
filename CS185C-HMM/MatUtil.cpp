#include "MatUtil.h"
#include "DataSet.h"
#include <fstream>
#include <unordered_map>
#include <sstream>

float* transpose(float* mat, unsigned int N, unsigned int M) {
	// NxM -> MxN
	float* t_mat = new float [M*N];

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < M; j++)
			t_mat[j*N + i] = mat[i*M + j];
	}

	return t_mat;
}

void transposeEmplace(float* mat, unsigned int N, unsigned int M, float* dest) {
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < M; j++)
			dest[j * N + i] = mat[i * M + j];
	}
}

void deleteArray(float* mat, unsigned int N, unsigned int M) {
	if (!mat)
		return;

	delete[] mat;
}

void deleteArray3(float* arr, unsigned int N, unsigned int M, unsigned int R) {
	if (!arr)
		return;

	delete[] arr;
}

float* allocMat(unsigned int N, unsigned int M) {
	float* r_val = new float [N*M];

	return r_val;
}

float* allocMat3(unsigned int N, unsigned int M, unsigned int R) {
	float* r_val = new float [N*M*R];

	return r_val;
}

float getMaxAbs(float* mat, unsigned int N, unsigned int M) {
	float abs_max = 0.0f;
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < M; j++) {
			float v = mat[i*M + j];
			if (abs(v) > abs(abs_max))
				abs_max = v;
		}
	}
	return abs_max;
}

float getMaxAbs(float* vec, unsigned int N) {
	float abs_max = 0.0f;
	for (unsigned int i = 0; i < N; i++) {
		float v = vec[i];
		if (abs(v) > abs(abs_max))
			abs_max = v;
	}
	return abs_max;
}

float* allocMat(float* init, unsigned int N, unsigned int M) {
	float* r_val = new float [N*M];

	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < M; j++)
			r_val[i*M + j] = init[i*M + j];

	return r_val;
}


float* allocVec(unsigned int N) {
	return new float[N];
}

float* allocVec(float* init, unsigned int N) {
	float* r_val = new float[N];
	for (unsigned int i = 0; i < N; i++)
		r_val[i] = init[i];
	return r_val;
}

void printMatrix(float* mat, unsigned int N, unsigned int M, bool transpose, unsigned int prec) {

	float abs_max = getMaxAbs(mat, N, M);
	std::ostringstream converter;
	converter.precision(prec);
	converter << std::fixed << abs_max;
	int max_padding = converter.str().length() + 3;

	unsigned int width = transpose ? M : N;
	unsigned int height = transpose ? N : M;


	for (unsigned int i = 0; i < width; i++) {
		std::cout << "\n|";
		for (unsigned int j = 0; j < height; j++) {
			float v = transpose ? mat[j*width + i] : mat[i*height + j];
			converter.str("");
			converter.clear();
			converter << std::fixed << v;
			auto num = converter.str();
			int l = num.length();
			int padding = max_padding - l;
			int pre_padding = padding >> 1; // padding /2
			int post_padding = (padding >> 1) + (1 & padding);// padding /2 + 1 if padding is odd

			pre_padding = v >= 0.0 ? pre_padding : pre_padding - 1;
			post_padding = v >= 0.0 ? post_padding : post_padding + 1;
			std::cout << std::string(pre_padding, ' ') << num << std::string(post_padding, ' ');
		}
		std::cout << "|";
	}
	std::cout << std::endl;

}

void printVector(float* vec, unsigned int N, unsigned int prec) {

	float abs_max = getMaxAbs(vec, N);
	std::ostringstream converter;
	converter.precision(prec);
	converter << std::fixed << abs_max;
	int max_padding = converter.str().length() + 3;

	std::cout << "\n|";
	for (unsigned int i = 0; i < N; i++) {

		float v = vec[i];			
		converter.str("");
		converter.clear();
		converter << std::fixed << v;
		auto num = converter.str();
		int l = num.length();
		int padding = max_padding - l;
		int pre_padding = padding >> 1; // padding /2
		int post_padding = (padding >> 1) + (1 & padding);// padding /2 + 1 if padding is odd

		pre_padding = v >= 0.0 ? pre_padding : pre_padding - 1;
		post_padding = v >= 0.0 ? post_padding : post_padding + 1;
		std::cout << std::string(pre_padding, ' ') << num << std::string(post_padding, ' ');
		
	}
	std::cout << "|";
	std::cout << std::endl;

}

DataMapper generateDataMapFromStats(const std::string& fpath, unsigned int symbolcount) {
	std::ifstream file(fpath);
	std::unordered_map<std::string, unsigned int> mapper;
	std::string cur;
	unsigned int i = 0;
	while (std::getline(file, cur) && i < symbolcount) {
		std::string opcode = cur.substr(0,cur.find(','));
		mapper[opcode] = i;
		i++;
	}

	DataMapper r_val = DataMapper(mapper);
	return r_val;
}