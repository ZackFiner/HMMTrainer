#include "HmmUtil.h"
#include "MatUtil.h"
#include "HMM.h"
#include <fstream>
#include <iostream>

void pickleHmm(HMM* hmm, std::string fpath) {
	unsigned int N, M, index;
	N = hmm->N;
	M = hmm->M;
	unsigned long words = 2 + N + N * N + N * M;
	float* buffer = new float[words];
	// simple pickle format:
	// 2 32 bit integers, M and N
	// next N*4 bytes are pi floats
	// next N*N*4 bytes are A floats
	// next M*N*4 bytes are B floats
	unsigned int* modifier = (unsigned int*)buffer;
	
	modifier[0] = (unsigned int)M;
	modifier[1] = (unsigned int)N;
	float* dataregion = &buffer[2];
	index = 0;
	for (unsigned int i = 0; i < N; i++)
		dataregion[index++] = hmm->Pi[i];
	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < N; j++)
			dataregion[index++] = hmm->A[i*N + j];
	for (unsigned int i = 0; i < M; i++)
		for (unsigned int j = 0; j < N; j++)
			dataregion[index++] = hmm->B[i*N + j];
	try {
		std::fstream file = std::fstream(fpath, std::ios::out | std::ios::binary);
		file.write((char*)&buffer[0], words << 2);
		file.close();
	}
	catch (const std::exception& e) {
		std::cerr << "Error: could not pickle hmm " << hmm << " to file " << fpath << std::endl;
	}
	delete[] buffer;
}

void initializeHmm(HMM* hmm, std::string fpath) {
	unsigned int M, N;

	std::ifstream file(fpath, std::ios::binary);
	file.read((char*)&M, sizeof(unsigned int));
	file.read((char*)&N, sizeof(unsigned int));

	if (hmm->M != M || hmm->N != N) {
		file.close();
		return;
	}
	float f_buff;
	for (unsigned int i = 0; i < N; i++) {
		file.read((char*)&f_buff, sizeof(float));
		hmm->Pi[i] = f_buff;
	}
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			file.read((char*)&f_buff, sizeof(float));
			hmm->A[i*N + j] = f_buff;
		}
	}
	
	transposeEmplace(hmm->A, N, N, hmm->A_T);
	for (unsigned int i = 0; i < M; i++) {
		for (unsigned int j = 0; j < N; j++) {
			file.read((char*)&f_buff, sizeof(float));
			hmm->B[i*N + j] = f_buff;
		}
	}
	file.close();

}

HMM loadHmm(std::string fpath) {
	unsigned int M, N;

	std::ifstream file(fpath, std::ios::binary);
	file.read((char*)&M, sizeof(unsigned int));
	file.read((char*)&N, sizeof(unsigned int));
	HMM r_hmm(N, M);
	float f_buff;
	for (unsigned int i = 0; i < N; i++) {
		file.read((char*)&f_buff, sizeof(float));
		r_hmm.Pi[i] = f_buff;
	}
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {
			file.read((char*)&f_buff, sizeof(float));
			r_hmm.A[i * N + j] = f_buff;
		}
	}

	transposeEmplace(r_hmm.A, N, N, r_hmm.A_T);
	for (unsigned int i = 0; i < M; i++) {
		for (unsigned int j = 0; j < N; j++) {
			file.read((char*)&f_buff, sizeof(float));
			r_hmm.B[i * N + j] = f_buff;
		}
	}
	file.close();

	return r_hmm;
}