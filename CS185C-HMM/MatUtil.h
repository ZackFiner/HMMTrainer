#pragma once


float** transpose(float** mat, unsigned int N, unsigned int M) {
	// NxM -> MxN
	float** t_mat = new float* [M];
	for (unsigned int i = 0; i < M; i++)
		t_mat[i] = new float[N];

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < M; j++)
			t_mat[j][i] = mat[i][j];
	}
	
	return t_mat;
}

void print(float** mat, unsigned int N, unsigned int M) {

}