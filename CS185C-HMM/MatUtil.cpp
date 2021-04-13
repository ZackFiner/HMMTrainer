#include <iostream>
#include <string>

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

void delete_array(float** mat, unsigned int N, unsigned int M) {
	for (unsigned int i = 0; i < N; i++) {
		delete[] mat[i];
	}

	delete[] mat;
}

float** alloc_mat(unsigned int N, unsigned int M) {
	float** r_val = new float* [N];
	for (unsigned int i = 0; i < N; i++)
		r_val[i] = new float[M];

	return r_val;
}

float get_max_abs(float** mat, unsigned int N, unsigned int M) {
	float abs_max = 0.0f;
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < M; j++) {
			float v = mat[i][j];
			if (abs(v) > abs(abs_max))
				abs_max = v;
		}
	}
	return abs_max;
}


float** alloc_mat(float** init, unsigned int N, unsigned int M) {
	float** r_val = new float* [N];
	for (unsigned int i = 0; i < N; i++)
		r_val[i] = new float[M];

	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < M; j++)
			r_val[i][j] = init[i][j];

	return r_val;
}


float* alloc_vec(unsigned int N) {
	return new float[N];
}

float* alloc_vec(float* init, unsigned int N) {
	float* r_val = new float[N];
	for (unsigned int i = 0; i < N; i++)
		r_val[i] = init[i];
	return r_val;
}

void print_matrix(float** mat, unsigned int N, unsigned int M) {

	float abs_max = get_max_abs(mat, N, M);
	int max_padding = std::to_string(abs_max).length() + 5;
	for (unsigned int i = 0; i < N; i++) {
		std::cout << "\n|";
		for (unsigned int j = 0; j < M; j++) {
			float v = mat[i][j];
			auto num = std::to_string(v);
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