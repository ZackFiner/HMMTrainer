#pragma once

float** transpose(float** mat, unsigned int N, unsigned int M);
void delete_array(float** mat, unsigned int N, unsigned int M);
float** alloc_mat(unsigned int N, unsigned int M);
float get_max_abs(float** mat, unsigned int N, unsigned int M);
float** alloc_mat(float** init, unsigned int N, unsigned int M);
float* alloc_vec(unsigned int N);
float* alloc_vec(float* init, unsigned int N);
void print_matrix(float** mat, unsigned int N, unsigned int M);