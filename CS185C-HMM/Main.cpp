#include <iostream>
#include "HMM.h"
#include "MatUtil.h"

int main() {
	std::cout << "Assignment 6 - HMM alpha_pass and beta_pass functions (+gamma)" << std::endl;

	// Initialize A
	float a1[] = { 0.7f, 0.3f }; // H [H, C]
	float a2[] = { 0.4f, 0.6f }; // C [H, C]
	float* A[] = { a1, a2 };

	// Initialize B
	float b1[] = { 0.1f, 0.4f, 0.5f }; // H [s, m, l]
	float b2[] = { 0.7f, 0.2f, 0.1f }; // C [s, m, l]
	float* B[] = { b1, b2 };

	// Initialize Pi
	float* Pi = new float[2];
	Pi[0] = 0.6f; Pi[1] = 0.4f;

	// Create an O = {Small, Medium, Small, Large}
	unsigned int* O = new unsigned int[4];
	O[0] = 0; O[1] = 1; O[2] = 0; O[3] = 2;

	std::cout << "\nA: " << std::endl;
	print_matrix(A, 2, 2);

	std::cout << "\nB: " << std::endl;
	print_matrix(B, 2, 3);


	std::cout << "\nObservation Sequence: small, medium, small, large" << std::endl;
	HMM test = HMM(
		alloc_mat(A, 2, 2), 
		alloc_mat(B, 2, 3), 
		Pi, 2, 3);

	std::cout << "\nGamma Debug Array: " << std::endl;
	int* S = test.getIdealStateSequence(O, 4);
	
	std::cout << "\n\nIdeal State Sequence: ";
	for (unsigned int i = 0; i < 4; i++) {
		std::cout << (S[i] ? "Cold " : "Hot ");
	}
	
	std::cout << std::endl;

	while (true);
	delete[] O;
	delete[] S;
	
	return 0;
}