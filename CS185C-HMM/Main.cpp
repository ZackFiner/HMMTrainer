#include <iostream>
#include "MatUtil.h"
int main() {
	std::cout << "Test" << std::endl;

	float test0[] = { 1.0f, 0.0f };
	float test1[] = { 0.0f, -1000.0f };
	float* test[] = { test0, test1 };
	print_matrix(test, 2, 2);
	while (true);
	return 0;
}