//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <omp.h>
#include <iostream>

//Compile: icpx -fiopenmp -fopenmp-targets=spir64 simple.cpp -o simple

constexpr int N = 16;
int main() {
  	int is_cpu = true;
  	int *data = static_cast<int *>(malloc(N * sizeof(int)));

	// Initialization
	for (int i = 0; i < N; i++){
		data[i] = i;
	}

	std::cout << "Dispositivos detectados: " << omp_get_num_devices() << "\n";

	// Add the target directive here, including the map clause.
	#pragma omp target map(from : is_cpu) map(tofrom : data [0:N])
	{
		is_cpu = omp_is_initial_device();
		#pragma omp parallel for
		for (int i = 0; i < N; i++) {
			data[i] *= 2;
		}
	}

	// Print Output
	std::cout << "Running on " << (is_cpu ? "CPU" : "GPU") << "\n";
	for (int i = 0; i < N; i++) {
		std::cout << data[i] << "\n";
	}

	free(data);
	return 0;
}
