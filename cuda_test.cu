#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device = 0;

    cudaError_t err = cudaGetDevice(&device);

    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << '\n';
        return 1;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "Compute Capability: "
              << prop.major << "." << prop.minor << '\n';
    std::cout << "SM Count: " << prop.multiProcessorCount << '\n';
    std::cout << "Global Memory: "
              << prop.totalGlobalMem / (1024 * 1024) << " MB\n";

    return 0;
}
