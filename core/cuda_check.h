#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

struct CudaError : std::runtime_error {
    cudaError_t code;

    CudaError(cudaError_t c, const char* file, int line)
        : std::runtime_error(std::string(cudaGetErrorName(c)) + ": " + cudaGetErrorString(c) +
                             "at" + file + ":" + std::to_string(line))
        , code(c) {}
};

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t cuda_try_err = (call);                                                         \
        if (cuda_try_err != cudaSuccess) {                                                         \
                                                                                                   \
            throw CudaError(cuda_try_err, __FILE__, __LINE__);                                     \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

