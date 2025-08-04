#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void my_add_kernel(const float* a, const float* b, float* c, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

void my_add_cuda(const float* a, const float* b, float* c, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    my_add_kernel<<<blocks, threads>>>(a, b, c, size);
}
