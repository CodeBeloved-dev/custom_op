#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

__global__ void add(const float* a, const float* b, float* c, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}


torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same size");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");

    auto c = torch::empty_like(a);

    int64_t size = a.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), size);

    return c;
}
