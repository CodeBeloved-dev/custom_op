#include <torch/extension.h>
#include <vector>

#include "cuda_ops.h"
// C++接口，调用CUDA kernel
torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same size");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");

    auto c = torch::empty_like(a);

    int64_t size = a.numel();
    my_add_cuda(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), size);

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_add", &my_add, "My CUDA add");
}
