#include <torch/extension.h>
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor add_opencl(torch::Tensor a, torch::Tensor b);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "My CUDA add");
    m.def("add_opencl", &add_opencl, "My OpenCL Add");
}