#include <torch/extension.h>
#ifdef WITH_CUDA
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
#endif
torch::Tensor add_opencl(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_CUDA
    m.def("add_cuda", &add_cuda, "My CUDA add");
#endif
    m.def("add_opencl", &add_opencl, "My OpenCL Add");
}