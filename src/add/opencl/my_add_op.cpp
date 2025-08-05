#include <torch/extension.h>
#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <vector>

#define CHECK_CL(err) if (err != CL_SUCCESS) { \
    std::cerr << "OpenCL error " << err << " at line " << __LINE__ << std::endl; \
    exit(1); \
}

std::string load_kernel(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open OpenCL kernel file.");
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
std::string getCurrentFileDir() {
    std::string path = __FILE__;
    size_t lastSlash = path.find_last_of("/\\");
    if (lastSlash == std::string::npos) {
        return "."; // 当前目录
    }
    return path.substr(0, lastSlash);
}
torch::Tensor add_opencl(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape.");
    TORCH_CHECK(a.device().is_cpu(), "Only CPU tensors are supported.");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Only float32 supported.");

    int64_t n = a.numel();
    auto out = torch::empty_like(a);

    // Load OpenCL kernel
    std::string kernel_code = load_kernel(getCurrentFileDir() + "/add.cl");
    const char* kernel_src = kernel_code.c_str();

    // Setup OpenCL
    cl_int err;
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;

    err = clGetPlatformIDs(1, &platform_id, nullptr); CHECK_CL(err);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, nullptr); CHECK_CL(err);
    context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err); CHECK_CL(err);
    queue = clCreateCommandQueue(context, device_id, 0, &err); CHECK_CL(err);

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_src, nullptr, &err); CHECK_CL(err);
    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // 打印构建错误信息
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        std::cerr << "Build log:\n" << build_log.data() << std::endl;
        CHECK_CL(err);
    }

    cl_kernel kernel = clCreateKernel(program, "add", &err); CHECK_CL(err);

    // 创建 buffer
    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, a.data_ptr(), &err); CHECK_CL(err);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, b.data_ptr(), &err); CHECK_CL(err);
    cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &err); CHECK_CL(err);

    // 设置 kernel 参数
    CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf));
    CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf));
    CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_buf));
    CHECK_CL(clSetKernelArg(kernel, 3, sizeof(int64_t), &n));

    size_t global_work_size = n;
    CHECK_CL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr));
    CHECK_CL(clFinish(queue));

    CHECK_CL(clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, sizeof(float) * n, out.data_ptr(), 0, nullptr, nullptr));

    // Cleanup
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(out_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return out;
}

