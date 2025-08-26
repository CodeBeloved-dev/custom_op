from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os
from glob import glob
import tomli as tomllib 
import torch

egg_dir = os.path.join(os.getcwd(), "build", "egg-info")
# 提前创建目录（如果不存在）
os.makedirs(egg_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错

# 读取 pyproject.toml 中的版本号
with open('pyproject.toml', 'rb') as f:
    pyproject_data = tomllib.load(f)
version = pyproject_data['project']['version']  # 从 pyproject.toml 中获取版本

    # 批量获取所有源文件
def find_source_files(root_dir, with_cuda):
    source_extensions = ['*.cpp']
    if with_cuda: source_extensions.append('*.cu')
    source_files = []
    
    for ext in source_extensions:
        pattern = os.path.join(root_dir, '**', ext)
        source_files.extend(glob(pattern, recursive=True))
    
    return source_files

if torch.cuda.is_available():
    WITH_CUDA=True

# 指定源文件根目录（根据你的项目结构修改）
SOURCE_ROOT = 'src'  # 假设所有源文件都在src目录下
source_files = find_source_files(SOURCE_ROOT, WITH_CUDA)
    
extra_compile_args = {"cxx": ["-O2"]}
if WITH_CUDA:
    extra_compile_args["cxx"].append("-DWITH_CUDA")
    ext_modules = [
            CUDAExtension(
                name='my_cuda_add',
                sources=source_files,  # 使用批量获取的文件列表
                include_dirs=[os.path.abspath('src')],  # 头文件目录（根据实际情况修改）
                extra_link_args=["-lOpenCL"],
                extra_compile_args=extra_compile_args,
            ),
        ]
else:
    ext_modules = [
        CppExtension(
            name="my_cuda_add",
            sources=source_files,
            include_dirs=[os.path.abspath('src')],  # 头文件目录（根据实际情况修改）
            extra_link_args=["-lOpenCL"],
            extra_compile_args=extra_compile_args,
        )
    ]
    
setup(
    name='my_cuda_add',
    version=version,  # 使用 pyproject.toml 中的版本
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
    