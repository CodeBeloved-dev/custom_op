from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from glob import glob
import tomli as tomllib 

egg_dir = os.path.join(os.getcwd(), "build", "egg-info")
# 提前创建目录（如果不存在）
os.makedirs(egg_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错

# 读取 pyproject.toml 中的版本号
with open('pyproject.toml', 'rb') as f:
    pyproject_data = tomllib.load(f)
version = pyproject_data['project']['version']  # 从 pyproject.toml 中获取版本

    # 批量获取所有源文件
def find_source_files(root_dir):
    source_extensions = ['*.cpp', '*.cu']
    source_files = []
    
    for ext in source_extensions:
        pattern = os.path.join(root_dir, '**', ext)
        source_files.extend(glob(pattern, recursive=True))
    
    return source_files

# 指定源文件根目录（根据你的项目结构修改）
SOURCE_ROOT = 'src'  # 假设所有源文件都在src目录下

# 获取所有源文件路径
source_files = find_source_files(SOURCE_ROOT)

# 打印找到的文件（可选，用于验证）
print("找到的源文件:")
for file in source_files:
    print(f"  - {file}")

setup(
    name='my_cuda_add',
    version=version,  # 使用 pyproject.toml 中的版本
    ext_modules=[
        CUDAExtension(
            name='my_cuda_add',
            sources=source_files,  # 使用批量获取的文件列表
            include_dirs=[os.path.abspath('src')]  # 头文件目录（根据实际情况修改）
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
    