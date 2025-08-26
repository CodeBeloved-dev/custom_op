import torch
import my_cuda_add
import pytest
import sys

# ------------------------------
# 2. 测试代码（pytest风格）
# ------------------------------
def test_cuda_add_basic():
    """测试基本加法功能"""
    a = torch.tensor([1.0, 2.0, 3.0], device='cpu')
    b = torch.tensor([4.0, 5.0, 6.0], device='cpu')
    result = my_cuda_add.add_opencl(a, b)
    assert torch.allclose(result, torch.tensor([5.0, 7.0, 9.0], device='cpu'))

def test_cuda_add_large_tensors():
    if torch.cuda.is_available():
        """测试大型张量加法"""
        a = torch.randn(80000, 10000, device='cuda')
        b = torch.randn(80000, 10000, device='cuda')
        result = my_cuda_add.add_cuda(a, b)
        assert torch.allclose(result, a + b)  # 与PyTorch内置加法对比
    
    
if __name__ == "__main__":
    # 当运行 python my_cuda_add.py 时，自动调用pytest运行当前文件中的测试
    pytest.main([__file__, "-v"])