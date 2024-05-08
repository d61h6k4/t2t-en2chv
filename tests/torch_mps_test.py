# PYTORCH_DEBUG_MPS_ALLOCATOR=1 python mps_leak_test.py
import torch

a = torch.randn(3, 1024, 1024, device="mps")

print("works fine:")
for i in range(3):
    a = a.to(device="cpu").to(dtype=torch.float32)
    a = a.to(device="mps").to(dtype=torch.float16)
    torch.mps.empty_cache()

print("memory leak:")
for i in range(3):
    a = a.to(device="cpu", dtype=torch.float32)
    a = a.to(device="mps", dtype=torch.float16)
    torch.mps.empty_cache()

print("exit")
