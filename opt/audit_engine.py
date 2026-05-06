import ctypes
import os

# Define the critical paths
torch_lib = r"C:\Users\Tysun\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
intel_mkl = r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin\intel64"
intel_cmp = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\intel64"
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64"

# Force Windows to look in these directories
for path in [torch_lib, intel_mkl, intel_cmp, cuda_bin]:
    if os.path.exists(path):
        os.add_dll_directory(path)
        print(f"[PATH ADDED] {path}")
    else:
        print(f"[PATH MISSING] {path}")

print("\n--- Testing Core Dependencies ---")
# The "Conducting" and "Math" Chain
deps = [
    ("vcruntime140_1.dll", "C++ Runtime"),
    ("libiomp5md.dll", "Intel OpenMP"),
    ("mkl_core.2.dll", "Intel MKL"),
    ("cudart64_13.dll", "CUDA Runtime"),
    ("cublas64_13.dll", "CUBLAS"),
    ("cudnn64_9.dll", "cuDNN") # Check if yours is 9 or 10
]

for dll, label in deps:
    try:
        ctypes.CDLL(dll)
        print(f"[PASS] {label} ({dll}) loaded.")
    except Exception as e:
        print(f"[FAIL] {label} ({dll}) error: {e}")

print("\n--- Final Link Test ---")
try:
    aoti = os.path.join(torch_lib, "aoti_custom_ops.dll")
    ctypes.CDLL(aoti)
    print("[SUCCESS] aoti_custom_ops.dll is fully linked.")
except Exception as e:
    print(f"[CRITICAL FAIL] {e}")