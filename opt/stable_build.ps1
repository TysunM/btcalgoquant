$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
# Targeting Ada Lovelace (8.9) for Blackwell compatibility
$env:TORCH_CUDA_ARCH_LIST = "8.9"
$env:USE_CUDA = "1"
$env:USE_CUDNN = "1"
$env:CMAKE_GENERATOR = "Ninja"
$env:PYTORCH_CUDA_VERSION = "12.4"
$env:MAX_JOBS = "8" 

cd C:\btc_quant\opt
.\venv\Scripts\activate
python -m pip install numpy pyyaml mkl mkl-include setuptools cmake packaging typing_extensions jinja2 sympy networkx

cd pytorch
python setup.py develop
