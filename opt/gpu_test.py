import torch

def verify_gpu():
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Device Count: {torch.cuda.device_count()}")
    else:
        print("WARNING: CUDA is not available. ML training will fall back to CPU.")
        print("We must resolve this before proceeding to Phase 3.")

if __name__ == "__main__":
    verify_gpu()
