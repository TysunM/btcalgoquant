import polars as pl
import torch
import time

INPUT = r'C:\btc_quant\data\processed\quant_data_clean.parquet'

def load_to_vram():
    print('\n[SYSTEM] Initializing CUDA Handshake...')
    if not torch.cuda.is_available():
        print('[CRITICAL ERROR] CUDA is offline. PyTorch cannot see the RTX 5060.')
        return
    
    device = torch.device('cuda:0')
    print(f'[INFO] GPU Locked: {torch.cuda.get_device_name(0)}')
    
    start = time.time()
    print('[INFO] Reading clean Parquet from NVMe...')
    df = pl.read_parquet(INPUT).select(['Price', 'Open', 'High', 'Low', 'Vol', 'Change'])
    
    print('[INFO] Forging Tensor and streaming directly to VRAM...')
    tensor_data = torch.tensor(df.to_numpy(), dtype=torch.float32, device=device)
    
    end = time.time()
    rows, cols = tensor_data.shape
    vram_mb = tensor_data.element_size() * tensor_data.nelement() / (1024 * 1024)
    
    print(f'\n[SUCCESS] Neural payload delivered in {round(end - start, 2)} seconds.')
    print(f'[TENSOR] Matrix Shape: {rows} rows x {cols} features')
    print(f'[VRAM] Active Memory Footprint: {round(vram_mb, 2)} MB')
    print('[STATUS] The RTX 5060 is primed and awaiting the model architecture.')

if __name__ == '__main__':
    load_to_vram()
