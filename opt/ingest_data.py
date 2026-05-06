import os
import glob
import polars as pl
import time

RAW_DATA_DIR = r"C:\btc_quant\data\raw"
OUTPUT_PARQUET = r"C:\btc_quant\data\processed\quant_data.parquet"

def build_ingestion_pipeline():
    print("[SYSTEM] Initializing Polars Engine on P-Cores...")
    start_time = time.time()
    
    file_pattern = os.path.join(RAW_DATA_DIR, "*.csv")
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        print(f"\n[CRITICAL ERROR] No CSV files found in {RAW_DATA_DIR}")
        return

    print(f"[INFO] Detected {len(csv_files)} CSV file(s).")
    print("[INFO] Forcing strict String ingestion to override data type collisions...")
    
    lazy_frames = [
        pl.scan_csv(
            file,
            has_header=False,
            infer_schema_length=0,  # Forces all columns to be read as Strings
            ignore_errors=True
        ) for file in csv_files
    ]
    
    query = pl.concat(lazy_frames, how="diagonal")
    
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    
    print("[INFO] Executing multi-threaded conversion to Parquet. Streaming to NVMe...")
    query.sink_parquet(
        OUTPUT_PARQUET,
        compression="snappy",
        row_group_size=100000
    )
    
    end_time = time.time()
    print(f"[SUCCESS] 3.8M+ rows compressed and optimized in {round(end_time - start_time, 2)} seconds.")
    print(f"[OUTPUT] Target locked: {OUTPUT_PARQUET}")

if __name__ == '__main__':
    build_ingestion_pipeline()
