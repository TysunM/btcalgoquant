import polars as pl
import os

RAW_DIR = r'C:\btc_quant\data\raw'
OUTPUT = r'C:\btc_quant\data\processed\quant_data_clean.parquet'

def rebuild_matrix():
    print('\n[SYSTEM] Ingesting Standardized Minute Data (2019-2026)...')
    
    files = [
        'BTCUSDT_2019_minute.csv', 'BTCUSDT_2020_minute.csv',
        'BTCUSDT_2021_minute.csv', 'BTCUSDT_2022_minute.csv',
        'BTCUSDT_2023_minute.csv', 'BTCUSDT_2024_minute.csv',
        'BTCUSDT_2025_minute.csv', 'BTCUSDT_2026_minute.csv'
    ]
    
    dfs = []
    for f in files:
        filepath = os.path.join(RAW_DIR, f)
        if not os.path.exists(filepath):
            print(f'[WARNING] Missing file: {filepath} (Skipping...)')
            continue
            
        print(f'[INFO] Ingesting {f}...')
        
        # The files are uniformly semicolon-separated with no headers
        df = pl.read_csv(filepath, separator=';', has_header=False)
        cols = df.columns
        df = df.rename({
            cols[0]: 'Datetime_raw', cols[1]: 'Open', cols[2]: 'High',
            cols[3]: 'Low', cols[4]: 'Price', cols[5]: 'Vol'
        })
        
        # Parse the unified YYYYMMDD HHMMSS string into a strict Datetime object
        df = df.with_columns(
            pl.col('Datetime_raw').cast(pl.Utf8).str.strptime(pl.Datetime, '%Y%m%d %H%M%S', strict=False).alias('Date')
        )
        df = df.select(['Date', 'Price', 'Open', 'High', 'Low', 'Vol'])
        dfs.append(df)
            
    if not dfs:
        print('[FATAL] No valid files found in C:\\btc_quant\\data\\raw. Aborting.')
        return
        
    print('\n[INFO] Fusing all 7 years of market data...')
    master_df = pl.concat(dfs)
    
    print('[INFO] Sorting timeline (Oldest to Newest)...')
    master_df = master_df.sort('Date')
    
    print('[INFO] Deduplicating overlapping minute ticks...')
    master_df = master_df.unique(subset=['Date'], keep='first')
    
    # Standardize Change column for backward compatibility with the Triple Barrier sweep
    master_df = master_df.with_columns((pl.col('Price') - pl.col('Open')).alias('Change'))
    
    final_df = master_df.select(['Date', 'Price', 'Open', 'High', 'Low', 'Vol', 'Change'])
    final_df = final_df.drop_nulls()
    
    start_date = final_df['Date'][0]
    end_date = final_df['Date'][-1]
    print(f'\n[SUCCESS] Matrix Purified. Total Valid Rows: {len(final_df):,}')
    print(f'   -> Timeline Start: {start_date}')
    print(f'   -> Timeline End:   {end_date}')
    
    print(f'\n[INFO] Compiling to high-performance Parquet format...')
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    final_df.write_parquet(OUTPUT)
    print(f'[SYSTEM] Clean build saved. Ready for the Triple Barrier X-Ray.')

if __name__ == '__main__':
    rebuild_matrix()
