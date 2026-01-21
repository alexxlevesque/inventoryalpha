from src.data.ingestor import DataIngestor
import time

def test_ingestor():
    path = 'M5 Forecasting Accuracy'
    ingestor = DataIngestor(path)
    ingestor.load_raw()
    
    print('Loading data for HOBBIES_1_001_CA_1...')
    start_time = time.time()
    
    try:
        data = ingestor.get_clean_series('HOBBIES_1_001', 'CA_1')
        duration = time.time() - start_time
        print(f'Done in {duration:.4f}s')
        print(f'Data shape: {data.shape}')
        print(f'First 5 values: {data[:5]}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    test_ingestor()
