#!/usr/bin/env python3
import os
import urllib.request
import ssl
import tarfile
import time

def download_with_retry(url, dest, max_retries=5, timeout=120):
    """Download file with retry logic"""
    for i in range(max_retries):
        try:
            context = ssl.create_default_context()
            with urllib.request.urlopen(url, context=context, timeout=timeout) as response:
                with open(dest, 'wb') as f:
                    f.write(response.read())
            print(f'Downloaded {dest}')
            return True
        except Exception as e:
            print(f'Retry {i+1}/{max_retries}: {e}')
            if i == max_retries - 1:
                raise
            time.sleep(60)
    return False

def main():
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    data_dir = '/mnt/datasets/cifar100'
    os.makedirs(data_dir, exist_ok=True)
    
    tar_path = os.path.join(data_dir, 'cifar-100-python.tar.gz')
    download_with_retry(url, tar_path)
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)
    print('Extraction complete')
    
    # Verify
    train_path = os.path.join(data_dir, 'cifar-100-python', 'train')
    test_path = os.path.join(data_dir, 'cifar-100-python', 'test')
    assert os.path.exists(train_path), 'Train data missing'
    assert os.path.exists(test_path), 'Test data missing'
    print(f'CIFAR-100 ready at {data_dir}')

if __name__ == '__main__':
    main()
