import os
import gzip
import shutil
import requests
from urllib.parse import urlsplit


def get(url):
    print(f'fetching {url}')
    response = requests.get(url)
    if response.status_code != 200:
        print(f'get response.status_code = {response.status_code} for {url}')

    filename = os.path.basename(urlsplit(url).path)
    gz_filepath = os.path.join('data', filename)

    print(f'writing {gz_filepath}')
    with open(gz_filepath, 'wb') as f:
        f.write(response.content)

    data_filepath, _ = os.path.splitext(gz_filepath)
    print(f'extracting {gz_filepath} -> {data_filepath}')
    with gzip.open(gz_filepath, 'rb') as f_in:
        with open(data_filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f'removing {gz_filepath}')
    os.remove(gz_filepath)


def _main():
    if not os.path.isdir('data'):
        os.makedirs('data')

    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]
    for url in urls:
        get(url)


if __name__ == "__main__":
    _main()
