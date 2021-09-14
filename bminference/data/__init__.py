import os
from tqdm import tqdm
import requests


_SOURCE_URL_PREFIX = "https://thunlp.oss-cn-qingdao.aliyuncs.com/bigmodels/"
_CACHE_PATH = os.path.expanduser("~/.cache/bigmodels/")


def set_source(url : str):
    global _SOURCE_URL_PREFIX

    if not url.endswith("/"):
        url = url + "/"
    _SOURCE_URL_PREFIX = url

def get_source() -> str:
    global _SOURCE_URL_PREFIX
    return _SOURCE_URL_PREFIX

def set_cache_path(path : str):
    global _CACHE_PATH
    _CACHE_PATH = path

def get_cache_path() -> str:
    global _CACHE_PATH
    return _CACHE_PATH

def ensure_file(model_name : str, filename : str):
    if model_name.startswith("file://"):
        return os.path.join(model_name[ len("file://"): ], filename)
    url = _SOURCE_URL_PREFIX + model_name + "/" + filename
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('Content-Length', 0))

    target_path = os.path.join(_CACHE_PATH, model_name, filename)
    if os.path.exists( target_path ):
        if os.stat( target_path ).st_size == total_size_in_bytes or total_size_in_bytes == 0:
            return target_path

    if not os.path.exists( os.path.join(_CACHE_PATH, model_name) ):
        os.makedirs( os.path.join(_CACHE_PATH, model_name) )
    
    block_size = 1024
    
    with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading %s/%s" % (model_name, filename)) as p_bar:
        with open(target_path, 'wb') as file:
            for data in response.iter_content(block_size):
                p_bar.update(len(data))
                file.write(data)
    return target_path