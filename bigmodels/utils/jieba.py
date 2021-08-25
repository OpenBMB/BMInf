import jieba
import os
jieba_cache_path = os.path.expanduser("~/.cache/jieba")
if not os.path.exists(jieba_cache_path):
    os.makedirs(jieba_cache_path)
jieba.dt.tmp_dir = jieba_cache_path
jieba.setLogLevel(20)
jieba.initialize()