import numpy as np
import tiktoken
import os

def process_file(file_path, output_path):
    enc = tiktoken.get_encoding("gpt2")
    print(f"基础词表大小: {enc.n_vocab}")
    
    if not os.path.exists(file_path):
        print(f"找不到文件: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # 编码
    ids = enc.encode_ordinary(data)
    ids_array = np.array(ids, dtype=np.uint16)
    
    # 保存为二进制
    ids_array.tofile(output_path)
    print(f"{file_path} 处理完成 -> {output_path} ({len(ids)} tokens)")

# 处理两个文件
process_file('../data/TinyStoriesV2-GPT4-train.txt', '../data/train.bin')
process_file('../data/TinyStoriesV2-GPT4-valid.txt', '../data/val.bin')