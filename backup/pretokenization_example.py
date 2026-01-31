import os
from typing import BinaryIO
import time
from collections import defaultdict
import re


FILE_PATH = "../data/TinyStoriesV2-GPT4-train.txt"
NUM_PROCESSES = 32
SPECIAL_TOKENS = b"<|endoftext|>"

SPERCIAL_TOKEN_ST = SPECIAL_TOKENS.decode("utf-8", errors="ignore")
SPECIAL_TOKEN_PATTERN = re.compile(re.escape(SPECIAL_TOKEN_STR))

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def pre_tokenization(text_chunk):
    """
    预分词逻辑：
    1. 使用正则表达式按特殊标记切分文本，保留分隔符。
    2. 遍历切分后的片段：
       - 如果是特殊标记：作为一个整体统计（不拆分）。
       - 如果是普通文本：拆分为字符列表进行统计。
    """
    stats = defaultdict(int)
    
    # 1. 构建正则表达式模式
    # re.escape: 转义特殊字符，例如 <|endoftext|> 里的 | 和 > 
    # | : 表示“或”的关系
    # (...) : 括号表示捕获分组，这样 split 结果里会保留这些特殊标记
    pattern = "(" + "|".join(re.escape(token) for token in SPERCIAL_TOKEN) + ")"
    
    # 2. 切分文本
    # 例如: "Hi[Doc]Bye" -> ['Hi', '[Doc]', 'Bye']
    parts = re.split(pattern, text_chunk)

    # 3. 处理每个片段
    for part in parts:
        if not part:  # 跳过空字符串
            continue
            
        if part in SPECIAL_TOKENS:
            # 这样 <|endoftext|> 就永远不会被拆成 '<', '|', 'e' ...
            stats[part] += 1
            
        else:
            # --- 原有逻辑 ---
            # 如果是普通文本，按字符拆分（或者按空格分词）
            # 进行 BPE 学习
            tokens = list(part) 
            for token in tokens:
                stats[token] += 1
                
    return stats


# ## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
    # for start, end in zip(boundaries[:-1], boundaries[1:]):
    #     f.seek(start)
    #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token


if __name__ == "__main__":

    # ==========================================
    # ⏱️ 开始计时
    # ==========================================
    start_time = time.time()

    with open(FILE_PATH, "rb") as f:
        # 获取文件大小
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        print(f"文件大小: {file_size / (1024*1024):.2f} MB")

        # 1.找到切分点
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, SPECIAL_TOKEN)
        global_stats = defaultdict(int)
        total_chunks = len(boundaries) - 1 # 总共有多少块

        # 2.处理数据块
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            # 👇 加上这行打印，i+1 是因为索引从 0 开始，人类习惯看 1 开始
            print(f"[进度] 处理块 {i + 1} / {total_chunks} (范围: {start} - {end})")
                
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            ## 2.1 Pre-tokenization [预分词 + 统计频率] 
            chunk_stats = pre_tokenization(chunk)

            ## 2.2 累计到全局统计
            for token, count in chunk_stats.items():
                global_stats[token] += count
            
    # ==========================================
    # ⏱️ 结束计时
    # ==========================================
        elapsed_time = time.time() - start_time

        print("-" * 40)
        print(f" 串行预分词完成！")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"共统计出 {len(global_stats)} 个不同的字符")
        # 可选：打印前 10 个最高频的字符
        sorted_stats = sorted(global_stats.items(), key=lambda x: -x[1])
        print(f"最高频的 10 个字符: {sorted_stats[:10]}")

        

