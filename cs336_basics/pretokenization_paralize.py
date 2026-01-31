import os
from typing import BinaryIO
import time
from collections import defaultdict
import multiprocessing as mp
import re

FILE_PATH = "../data/TinyStoriesV2-GPT4-train.txt"
NUM_PROCESSES = 32
SPECIAL_TOKENS = b"<|endoftext|>"

SPECIAL_TOKEN_STR = SPECIAL_TOKENS.decode("utf-8", errors="ignore")
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

def worker(chunk_data:str):
    """
    对一个 chunk 的文本做：
    1. 先按特殊 token <|endoftext|> 拆成若干段（文档）；
    2. 对每个文档内部做预分词 / 统计；
    3. 不跨 <|endoftext|> 进行合并。
    """
    locate_stats = defaultdict(int)

    segments = SPECIAL_TOKEN_PATTERN.split(chunk_data)

    for segment in segments:
        # 跳过空段（可能在开头/结尾或连续 special token 时出现）
        if not segment:
            continue 

        for ch in segment:
            locate_stats[ch] += 1

    return locate_stats

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

        # 1.找到切分点（在字节级按 <|endoftext|> 对齐）
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, SPECIAL_TOKENS)
        print(f"将文件切分为 {len(boundaries) - 1} 个块进行并行处理...")

        global_stats = defaultdict(int)
        total_chunks = len(boundaries) - 1  # 总共有多少块

        # 2.切分chunks（读成字符串）
        chunks = []  # 这个列表用来存所有的数据块 (字符串)
        with open(FILE_PATH, "rb") as f2:
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                print(f"[准备] 加载块 {i + 1} / {len(boundaries) - 1}")
                f2.seek(start)
                raw_data = f2.read(end - start)
                # 解码 (注意：错误处理)
                text_chunk = raw_data.decode("utf-8", errors="ignore")

                chunks.append(text_chunk)

        # 3.启动多进程
        print(f"启动 {NUM_PROCESSES} 个进程进行并行统计...")
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(worker, chunks)

        # 4.归约
        print("正在合并结果...")
        for local_dict in results:
            for token, count in local_dict.items():
                global_stats[token] += count

        # ==========================================
        # ⏱️ 结束计时
        # ==========================================
        elapsed_time = time.time() - start_time

        print("-" * 40)
        print(f" 并行预分词完成！")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"共统计出 {len(global_stats)} 个不同的字符")
        # 可选：打印前 10 个最高频的字符
        sorted_stats = sorted(global_stats.items(), key=lambda x: -x[1])
        print(f"最高频的 10 个字符: {sorted_stats[:10]}")