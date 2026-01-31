#!/bin/bash

# 创建并进入 data 目录
mkdir -p data
cd data

# --- 使用国内镜像下载 TinyStories ---
# 将链接中的 huggingface.co 替换为 hf-mirror.com

echo "正在从国内镜像下载 TinyStories 训练集..."
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

echo "正在从国内镜像下载 TinyStories 验证集..."
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt


# --- 下载 OpenWebText (OWT) ---
# 注意：Stanford 的数据集在 Hugging Face 上也有镜像，同样替换域名
echo "正在下载 OpenWebText 训练集 (压缩包)..."
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz

echo "正在解压 owt_train.txt.gz..."
gunzip owt_train.txt.gz

echo "正在下载 OpenWebText 验证集 (压缩包)..."
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

echo "正在解压 owt_valid.txt.gz..."
gunzip owt_valid.txt.gz

echo "所有数据集下载并解压完成！"
cd ..
