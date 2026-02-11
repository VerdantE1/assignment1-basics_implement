import torch
import numpy as np 

def get_batch(x, batch_size, context_length, device):
    """
    从 numpy 数组中采样一批输入序列和目标序列。
    
    参数:
    x: np.ndarray, 原始 token ID 数组
    batch_size: int, 每批次的样本数
    context_length: int, 每个样本的序列长度
    device: str, 目标设备 ('cpu', 'cuda:0' 等)
    
    返回:
    (inputs, targets): 形状均为 (batch_size, context_length) 的 torch.Tensor
    """
    # 1. 确定合法的起始位置范围
    # 需要留出一个位置给 target，所以减 1
    max_idx = len(x) - context_length - 1

    # 2. 随机生成 batch_size 个起始索引
    ix = np.random.randint(0, max_idx + 1, size=(batch_size,))

    # 3. 根据索引提取数据
    inputs_list = [x[i : i + context_length] for i in ix]
    targets_list = [x[i + 1 : i + context_length + 1] for i in ix]

    # 4. 转化为tensor并移动到device
    # 4. 转换为 Tensor 并移动到 device
    # 使用 np.stack 提高效率，再转为 torch.from_numpy
    inputs = torch.from_numpy(np.stack(inputs_list)).to(device).long()
    targets = torch.from_numpy(np.stack(targets_list)).to(device).long()

    return inputs, targets


