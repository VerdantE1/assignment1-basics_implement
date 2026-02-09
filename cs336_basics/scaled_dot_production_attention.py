import torch
from torch import Tensor

def stable_softmax(x:torch.Tensor, dim: int) -> torch.Tensor:
    # 获取dim这一轴的最大值
    max_element = torch.max(x, dim=dim, keepdim=True).values

    # 广播。让x每个元素减去对应行的max_element
    exps = torch.exp(x-max_element)

    # 计算总和
    sum_exps = torch.sum(exps,dim=dim, keepdim=True)

    # 归一化得到概率分布
    return exps / sum_exps

    



def scaled_dot_production_attention(
    queries,      # Shape: (batch_size, ..., seq_len, d_k)
    keys,         # Shape: (batch_size, ..., seq_len, d_k)
    values,       # Shape: (batch_size, ..., seq_len, d_v)
    mask=None     # Optional, Shape: (seq_len, seq_len)
):
    d_k = queries.shape[-1]

    # 1. 计算注意力
    atten_scores = torch.einsum('...qd, ...kd -> ...qk',queries, keys)
    
    # 2. 缩放稳定
    atten_scores = atten_scores / torch.sqrt(torch.tensor(d_k, dtype=queries.dtype))  # ...qk

    # 3. Mask策略
    if mask is not None:
        atten_scores = atten_scores.masked_fill(mask == 0 , -1e9)  # 广播
    
    # 4. Softmax归一化
    atten_scores = stable_softmax(atten_scores, dim=-1)  # 列归一化，从key维度上归一化

    # 5. 与Value相乘
    output = torch.einsum('...qk, ...kv -> ...qv', atten_scores, values)

    return output


