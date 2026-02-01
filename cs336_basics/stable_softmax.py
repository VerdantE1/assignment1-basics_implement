import torch

def stable_softmax(x:torch.Tensor, dim: int) -> torch.Tensor:
    # 获取dim这一轴的最大值
    max_element = torch.max(x, dim=dim, keepdim=True).values

    # 广播。让x每个元素减去对应行的max_element
    exps = torch.exp(x-max_element)

    # 计算总和
    sum_exps = torch.sum(exps,dim=dim, keepdim=True)

    # 归一化得到概率分布
    return exps / sum_exps

    
