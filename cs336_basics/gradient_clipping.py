import torch

def clip_grad_norm(parameters, max_norm, eps=1e-6):

    # 1. 过滤掉没有梯度的参数
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    
    # 2. 计算所有梯度拼接后的 L2 范数
    #公式: total_norm = sqrt(sum(element^2))
    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g, 2) for g in grads]), 
        2
    )

    # 3. 计算缩放系数
    # 如果 total_norm > max_norm，则系数 < 1；否则系数 >= 1
    # 但我们只在 total_norm 超过 max_norm 时才缩小它
    clip_coeff = max_norm / (total_norm + eps)
    
    # 4. 原地修改梯度
    # 只有当 clip_coeff < 1 时才进行操作（即总范数超过了阈值）
    if clip_coeff < 1.0:
        for g in grads:
            g.detach().mul_(clip_coeff)
            
    return total_norm
    

