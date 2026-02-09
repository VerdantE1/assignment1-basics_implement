import torch
import torch.nn.functional as F

def cross_entropy(logits, targets):
    """
    计算交叉熵损失
    
    参数:
    - logits: 预测张量，形状为 (Batch..., Vocab_Size)
    - targets: 目标张量，形状为 (Batch...)，包含类别的索引
    
    特点:
    - 自动处理任意数量的 batch 维度
    - 使用 Max-subtraction 保证数值稳定
    - 结合 Log-Sum-Exp 简化计算
    """
    # 1. 确定词表维度
    vocab_dim = -1
    
    # 2. 数值稳定性：减去最大值 M
    # M 的形状是 (batch_size, 1)
    M, _ = torch.max(logits, dim=vocab_dim, keepdim=True)
    
    # 3. 计算 LogSumExp 部分：log(sum(exp(oi - M))) + M
    # 这一步抵消了大量的 exp 和 log，且保证了 log 的参数不为 0
    logits_stable = logits - M
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=vocab_dim, keepdim=True)) + M
        
    # 4. 提取目标类别的 Logit：oi[xi+1]
    # targets.unsqueeze(-1) 将 (B,) 变为 (B, 1)
    target_logits = torch.gather(logits, vocab_dim, targets.unsqueeze(-1))
    
    # 5. 根据公式：Loss = -target_logit + LogSumExp
    # 这里完全抵消了 log(exp(...))
    loss = -target_logits + log_sum_exp
    
    # 6. 处理所有 batch 维度并取平均
    return torch.mean(loss)