import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import Tuple, Iterable, Dict, Any

class AdamW(Optimizer):
    def __init__(
        self, 
        params: Iterable[torch.nn.Parameter], 
        lr: float = 1e-3, 
        betas: Tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8, 
        weight_decay: float = 0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # 状态初始化 (First-time initialization)
                if len(state) == 0:
                    state["step"] = 0
                    # 指数移动平均 (First moment)
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # 平方指数移动平均 (Second moment)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                
                # --- AdamW 核心逻辑开始 ---

                # 1. 直接应用权重衰减 (Decoupled Weight Decay)
                # 区别于 Adam：它不计入 m_t 和 v_t 的计算
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # 2. 更新一阶和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3. 计算偏差修正 (Bias correction)
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                # 4. 计算步长
                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group["eps"])

                # 5. 更新参数
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss