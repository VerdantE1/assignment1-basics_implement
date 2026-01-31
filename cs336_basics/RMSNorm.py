import torch.nn as nn
import torch 
import torch.nn.init as init

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5, device=None, dtype=None):
        super().__init__()
        kwargs={"device":device,"dtype":dtype}

        self.d_model = d_model
        self.weights = nn.Parameter(torch.ones(d_model,**kwargs))
        self.eps = eps
        

    def _compute_rms(self,x):
        return torch.sqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    

    ## 处理embedding维度，归一化
    ## x[batch_sz,sqe_len, embedding_dim]
    def forward(self, x):
        # 1. 转化成高精度浮点数防止溢出
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # 2. 计算RMS值
        rms = self._compute_rms(x)

        # 3. RMS标准化
        x = x/rms * self.weights

        # 4. 返回x的原本类型
        x = x.to(in_dtype)
        return x