import torch
import torch.nn as nn
import torch.nn.init as init
from cs336_basics.mylinear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None ,dtype=None):
        super().__init__()
        kwargs = {"device":device, "dtype":dtype}
        self.d_model = d_model
        self.d_ff = d_ff

        # 1. 线性层
        self.w3 = Linear(d_model, d_ff, **kwargs)

        # 2. 门控层
        self.w1 = Linear(d_model, d_ff, **kwargs)

        # 3. 映射层
        self.w2 = Linear(d_ff, d_model, **kwargs)

    def silu(self, x):
        return  x * torch.sigmoid(x)

    def forward(self,x: torch.Tensor):
        # 1. 线性层传播
        y1 = self.w3.forward(x)

        # 2. 门控层传播
        y2 = self.w1.forward(x)
        y2 = self.silu(y2)

        # 3. 哈达玛积
        y_inter = torch.mul(y1,y2)

        # 4. 映射回去
        output = self.w2(y_inter)
        return output




        