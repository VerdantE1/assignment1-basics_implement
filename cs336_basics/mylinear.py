import torch
import torch.nn as nn 
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 device=None, 
                 dtype=None
                 ):
        super().__init__()
    
        # 保存维度信息,设备信息压缩成包
        self.in_features = in_features
        self.out_features = out_features
        device_kwargs = {"device": device, "dtype": dtype}

        # 定义模型参数
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **device_kwargs)
        )
        
        # 初始化参数
        self.reset_parameter()
    
    def reset_parameter(self) -> None:
        weights_std = (2 / (self.in_features + self.out_features)) ** 0.5
        init.trunc_normal_(self.weight, mean=0, std=weights_std,a=-3*weights_std,b=3*weights_std)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()




# 1. Linear继承nn.Module使得能够注册到Pytorch框架模块。享受Pytorch关于Module的参数管理、设备管理、保存加载等

