import torch
import torch.nn as nn 
import torch.nn.init as init

class RotaryPositionalEmbedding(nn.Module):
        def __init__(self, theta: float, d_k: float, max_seq_len: int, device=None):
            super().__init__()
            
            # 每个token的embedding维度d_k，按两两分组。得到每一组的频率
            indices = torch.arange(0, d_k, 2).float()
            freqs = 1.0 / (theta ** (indices / d_k)) 
            
            # 得到一个sequence里所有的token的旋转角度
            m = torch.arange(max_seq_len).float()
            angles = torch.outer(m, freqs)   # 行是token i, 列是每个token的每组旋转角度
            
            # [token i , group k angles]
            self.register_buffer("cos", torch.cos(angles))
            self.register_buffer("sin", torch.sin(angles))
            
        def forward(self, x, token_positions) -> torch.Tensor:
            # 1. 获取所有维度
            *batch_dim, seq_len, d_k = x.shape

            # 2. 重塑张量: 将维度分组成 (..., seq_len, d_k/2, 2)
            # 注意：d_k/2 在 python3 中是 float，view 需要 int
            x_reshaped = x.view(*batch_dim, seq_len, d_k // 2, 2)

            # 3. 拆分实部与虚部表示第一个和第二个元素
            x_real = x_reshaped[..., 0] # (..., seq_len, d_k/2)
            x_imag = x_reshaped[..., 1] # (..., seq_len, d_k/2)

            # 4. 根据 token_positions 提取对应的 cos 和 sin
            # self.cos 形状为 (max_seq_len, d_k/2)
            # 提取后形状为 (..., seq_len, d_k/2)
            cos = self.cos[token_positions] 
            sin = self.sin[token_positions]

            # 5. 应用旋转变换 (复数乘法逻辑)
            # x_out_real = x_real * cos - x_imag * sin
            # x_out_imag = x_real * sin + x_imag * cos
            x_out_real = x_real * cos - x_imag * sin
            x_out_imag = x_real * sin + x_imag * cos

            # 6. 合并并恢复形状
            # stack 之后形状为 (..., seq_len, d_k/2, 2)
            out = torch.stack([x_out_real, x_out_imag], dim=-1)
            
            # flatten 回原始形状 (..., seq_len, d_k)
            return out.flatten(start_dim=-2)



