import torch
import torch.nn as nn
from cs336_basics.multi_head_self_attention import multi_head_self_attetion
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.RoPE import RotaryPositionalEmbedding
from cs336_basics.SwiGLU import SwiGLU
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = multi_head_self_attetion(d_model, num_heads)
        self.RoPE = None # 延迟初始化
        self.ffn = SwiGLU(d_model, d_ff)

    
    def forward(self, x, theta, max_seq_len, token_positions):
        

        # 1. MHDA 阶段
        residual = x 
        x = self.ln1(x)
        self.RoPE = RotaryPositionalEmbedding(theta, self.d_model // self.num_heads, max_seq_len, device = x.device)
        x = self.attn.forward(x, qk_mask=None, rope=self.RoPE, token_positions=token_positions)   # 得到x的v修正
        x = residual + x

        # 2. FFN 阶段
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x

        return x 


        