import torch
import torch.nn as nn
from cs336_basics.multi_head_self_attention import multi_head_self_attetion
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.RoPE import RotaryPositionalEmbedding
from cs336_basics.SwiGLU import SwiGLU
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=10000.0, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = multi_head_self_attetion(d_model, num_heads)
        
        # --- 核心修改：在初始化时创建，且只创建一次 ---
        self.rope = RotaryPositionalEmbedding(
            theta=theta, 
            d_k=d_model // num_heads, 
            max_seq_len=max_seq_len
        )
        
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions): # 去掉那些不需要的参数
        # 1. MHDA 阶段
        residual = x 
        x = self.ln1(x)
        
        # 直接把初始化好的 self.rope 传进去
        x = self.attn.forward(x, qk_mask=None, rope=self.rope, token_positions=token_positions)
        x = residual + x

        # 2. FFN 阶段
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x

        return x
        