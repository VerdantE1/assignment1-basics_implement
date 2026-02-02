import torch
import torch.nn as nn
from cs336_basics.scaled_dot_production_attention import scaled_dot_production_attention
from einops import rearrange

class multi_head_self_attetion(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        # 确保能整除多头
        assert(d_model % num_heads == 0)
        
        self.d_model = d_model
        self.num_heads = num_heads

        # QKV三个形变空间
        self.q_proj = nn.Linear(d_model, d_model, bias =False)
        self.k_proj = nn.Linear(d_model, d_model, bias =False)
        self.v_proj = nn.Linear(d_model, d_model, bias =False)

        # 多头整合空间
        self.output_proj = nn.Linear(d_model, d_model, bias =False)
    
    def forward(self, x, qk_mask=None, rope=None, token_positions = None):
        # x.shape(..., seq_len, d_model)
        s = x.shape[-2]
        device = x.device

        # 1. 生成查询、回答、贡献空间即QKV.
        Q_total = self.q_proj(x)   # ..., (seq_len, d_model)
        K_total = self.k_proj(x)   # ..., (seq_len, d_model)
        V_total = self.v_proj(x)   # ..., (seq_len, d_model)

        # 2. 逻辑切分多头
        Q_total = rearrange(Q_total, '... s (h di) -> ... h s di', h=self.num_heads)
        K_total = rearrange(K_total, '... s (h di) -> ... h s di', h=self.num_heads)
        V_total = rearrange(V_total, '... s (h di) -> ... h s di', h=self.num_heads)
        
        # 3. 对每一个head进行自注意力求值
        if qk_mask is None:
            qk_mask = torch.tril(torch.ones(s,s,device=device), diagonal=0)
            
        if rope is not None:
            # 根据 RoPE 实现，可能需要传入 token_positions
            Q_total = rope(Q_total, token_positions=token_positions)
            K_total = rope(K_total, token_positions=token_positions)
        

        mh_output = scaled_dot_production_attention(Q_total, K_total, V_total, qk_mask)  # ...h q di
        
        # 4. 重新整合多头
        mh_output = rearrange(mh_output, '... h q di -> ... q (h di)')  # ... d d
        
        # 5. 整合空间
        res = self.output_proj(mh_output)
        return res 



      

        



        


        