import torch
import torch.nn as nn

from cs336_basics.embedding import Embedding
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.mylinear import Linear
from cs336_basics.stable_softmax import stable_softmax

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, 
                 num_layer, num_heads, d_ff,
                 device=None, dtype=None ):
        super().__init__()

        kwargs = {"device":device, "dtype":dtype}
        # token_embedding
        self.token_embeddings = Embedding(vocab_size, d_model,**kwargs)

        # tf_blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layer)
        ])

        # final rms layer
        self.ln_final = RMSNorm(d_model)

        # output embedding
        # 最后根据该token的特征去生成最后的vocab_size分数
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices, theta=10000.0, context_length=None):
        # 1. Token Embedding
        # 输入是句子的索引Tensor, 输出是将特征转化为嵌入向量: [b,s] -> [b,s,d]
        x = self.token_embeddings(in_indices)

        # 2. TransformerBlock: 每个Block计算 preNorm -> (RoPE -> MHDA) -> preNorm -> FFN 
        b,s = x.shape[0],x.shape[1]
        token_positions = torch.arange(s,device = x.device)

        for block in self.layers:
            x = block(x, theta, context_length, token_positions)
        
        # 3. Final RMS Layer
        x = self.ln_final(x)

        # 4. 生成vocab分数, 以及softmax归一化
        logits = self.lm_head(x)

        return logits
       

        
