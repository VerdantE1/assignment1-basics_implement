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
                 max_seq_len=1024, theta=10000.0, # 增加这两个配置参数
                 device=None, dtype=None):
        super().__init__()

        kwargs = {"device": device, "dtype": dtype}
        self.token_embeddings = Embedding(vocab_size, d_model, **kwargs)

        # 在这里把配置传给每一个 Block
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta=theta, max_seq_len=max_seq_len)
            for _ in range(num_layer)
        ])

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices, targets=None): # 简化参数
            # 1. Token Embedding
            x = self.token_embeddings(in_indices)

            # 2. 准备位置信息
            b, s = x.shape[0], x.shape[1]
            token_positions = torch.arange(s, device=x.device)

            # 3. 逐层通过 TransformerBlock
            for block in self.layers:
                # 关键：这里只传 x 和位置信息，对齐 TransformerBlock 的新 forward 签名
                x = block(x, token_positions) 
            
            # 4. Final LayerNorm & Head
            x = self.ln_final(x)
            logits = self.lm_head(x)

            # 5. 计算 Loss (为了兼容你的训练循环 logits, loss = model(X, Y))
            loss = None
            if targets is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1)
                )

            return logits, loss

        
