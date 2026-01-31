import torch
import torch.nn as nn
import torch.nn.init as init

class Embbeding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        # 1.存储参数信息
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embedding_dim
        kwargs = {"device":device,"dtype":dtype}

        # 2.为本层参数创建空间
        self.weights = nn.Parameter(
            torch.empty(
            self.num_embeddings,
            self.embeddings_dim,
            **kwargs
            )
        )

        # 3.初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        init.trunc_normal_(self.weights, 0, 1, -3, 3)

    # Lookup Table
    def forward(self,token_ids):
        """
        forward 的 Docstring
        
        :param token_idx: [batch_sz, seq_len] , each elements is Int.
        
        Returns:
            [batch_sz, seq_len, embedding] 

        """
        return self.weights[token_ids]

