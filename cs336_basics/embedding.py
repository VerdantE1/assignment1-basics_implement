import torch
import torch.nn as nn
import torch.nn.init as init

class Embedding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        # 1.存储参数信息
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embedding_dim
        kwargs = {"device":device,"dtype":dtype}

        # 2.为本层参数创建空间
        self.weight = nn.Parameter(
            torch.empty(
            self.num_embeddings,
            self.embeddings_dim,
            **kwargs
            )
        )

        # 3.初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
            # 1. 先在 CPU 上创建一个同样的权重张量
            # 假设 self.weight 已经在 GPU 上了，我们建一个 CPU 版的副本
            cpu_weight = torch.empty_like(self.weight, device='cpu')
            
            # 2. 在 CPU 上进行初始化 (这一步不会触发 nvrtc 报错)
            torch.nn.init.trunc_normal_(cpu_weight, 0, 1, -3, 3)
            
            # 3. 将初始化好的值拷贝回 GPU 上的 self.weight
            with torch.no_grad():
                self.weight.copy_(cpu_weight)

    # Lookup Table
    def forward(self,token_ids):
        """
        forward 的 Docstring
        
        :param token_idx: [batch_sz, seq_len] , each elements is Int.
        
        Returns:
            [batch_sz, seq_len, embedding] 

        """
        return self.weight[token_ids]

