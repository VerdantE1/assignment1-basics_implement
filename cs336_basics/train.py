import os
import time
import torch
import numpy as np
from Transformer_LLM import Transformer
from checkpoint import load_checkpoint, save_checkpoint
from learning_rate_sche import get_learning_rate
from get_batch import get_batch
from gradient_clipping import clip_grad_norm
from cross_entropy import cross_entropy

def train(model):
    # 1. 超参数配置
    batch_size = 12  # 根据显存调整
    context_length = 1024
    max_iters = 20000
    learning_rate = 6e-4
    min_lr = 6e-5
    warmup_iters = 2000
    lr_decay_iters = 400000
    weight_decay = 0.1
    grad_clip = 1.0
    eval_interval = 500
    save_interval = 2000

    # --- 2. 数据加载 (Memory-efficient Loading) ---
    # 假设数据已经预处理为 bin 文件
    train_data = np.memmap('../data/train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('../data/val.bin', dtype=np.uint16, mode='r')

    # --- 3. 模型与优化器初始化 ---
    # model.to(device)  

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    iter_num = 0
    best_val_loss = float('inf')


    # --- 4. 训练循环 ---
    while iter_num < max_iters:
        # A. 动态调整学习率 (Scheduler)
        lr = get_learning_rate(iter_num, learning_rate, min_lr, warmup_iters, lr_decay_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

       # B. 获取数据并前馈
        X, Y = get_batch(train_data, batch_size, context_length, device)
        
        # 1. 既然模型返回两个值，我们就用两个变量接住
        logits, loss = model(X, targets=Y) 

        # 2. 删掉下面这段“手动算 loss”的代码，因为模型已经帮你算好了
        # loss = cross_entropy(logits.view(-1, ...), Y.view(-1))


        # C. 后向传播与梯度裁剪
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if grad_clip != 0.0:
            clip_grad_norm(model.parameters(), grad_clip)
        
        optimizer.step()

        # D. 日志记录与评估
        if iter_num % 100 == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}, lr {lr:.2e}")

        if iter_num % eval_interval == 0:
            val_loss = estimate_loss(model, val_data, batch_size, context_length, device)
            print(f"step {iter_num}: val loss {val_loss:.4f}")
            # 这里可以集成 Weights & Biases: wandb.log({"val/loss": val_loss})

        # E. 保存存档
        if iter_num % save_interval == 0:
            save_checkpoint(model, optimizer, iter_num, f"checkpoints/ckpt_{iter_num}.pt")

        iter_num += 1

@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device):
    """在验证集上简单评估 Loss"""
    model.eval()
    losses = []
    for _ in range(10): # 随机取 10 个 batch 做平均
        X, Y = get_batch(data, batch_size, context_length, device)
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

if __name__ == "__main__":
    vocab_size = 50257
    d_model = 512
    num_layer = 4
    num_heads = 16
    d_ff = 1344
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'

    # 实例化模型
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layer=num_layer,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device
    )
    model.to(device)

    print(f"模型实例化成功！当前设备: {device}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    train(model)