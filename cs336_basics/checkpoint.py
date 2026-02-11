import torch

def save_checkpoint(model, optimizer, iteration, out):
    """
    将模型、优化器状态和迭代次数保存到存档中。
    """
    # 将所有需要恢复的“记忆”打包进一个字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # 使用 torch.save 序列化并写入文件
    # out 可以是文件路径，也可以是文件对象
    torch.save(checkpoint, out)
    print(f"Checkpoint saved at iteration {iteration}")

def load_checkpoint(src, model, optimizer):
    """
    从存档中恢复模型和优化器状态，并返回保存时的迭代次数。
    """
    # 1. 加载字典（注意：通常建议加 weights_only=True 以增强安全性，
    # 但根据题目要求，直接 load 即可）
    checkpoint = torch.load(src, map_location='cpu') # 先加载到内存，防止显存爆炸
    
    # 2. 恢复模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. 恢复优化器状态（动量、步数等）
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 4. 获取迭代次数
    iteration = checkpoint['iteration']
    
    print(f"Resuming from iteration {iteration}")
    return iteration