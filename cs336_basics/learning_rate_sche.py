import math

def get_learning_rate(t, alpha_max, alpha_min, Tw, Tc):
    # 1. Warm-up stage
    if t < Tw:
        return (t/Tw) * alpha_max

    # 2. Post-anneling stage
    if t > Tc:
        return alpha_min
    
    # 3. Consine anneling stage
    progress = (t - Tw) / (Tc - Tw)
    consine_out = 0.5 * (1 + math.cos(progress * math.pi))
    return alpha_min + consine_out * (alpha_max - alpha_min)