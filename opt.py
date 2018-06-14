import math
import numpy as np

def warmup_cosine(x, warmup=0.002):
    pass

def warmup_constant(x, warmup=0.002):
    pass

def warmup_linear(x, warmup=0.002):
    pass

schedules = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}

def adam(params, grads, lr, schedule, t_total, b1=0.9, b2=0.999, e=1e-8, l2=0, vector_l2=False, max_grad_norm=-1, **kwargs):
    """
    adam with weight decay fix
    """
    pass
