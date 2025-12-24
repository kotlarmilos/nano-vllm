import math

def adamw_step(w, g, m, v, t, *,
               lr=1e-3,
               beta1=0.9,
               beta2=0.999,
               eps=1e-8,
               weight_decay=0.01):
    # Update moments
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g * g)

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # AdamW update (decoupled weight decay)
    w = w - lr * (m_hat / (math.sqrt(v_hat) + eps) + weight_decay * w)

    return w, m, v