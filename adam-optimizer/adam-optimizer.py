import math

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here

    param_new = []
    m_new = []
    v_new = []

    for p, g, m_i, v_i in zip(param, grad, m, v):

        # Update biased first moment
        m_t = beta1 * m_i + (1 - beta1) * g

        # Update biased second moment
        v_t = beta2 * v_i + (1 - beta2) * (g ** 2)

        # Bias correction
        m_hat = m_t / (1 - beta1 ** t)
        v_hat = v_t / (1 - beta2 ** t)

        # Parameter update
        p_new = p - lr * m_hat / (math.sqrt(v_hat) + eps)

        param_new.append(p_new)
        m_new.append(m_t)
        v_new.append(v_t)

    return param_new, m_new, v_new

                
                