import numpy as np
from formula import calculate_box_payout

def linear_scaling(offset, max_offset=100, min_scale=1, max_scale=10):
    scaled = (abs(offset) / max_offset) * (max_scale - min_scale) + min_scale
    return min(max_scale, max(min_scale, scaled))

def sigmoid_scaling(offset, max_offset=100, k=0.1, min_scale=1, max_scale=10):
    x = k * (abs(offset) - max_offset / 2)
    sig = 1 / (1 + np.exp(-x))
    scaled = min_scale + (max_scale - min_scale) * sig
    return scaled

def scale_payout(offset, method='linear', **kwargs):
    if method == 'linear':
        return linear_scaling(offset, **kwargs)
    elif method == 'sigmoid':
        return sigmoid_scaling(offset, **kwargs)
    else:
        raise ValueError(f"Unsupported scaling method '{method}'. Use 'linear' or 'sigmoid'.")

def scale_payout_vectorized(offsets, method='linear', **kwargs):
    func = linear_scaling if method == 'linear' else sigmoid_scaling if method == 'sigmoid' else None
    if func is None:
        raise ValueError(f"Unsupported scaling method '{method}'. Use 'linear' or 'sigmoid'.")
    return np.array([func(offset, **kwargs) for offset in offsets])

def calculate_and_scale_payout(
    S0, Klower, Kupper, ts, te, r, sigma, P, chit=0.1, cmiss=0.1,
    F=0.01, M=20.0, hit=True, offset=0, scaling_method='linear', scaling_kwargs={}
):
    """
    Calculate the raw box payout and scale it based on offset with  chosen method.
    """
    raw_payout = calculate_box_payout(S0, Klower, Kupper, ts, te, r, sigma, P, chit, cmiss, F, M, hit)
    scale = scale_payout(offset, method=scaling_method, **scaling_kwargs)
    return raw_payout * scale

# Example usage (remove or comment out for production)
if __name__ == "__main__":
    # Test offsets
    test_offsets = [-150, -50, 0, 25, 75, 150]
    print("Scaled payoffs for test offsets based on raw payout:")
    for offset in test_offsets:
        payout = calculate_and_scale_payout(
            S0=122200, Klower=122200-2.5, Kupper=122200+2.5,
            ts=0.0001, te=0.0002, r=0.01, sigma=0.8, P=1,
            offset=offset, scaling_method='sigmoid', scaling_kwargs={'max_offset':100, 'k':0.1}
        )
        print(f"Offset {offset}: Scaled payout = {payout:.4f}")
