def nt_xent_loss(z: np.ndarray, temperature: float) -> float:
    N = z.shape[0]
    sim = (z @ z.T) / temperature
    
    sim_exp = np.exp(sim - np.max(sim, axis=1, keepdims=True))
    mask_diag = ~np.eye(N, dtype=bool)
    denominator = np.sum(sim_exp * mask_diag, axis=1)
    
    indices = np.arange(N)
    pos_indices = indices + 1 - 2 * (indices % 2)
    numerator = sim_exp[indices, pos_indices]
    
    losses = -np.log(numerator / denominator)
    
    return float(np.mean(losses))