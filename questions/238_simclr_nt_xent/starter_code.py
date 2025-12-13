def nt_xent_loss(z: np.ndarray, temperature: float) -> float:
    """
    Compute the NT-Xent loss for contrastive learning.
    
    Args:
        z: L2-normalized embeddings, shape (2N, embedding_dim)
           Positive pairs: z[2k] and z[2k+1] are views of image k
        temperature: Temperature scaling parameter (τ > 0)
    
    Returns:
        The scalar NT-Xent loss value
    """
    pass