import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

"""
    Optimization Strategies 
    1. Reusing Calculations - In GSO, update only the affected rows 
        rather recalculating entire process after each modification
    2. Vectorization with NumPy - Leverage Numpy for Vectorization 
        by replacing loops for vectorized operations  
    3. Avoiding Redundant Calculations - Caching the norms and 
        dot products values and reusing them for subesquent use   
"""
# Incremental Gram-Schmidt orthogonalization
def incremental_gram_schmidt(basis, mu, ortho_basis, modified_index):
    """
    Incrementally update Gram-Schmidt orthogonalization for modified rows.
    """
    for i in range(modified_index, len(basis)):
        ortho_basis[i] = basis[i]
        for j in range(i):
            mu[i, j] = np.dot(basis[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j])
            ortho_basis[i] -= mu[i, j] * ortho_basis[j]
    return ortho_basis, mu

# LLL algorithm with optimizations

def lll_algorithm_optimized(basis, delta=0.75):
    """
    Optimized LLL algorithm implementation.
    """
    n = basis.shape[0]
    ortho_basis = np.zeros_like(basis, dtype=float)
    mu = np.zeros((n, n), dtype=float)
    ortho_basis, mu = incremental_gram_schmidt(basis, mu, ortho_basis, 0)

    k = 1
    iteration_count = 0

    while k < n:
        # Size reduction
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                q = round(mu[k, j])
                basis[k] -= q * basis[j]
                iteration_count += 1
                ortho_basis, mu = incremental_gram_schmidt(basis, mu, ortho_basis, k)

        # Lovasz condition
        if (delta - mu[k, k - 1] ** 2) * np.dot(ortho_basis[k - 1], ortho_basis[k - 1]) > np.dot(ortho_basis[k], ortho_basis[k]):
            # Swap
            basis[[k, k - 1]] = basis[[k - 1, k]]
            iteration_count += 1
            ortho_basis, mu = incremental_gram_schmidt(basis, mu, ortho_basis, k - 1)
            k = max(k - 1, 1)
        else:
            k += 1

    return basis, iteration_count
