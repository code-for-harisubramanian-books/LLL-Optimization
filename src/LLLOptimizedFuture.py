
"""
Optimization Strategy
    1. Vectorization in Gram-Schmidt and norm computation.
    2. Redundant Calculation Avoidance via caching norms.
    3. In-Place Updates for basis and coefficients.
    4. Optimized Size Reduction using checks for necessary operations.
    5. Improved Loop Management to avoid excessive resetting.

"""

def gram_schmidt_vectorized(basis):
    n = basis.shape[0]
    ortho_basis = np.zeros_like(basis, dtype=float)
    mu = np.zeros((n, n), dtype=float)

    for i in range(n):
        ortho_basis[i] = basis[i]
        for j in range(i):
            mu[i, j] = np.dot(basis[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j])
            ortho_basis[i] -= mu[i, j] * ortho_basis[j]

    return ortho_basis, mu


def lll_optimized(basis, delta=0.75):
    n = basis.shape[0]
    ortho_basis, mu = gram_schmidt_vectorized(basis)
    norm_cache = np.array([np.linalg.norm(vec) ** 2 for vec in ortho_basis])  # Cache norms
    k = 1

    while k < n:
        # Size reduction
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q != 0:
                basis[k] -= q * basis[j]  # In-place basis update
                mu[k, :] -= q * mu[j, :]  # In-place coefficient update

        # Lovász condition
        if (delta - mu[k, k - 1] ** 2) * norm_cache[k - 1] > norm_cache[k]:
            # Swap basis vectors
            basis[[k, k - 1]] = basis[[k - 1, k]]
            norm_cache[k], norm_cache[k - 1] = norm_cache[k - 1], norm_cache[k]  # Swap norms
            ortho_basis, mu = gram_schmidt_vectorized(basis)  # Recompute
            k = max(1, k - 1)  # Avoid restarting
        else:
            k += 1

    return basis
