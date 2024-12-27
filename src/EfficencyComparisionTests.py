import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Evaluation for efficiency comparison between pre- and post-optimized LLL

def evaluate_efficiency_comparison(dimensions, basis_generator, pre_lll, post_lll, delta=0.75):
    """
    Compare runtime and iteration count of pre- and post-optimized LLL algorithms.
    """
    data = []

    for dim in dimensions:
        basis = basis_generator(dim)

        # Pre-optimization evaluation
        start_time = time.time()
        _, pre_iterations = pre_lll(basis.copy(), delta)
        pre_runtime = time.time() - start_time

        # Post-optimization evaluation
        start_time = time.time()
        _, post_iterations = post_lll(basis.copy(), delta)
        post_runtime = time.time() - start_time

        data.append({
            "Dimension": dim,
            "Pre Runtime (s)": pre_runtime,
            "Pre Iterations": pre_iterations,
            "Post Runtime (s)": post_runtime,
            "Post Iterations": post_iterations
        })

    return pd.DataFrame(data)

# Plot comparison results
def plot_efficiency_comparison(df):
    """
    Plot comparison of runtime and iteration count.
    """
    # Plot runtime comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df["Dimension"], df["Pre Runtime (s)"], marker="o", label="Pre-Optimization Runtime")
    plt.plot(df["Dimension"], df["Post Runtime (s)"], marker="s", label="Post-Optimization Runtime")
    plt.xlabel("Dimension")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Comparison: Pre vs Post Optimization")
    plt.legend()
    plt.grid()
    plt.show()

# Example basis generator
def random_basis_generator(dim):
    return np.random.randint(-50, 50, size=(dim, dim)).astype(float)

# Example usage
# Pre-optimization and Post-optimization LLL function stubs
# Replace these with your actual pre- and post-optimized LLL functions
def pre_optimized_lll(basis, delta):
    from LLLImpl import lll_algorithm
    return lll_algorithm(basis, delta)

def post_optimized_lll(basis, delta):
    from LLLOptimizedImpl import lll_algorithm_optimized
    return lll_algorithm_optimized(basis, delta)

# Dimensions to test
dimensions = [2, 3, 5, 10, 15, 20, 25]

evaluation_df = evaluate_efficiency_comparison(dimensions, random_basis_generator, pre_optimized_lll, post_optimized_lll)
print(evaluation_df)

# Plot results
plot_efficiency_comparison(evaluation_df)
