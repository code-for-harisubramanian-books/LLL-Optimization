import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from LLLImpl import *

# Theoretical complexity function
def theoretical_complexity(n, B):
    return n ** 3 * (np.log(B) ** 2)
# Evaluate efficiency
def evaluate_efficiency_comparison(dimensions, basis_generator, lll_algorithm, delta=0.75):
    data = []

    for dim in dimensions:
        # Generate basis
        basis = basis_generator(dim)
        B = np.linalg.norm(basis, ord=np.inf)

        # Measure runtime
        start_time = time.time()
        _, iteration_count = lll_algorithm(basis, delta)
        runtime = time.time() - start_time

        # Calculate theoretical complexity
        complexity = theoretical_complexity(dim, B)

        data.append({
            "Dimension": dim,
            "Runtime (s)": runtime,
            "Iteration Count": iteration_count,
            "Theoretical Complexity": complexity
        })

    return pd.DataFrame(data)

# Plot results
def plot_efficiency_comparison(df):
    """
    Plot runtime, iteration count, and theoretical complexity from the DataFrame.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Plot runtime and theoretical complexity
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Runtime (s) / Complexity", color="tab:blue")
    ax1.plot(df["Dimension"], df["Runtime (s)"], marker="o", label="Runtime (s)", color="tab:blue")
    ax1.plot(df["Dimension"], df["Theoretical Complexity"], marker="^", label="Theoretical Complexity", linestyle="--", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Add iteration count
    ax2 = ax1.twinx()
    ax2.set_ylabel("Iteration Count", color="tab:orange")
    ax2.plot(df["Dimension"], df["Iteration Count"], marker="s", label="Iteration Count", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Efficiency Evaluation: Runtime, Iteration Count, and Theoretical Complexity")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.grid()
    plt.show()

# Example basis generator
def random_basis_generator(dim):
    """
    Generate a random integer basis for a given dimension.
    """
    return np.random.randint(-50, 50, size=(dim, dim)).astype(float)

# Example usage
# Replace 'lll_algorithm' with the function name of your implemented LLL algorithm
dimensions = [2, 3, 5, 10, 15, 20,23,25,27,30]
df_efficiency = evaluate_efficiency_comparison(dimensions, random_basis_generator, lll_algorithm)

# Display table
print(df_efficiency)

# Plot results
plot_efficiency_comparison(df_efficiency)
