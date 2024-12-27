import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc

from LLLImpl import *
from LLLOptimizedImpl import *

# Generate random basis
def generate_random_basis(dimension, max_value=100):
    return np.random.randint(-max_value, max_value, size=(dimension, dimension))


# Function to test LLL and collect metrics
def test_lll(basis, lll_algorithm):
    start_time = time.time()
    tracemalloc.start()
    reduced_basis = lll_algorithm(basis)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = time.time() - start_time
    iteration_count = len(reduced_basis)  # Simplified as row count for demo purposes
    return {
        "runtime": runtime,
        "memory_usage": peak / (1024 * 1024),  # Convert to MB
        "iteration_count": iteration_count,
    }


# Function to compare results for various dimensions
def compare_lll_performance(min_dim, max_dim, step=2):
    dimensions = list(range(min_dim, max_dim + 1, step))
    non_optimized_results = {"runtime": [], "memory": [], "iterations": []}
    optimized_results = {"runtime": [], "memory": [], "iterations": []}

    for dim in dimensions:
        basis = generate_random_basis(dim)

        # Test non-optimized
        non_opt_results = test_lll(basis, lll_algorithm)
        non_optimized_results["runtime"].append(non_opt_results["runtime"])
        non_optimized_results["memory"].append(non_opt_results["memory_usage"])
        non_optimized_results["iterations"].append(non_opt_results["iteration_count"])

        # Test optimized
        opt_results = test_lll(basis, lll_algorithm_optimized)
        optimized_results["runtime"].append(opt_results["runtime"])
        optimized_results["memory"].append(opt_results["memory_usage"])
        optimized_results["iterations"].append(opt_results["iteration_count"])

    return dimensions, non_optimized_results, optimized_results


# Plot comparison results
def plot_comparison(dimensions, non_optimized_results, optimized_results):
    plt.figure(figsize=(15, 10))

    # Runtime comparison
    plt.subplot(3, 1, 1)
    plt.plot(dimensions, non_optimized_results["runtime"], label="Non-Optimized", marker="o")
    plt.plot(dimensions, optimized_results["runtime"], label="Optimized", marker="s")
    plt.xlabel("Dimensions")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Comparison")
    plt.legend()

    # Memory usage comparison
    plt.subplot(3, 1, 2)
    plt.plot(dimensions, non_optimized_results["memory"], label="Non-Optimized", marker="o")
    plt.plot(dimensions, optimized_results["memory"], label="Optimized", marker="s")
    plt.xlabel("Dimensions")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage Comparison")
    plt.legend()

    # Iteration count comparison
    plt.subplot(3, 1, 3)
    plt.plot(dimensions, non_optimized_results["iterations"], label="Non-Optimized", marker="o")
    plt.plot(dimensions, optimized_results["iterations"], label="Optimized", marker="s")
    plt.xlabel("Dimensions")
    plt.ylabel("Iteration Count")
    plt.title("Iteration Count Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Run the comparison and plot results
min_dim = 2
max_dim = 50
step = 5
dimensions, non_opt_results, opt_results = compare_lll_performance(min_dim, max_dim, step)
plot_comparison(dimensions, non_opt_results, opt_results)
