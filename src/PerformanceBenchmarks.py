import numpy as np
import time
import matplotlib.pyplot as plt
import memory_profiler
import tracemalloc

from LLLImpl import *

# Benchmark LLL algorithm
def benchmark_lll(dimensions, trials=10, seed=42):

    results = {
        "dimension": [],
        "time": [],
        "mem_usage":[]
    }

    for dim in dimensions:
        total_time = 0

        for _ in range(trials):
            basis = np.random.randint(-100, 100, size=(dim, dim)).astype(float)
            (reduced_basis, iteration_count), mem_usage = measure_memory_usage(lll_algorithm, basis)
            start_time = time.time()
            reduced_basis = lll_algorithm(basis)
            end_time = time.time()

            total_time += end_time - start_time

        results["dimension"].append(dim)
        results["time"].append(total_time / trials)
        results["mem_usage"].append(mem_usage)

    return results


# Plot benchmark results
def plot_benchmark_results(results):
    fig, ax1 = plt.subplots()

    # Plot execution time
    ax1.set_xlabel("Lattice Dimension")
    ax1.set_ylabel("Runtime (S)", color="tab:blue")
    ax1.plot(results["dimension"], results["time"], color="tab:blue", marker="o", label="Runtime (S)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    # Create a twin axis for defect
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mem Usage (MB)", color="tab:red")
    ax2.plot(results["dimension"], results["mem_usage"], color="tab:red", marker="x", label="Mem Usage (MB)")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    # Add title and legend
    fig.suptitle("LLL Algorithm Performance Benchmark")
    fig.tight_layout()
    plt.show()

def measure_memory_usage(func, *args, **kwargs):
    """
    Measure peak memory usage of a function using tracemalloc.
    """
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Convert memory usage to MB
    peak_memory_mb = peak / (1024 * 1024)
    return result, peak_memory_mb

# Example usage
dimensions = range(2, 21, 2)  # Vary dimensions from 2 to 20 in steps of 2
results = benchmark_lll(dimensions, trials=5)
plot_benchmark_results(results)
