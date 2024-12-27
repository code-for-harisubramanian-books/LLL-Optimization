import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

from LLLImpl import *

# Benchmarking function
def benchmark_lll(input_basis):
    process = psutil.Process()
    start_mem = process.memory_info().rss / 1024 / 1024  # Memory in MB
    start_time = time.time()
    output_basis = lll_algorithm(input_basis)
    end_time = time.time()
    end_mem = process.memory_info().rss / 1024 / 1024  # Memory in MB

    runtime = end_time - start_time
    memory_usage = end_mem - start_mem
    cpu_usage = process.cpu_percent(interval=0.1)

    return runtime, memory_usage, cpu_usage, output_basis

# Scaling evaluation
input_sizes = [5, 10, 15, 20, 25,27,30,32]  # Example sizes for input basis
scaling_results = []

for size in input_sizes:
    input_basis = np.random.randint(-20, 20, size=(size, size))

    runtime, memory_usage, cpu_usage, _ = benchmark_lll(input_basis)
    scaling_results.append([size, runtime, memory_usage, cpu_usage])


# Plotting runtime and memory usage
def plot_scaling_metrics(results):
    sizes = [result[0] for result in results]
    runtimes = [result[1] for result in results]
    memory_usages = [result[2] for result in results]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Dimensions")
    ax1.set_ylabel("Runtime (s)", color="tab:blue")
    ax1.plot(sizes, runtimes, label="Runtime", color="tab:blue", marker="o")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Memory Usage (MB)", color="tab:green")
    ax2.plot(sizes, memory_usages, label="Memory Usage", color="tab:green", marker="x")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.legend(loc="upper right")

    plt.title("Scaling Analysis: Runtime and Memory Usage")
    plt.show()


# Display scaling plots
plot_scaling_metrics(scaling_results)
