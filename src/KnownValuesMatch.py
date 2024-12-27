import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from LLLImpl import *

def test_lll_reduction(test_cases):
    """Tests the LLL algorithm with multiple input bases."""
    results = []
    for i, (input_basis, expected_output) in enumerate(test_cases, 1):
        reduced_basis, itr_count = lll_algorithm(input_basis)
        actual_output = np.array(reduced_basis).tolist()

        # Check if the actual output matches the expected output
        is_match = actual_output == expected_output

        # Collect results
        results.append({
            "Test Case": i,
            "Input Basis": input_basis,
            "Expected Output": expected_output,
            "Actual Output": actual_output,
            "Match": "Yes" if is_match else "No"
        })

    return results


def display_results_table(results):
    """Displays the test results in a table format."""
    headers = ["Test Case", "Match", "Input Basis", "Expected Output", "Actual Output"]
    table = []
    for result in results:
        table.append([
            result["Test Case"],
            result["Match"],
            result["Input Basis"],
            result["Expected Output"],
            result["Actual Output"]
        ])
    print(tabulate(table, headers=headers, tablefmt="grid"))


# Define test cases: (Input Basis, Expected Output)
test_cases = [
    (
        [[1, 1], [0, 1]],
        [[0, 1], [1, 1]]
    ),
    (
        [[1, 0, 1], [0, 1, 1], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
    ),
    (
        [[105, 821, 137], [723, 452, 842], [320, 582, 129]],
        [[215, -239, -8], [535, 343, 121], [-347, -234, 600]]
    ),
    (
        [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]],
        [[-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
    ),
    (
        [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    )
]

# Run tests and display results
results = test_lll_reduction(test_cases)
display_results_table(results)

