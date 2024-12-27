import numpy as np
from tabulate import tabulate

from LLLImpl import lll_reduction as lll_algorithm



# Random basis generator
def generate_random_basis(dimension, max_value=50):
    return np.random.randint(-max_value, max_value, size=(dimension, dimension))

# Lovász condition checker
def lovasz_condition(basis):
    for i in range(1, len(basis)):
        mu = np.dot(basis[i], basis[i - 1]) / np.dot(basis[i - 1], basis[i - 1])
        if not (0.75 <= 1 - mu ** 2 <= 1):
            return False
    return True


# Size reduction checker
def size_reduction(basis):
    for i in range(1, len(basis)):
        for j in range(i):
            mu = np.dot(basis[i], basis[j]) / np.dot(basis[j], basis[j])
            if abs(mu) > 0.5:
                return False
    return True


# Orthogonality checker
def orthogonality_checker(basis, threshold=1e-10):
    for i in range(len(basis)):
        for j in range(i):
            dot_product = np.dot(basis[i],basis[j])
            if abs(dot_product) > threshold :
                print(f"Non-orthogonal vectors: {i}, {j} with dot {dot_product}")
                return False
    return True


# Test correctness across dimensions
def test_lll_correctness(min_dim, max_dim, num_tests_per_dim=5):
    results = []

    for dim in range(min_dim, max_dim + 1):
        for test_id in range(num_tests_per_dim):
            input_basis = generate_random_basis(dim)
            reduced_basis, itr_count = lll_algorithm(input_basis)

            # Calculate properties
            rank_match = np.linalg.matrix_rank(input_basis) == np.linalg.matrix_rank(reduced_basis)
            determinant_match = not np.isclose(np.linalg.det(input_basis), 0) and np.isclose(
                np.abs(np.linalg.det(input_basis)), np.abs(np.linalg.det(reduced_basis))
            )
            lovasz_satisfied = lovasz_condition(reduced_basis)
            size_reduction_satisfied = size_reduction(reduced_basis)
            orthogonality_satisfied = orthogonality_checker(reduced_basis, 1e-10)

            # Append results
            results.append({
                "Dimension": dim,
                "Rank Match": "Pass" if rank_match else "Fail",
                "Determinant Match": "Pass" if determinant_match else "Fail",
                "Lovasz Satisfied": "Pass" if lovasz_satisfied else "Fail",
                "Size Reduction Satisfied": "Pass" if size_reduction_satisfied else "Fail",
                "Orthogonality Satisfied": "Pass" if orthogonality_satisfied else "Fail",
            })

    return results


# Run the tests and display results
min_dimension = 5
max_dimension = 25
num_tests = 2  # Number of tests per dimension

test_results = test_lll_correctness(min_dimension, max_dimension, num_tests)
summary_table = [
    [
        result["Dimension"],
        result["Rank Match"],
        result["Determinant Match"],
        result["Lovasz Satisfied"],
        result["Size Reduction Satisfied"],
        result["Orthogonality Satisfied"],
    ]
    for result in test_results
]

headers = [
    "Dimension",
    "Rank Match",
    "Determinant Match",
    "Lovasz Satisfied",
    "Size Reduction Satisfied",
    "Orthogonality Satisfied",
]
print(tabulate(summary_table, headers=headers, tablefmt="grid"))
