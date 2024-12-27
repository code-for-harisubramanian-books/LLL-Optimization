import numpy as np

from LLLImpl import lll_reduction as lll_algorithm


def basic_test(lll_function):
    # Basic Test 1
    basis_1 = np.array([[105, 821, 137], [723, 452, 842], [320, 582, 129]])
    expected_output_1 = np.array([[215, -239, -8], [535, 343, 121], [-347, -234, 600]])
    reduced_basis_1, itr_count = lll_function(basis_1)
    assert np.allclose(reduced_basis_1, expected_output_1), "Test 1 Failed"
    print("Basic Test 1 Passed: Reduced Basis")
    print(reduced_basis_1)

    # Basic Test 2
    basis_2 = np.array([[1, 1], [0, 1]])
    expected_output_2 = np.array([[0, 1], [1, 0]])
    reduced_basis_2, itr_count = lll_function(basis_2)
    assert np.allclose(reduced_basis_2, expected_output_2), "Test 2 Failed"
    print("Basic Test 2 Passed: Reduced Basis")
    print(reduced_basis_2)

# Run Basic Tests
basic_test(lll_algorithm)
