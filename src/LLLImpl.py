import numpy as np


def gram_schmidt(input_basis, threshold=1e-10):
    num_vectors = len(input_basis)
    ortho_basis = np.zeros_like(input_basis, dtype=float)
    gs_coefficients = np.zeros((num_vectors, num_vectors), dtype=float)

    for i in range(num_vectors):
        ortho_basis[i] = input_basis[i]
        for j in range(i):
            if np.linalg.norm(ortho_basis[j]) > threshold:
                gs_coefficients[i, j] = np.dot(input_basis[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j])
            else:
                print(f"Warning: Ortho vector {j} below threshold: {np.linalg.norm(ortho_basis[j])}")
            ortho_basis[i] -= gs_coefficients[i, j] * ortho_basis[j]

        if np.linalg.norm(ortho_basis[i]) < threshold:
            print(f"Warning: Vector {i} nearly zero after orthogonalization.")

    return ortho_basis, gs_coefficients


def size_reduction(lattice_basis, ortho_basis, gs_coefficients, current_index, threshold=1e-10):
    for j in range(current_index - 1, -1, -1):
        if abs(gs_coefficients[current_index, j]) > 0.5:
            lattice_basis[current_index] -= round(gs_coefficients[current_index, j]) * lattice_basis[j]
            ortho_basis, gs_coefficients = gram_schmidt(lattice_basis, threshold)
    return lattice_basis, ortho_basis, gs_coefficients


def lovasz_condition(ortho_basis, gs_coefficients, current_index, delta, threshold=1e-10):
    prev_norm = np.dot(ortho_basis[current_index - 1], ortho_basis[current_index - 1])
    curr_norm = np.dot(ortho_basis[current_index], ortho_basis[current_index])
    gs_term = gs_coefficients[current_index, current_index - 1] ** 2 * prev_norm
    condition_value = delta * prev_norm - gs_term
    if curr_norm < condition_value - threshold:
        print(f"Lovász condition failed: {curr_norm} < {condition_value}")
    return curr_norm >= condition_value


def swap_vectors(lattice_basis, current_index):
    lattice_basis[[current_index, current_index - 1]] = lattice_basis[[current_index - 1, current_index]]
    return lattice_basis


def lll_reduction(lattice_basis, delta=0.75, gs_threshold=1e-10):
    lattice_basis = np.array(lattice_basis, dtype=float)
    num_vectors = len(lattice_basis)
    iteration_count = 0
    ortho_basis, gs_coefficients = gram_schmidt(lattice_basis, gs_threshold)

    current_index = 1
    while current_index < num_vectors:
        iteration_count += 1

        lattice_basis, ortho_basis, gs_coefficients = size_reduction(
            lattice_basis, ortho_basis, gs_coefficients, current_index, gs_threshold
        )

        if lovasz_condition(ortho_basis, gs_coefficients, current_index, delta, gs_threshold):
            current_index += 1
        else:
            lattice_basis = swap_vectors(lattice_basis, current_index)
            ortho_basis, gs_coefficients = gram_schmidt(lattice_basis, gs_threshold)
            current_index = max(current_index - 1, 1)

    return lattice_basis, iteration_count


# Example Usage
if __name__ == "__main__":
    lattice = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]
    reduced_basis, iterations = lll_reduction(lattice)
    print("Reduced Basis:")
    print(reduced_basis)
    print(f"Number of iterations: {iterations}")
