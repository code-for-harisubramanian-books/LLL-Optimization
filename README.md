# LLL Reduction Algorithm Implementation

This repository contains an implementation of the Lenstra-Lenstra-Lovász (LLL) lattice basis reduction algorithm. The code is written in Python and aims to provide a robust and accurate solution for lattice reduction problems in practical applications.

## Features
- Implements the Gram-Schmidt orthogonalization process with a configurable threshold for numerical stability.
- Performs size reduction to ensure lattice basis vectors are shortened.
- Validates the Lovász condition to ensure the reduced basis satisfies the desired properties.
- Supports customizable parameters such as the `delta` reduction parameter and a numerical threshold for small vector detection.

## Requirements
The implementation requires the following:
- Python 3.7+
- NumPy library

To install NumPy, use:
```bash
pip install numpy
```

## Usage
### Importing the Code
Save the implementation in a Python file (e.g., `lll_reduction.py`) and import it in your project:
```python
from lll_reduction import lll_reduction
```

### Example
Here is an example of how to use the LLL reduction algorithm:
```python
import numpy as np
from lll_reduction import lll_reduction

# Define a lattice basis
lattice_basis = np.array([
    [105, 821, 605],
    [921, 1824, 2121],
    [721, 1654, 1321]
])

# Perform LLL reduction
reduced_basis, iterations = lll_reduction(lattice_basis, delta=0.75)

print("Reduced Basis:")
print(reduced_basis)
print(f"Number of iterations: {iterations}")
```

## Functions
### 1. `gram_schmidt(input_basis, threshold=1e-10)`
Performs Gram-Schmidt orthogonalization with numerical stability checks.

### 2. `compute_gs_coefficient(vector, ortho_vector, threshold=1e-10)`
Calculates the Gram-Schmidt coefficient with a safeguard against small denominators.

### 3. `size_reduction(lattice_basis, ortho_basis, gs_coefficients, current_index)`
Reduces the size of basis vectors using Gram-Schmidt coefficients.

### 4. `lovasz_condition(ortho_basis, gs_coefficients, current_index, delta, threshold=1e-10)`
Validates the Lovász condition for the current basis vector.

### 5. `swap_vectors(lattice_basis, current_index)`
Swaps two basis vectors when the Lovász condition is not satisfied.

### 6. `lll_reduction(lattice_basis, delta=0.75, gs_threshold=1e-10)`
Orchestrates the LLL reduction process.

## Parameters
- `lattice_basis`: A NumPy array representing the lattice basis.
- `delta`: Reduction parameter in the range (0.5, 1.0), default is 0.75.
- `gs_threshold`: Threshold to identify numerically small values during Gram-Schmidt orthogonalization.

## Output
- `reduced_basis`: The LLL-reduced basis as a NumPy array.
- `iteration_count`: Number of iterations performed during the reduction.

## Testing
The algorithm can be tested with random lattice bases or specific cases. Ensure that the input basis is well-conditioned for best results.

## Limitations
- The accuracy depends on the choice of the `gs_threshold` parameter.
- May require fine-tuning for highly ill-conditioned bases.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code.

---

For questions or issues, please contact the repository maintainer.
