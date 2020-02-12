import numpy as np
import cirq
from cirq.experiments.grid_xeb import (
    estimate_parallel_two_qubit_xeb_fidelity_on_grid)


def test_estimate_parallel_two_qubit_xeb_fidelity_on_grid():

    qubits = cirq.GridQubit.square(2)
    two_qubit_gate = cirq.ISWAP**0.5

    # No noise, fidelities should be close to 1
    cycles = [2, 4, 6]
    results = estimate_parallel_two_qubit_xeb_fidelity_on_grid(
        sampler=cirq.Simulator(seed=50611),
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        num_circuits=10,
        repetitions=1000,
        cycles=cycles,
        seed=43435)

    assert len(results) == 4
    for result in results.values():
        fidelities = [xeb_pair.xeb_fidelity for xeb_pair in result.data]
        np.testing.assert_allclose(fidelities, 1.0, atol=0.1)

    # With depolarizing probability p
    cycles = [10, 20, 30]
    e = 0.01
    results = estimate_parallel_two_qubit_xeb_fidelity_on_grid(
        sampler=cirq.DensityMatrixSimulator(noise=cirq.depolarize(e),
                                            seed=65009),
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        num_circuits=100,
        repetitions=100_000,
        cycles=cycles,
        seed=14947)

    assert len(results) == 4
    for result in results.values():
        fidelities = [xeb_pair.xeb_fidelity for xeb_pair in result.data]
        np.testing.assert_allclose(fidelities,
                                   [(1 - e * 16 / 15)**(4 * c) for c in cycles],
                                   atol=1e-3)
