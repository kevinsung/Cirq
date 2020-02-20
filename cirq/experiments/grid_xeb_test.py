import os
import numpy as np
import cirq
from cirq.experiments.grid_xeb import (
    collect_parallel_two_qubit_xeb_on_grid_data,
    compute_parallel_two_qubit_xeb_on_grid_fidelities)


def test_estimate_parallel_two_qubit_xeb_fidelity_on_grid(tmpdir):

    base_dir = os.path.abspath(tmpdir)
    qubits = cirq.GridQubit.square(2)
    two_qubit_gate = cirq.ISWAP**0.5

    # No noise, fidelities should be close to 1
    cycles = [2, 4, 6]
    data_collection_id = collect_parallel_two_qubit_xeb_on_grid_data(
        sampler=cirq.Simulator(seed=50611),
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        num_circuits=10,
        repetitions=1000,
        cycles=cycles,
        seed=43435,
        base_dir=base_dir)
    results = compute_parallel_two_qubit_xeb_on_grid_fidelities(
        data_collection_id, base_dir=base_dir)

    assert len(results) == 4
    for result in results.values():
        fidelities = [xeb_pair.xeb_fidelity for xeb_pair in result.data]
        np.testing.assert_allclose(fidelities, 1.0, atol=0.1)

    # With depolarizing probability e
    cycles = [60, 80, 100]
    e = 0.01
    data_collection_id = collect_parallel_two_qubit_xeb_on_grid_data(
        sampler=cirq.DensityMatrixSimulator(noise=cirq.depolarize(e),
                                            seed=65009),
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        num_circuits=50,
        repetitions=10_000,
        cycles=cycles,
        seed=14947,
        base_dir=base_dir)
    results = compute_parallel_two_qubit_xeb_on_grid_fidelities(
        data_collection_id, base_dir=base_dir)

    assert len(results) == 4
    for result in results.values():
        fidelities = [xeb_pair.xeb_fidelity for xeb_pair in result.data]
        np.testing.assert_allclose(fidelities,
                                   [(1 - e * 16 / 15)**(4 * c) for c in cycles],
                                   atol=1e-3)
