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
    cycles = [20, 40, 60]
    e = 0.01
    data_collection_id = collect_parallel_two_qubit_xeb_on_grid_data(
        sampler=cirq.DensityMatrixSimulator(noise=cirq.depolarize(e),
                                            seed=65009),
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        num_circuits=20,
        repetitions=1_000,
        cycles=cycles,
        seed=14947,
        base_dir=base_dir)
    results = compute_parallel_two_qubit_xeb_on_grid_fidelities(
        data_collection_id, num_processors=32, base_dir=base_dir)

    assert len(results) == 4
    for result in results.values():
        depolarizing_model = result.depolarizing_model()
        cycle_pauli_error = (1 - depolarizing_model.decay_constant) * 15 / 16
        np.testing.assert_allclose(1 - cycle_pauli_error, (1 - e)**4)
        np.testing.assert_allclose(depolarizing_model.coefficient, 1.0)
