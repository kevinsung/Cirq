from typing import Dict, Iterable, List, NamedTuple, TYPE_CHECKING, Tuple

import collections
import datetime
import itertools
import os

import numpy as np

from cirq import circuits, devices, ops, sim, value
from cirq.experiments.cross_entropy_benchmarking import (CrossEntropyResult,
                                                         CrossEntropyPair)
from cirq.experiments.random_quantum_circuit_generation import (
    GridInteractionLayer,
    random_rotations_between_grid_interaction_layers_circuit)

if TYPE_CHECKING:
    import cirq

BASE_DIR = os.path.expanduser(
    os.path.join('~', 'cirq-results', 'grid-parallel-xeb'))

LAYER_A = GridInteractionLayer(col_offset=0, vertical=True, stagger=True)
LAYER_B = GridInteractionLayer(col_offset=1, vertical=True, stagger=True)
LAYER_C = GridInteractionLayer(col_offset=1, vertical=False, stagger=True)
LAYER_D = GridInteractionLayer(col_offset=0, vertical=False, stagger=True)

SINGLE_QUBIT_GATES = [
    ops.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a)
    for a, z in itertools.product(np.linspace(0, 7 / 8, 8), repeat=2)
]


class CircuitAndTrialResult(NamedTuple):
    circuit: 'cirq.Circuit'
    trial_result: 'cirq.TrialResult'


class CrossEntropyData(NamedTuple):
    num_cycle: int
    data: List[CircuitAndTrialResult]


def collect_parallel_two_qubit_xeb_on_grid_data(
        sampler: 'cirq.Sampler',
        qubits: Iterable['cirq.GridQubit'],
        two_qubit_gate: 'cirq.Gate',
        *,
        num_circuits: int = 20,
        repetitions: int = 100_000,
        cycles: Iterable[int] = range(2, 103, 10),
        seed: 'cirq.value.RANDOM_STATE_LIKE' = None,
        data_collection_id: Optional[str] = None) -> None:
    if data_collection_id is None:
        data_collection_id = datetime.datetime.now().isoformat()
    qubits = list(qubits)
    coupled_qubit_pairs = _coupled_qubit_pairs(qubits)
    prng = value.parse_random_state(seed)
    max_cycles = max(cycles)

    data = collections.defaultdict(
        list)  # type: Dict[GridInteractionLayer, List[CrossEntropyData]]

    for layer in (LAYER_A, LAYER_B, LAYER_C, LAYER_D):
        print(f'layer {layer}')
        print()
        circuits_ = [
            random_rotations_between_grid_interaction_layers_circuit(
                qubits=qubits,
                depth=max_cycles,
                two_qubit_op_factory=lambda a, b, _: two_qubit_gate(a, b),
                pattern=[layer],
                single_qubit_gates=SINGLE_QUBIT_GATES,
                add_final_single_qubit_layer=False,
                seed=prng) for _ in range(num_circuits)
        ]
        for depth in cycles:
            print(f'depth {depth}')
            truncated_circuits = [circuit[:2 * depth] for circuit in circuits_]
            circuits_and_trial_results = []
            for truncated_circuit in truncated_circuits:
                truncated_circuit.append(ops.measure(*qubits, key='m'))
                trial_result = sampler.run(truncated_circuit,
                                           repetitions=repetitions)
                circuit_and_trial_result = CircuitAndTrialResult(
                    circuit=truncated_circuit, trial_result=trial_result)
                circuits_and_trial_results.append(circuit_and_trial_result)
            xeb_data = CrossEntropyData(num_cycle=depth,
                                        data=circuits_and_trial_results)
            data[layer].append(xeb_data)

    fn = os.path.join(BASE_DIR, data_collection_id, 'raw_data.json')
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    cirq.to_json(list(data.items()), fn)


def estimate_parallel_two_qubit_xeb_fidelity_on_grid(
        sampler: 'cirq.Sampler',
        qubits: Iterable['cirq.GridQubit'],
        two_qubit_gate: 'cirq.Gate',
        *,
        num_circuits: int = 20,
        repetitions: int = 100_000,
        cycles: Iterable[int] = range(2, 103, 10),
        seed: 'cirq.value.RANDOM_STATE_LIKE' = None
) -> Dict[Tuple['cirq.GridQubit', 'cirq.GridQubit'], CrossEntropyResult]:

    qubits = list(qubits)
    coupled_qubit_pairs = _coupled_qubit_pairs(qubits)
    prng = value.parse_random_state(seed)
    max_cycles = max(cycles)
    xeb_results = {}

    for layer in (LAYER_A, LAYER_B, LAYER_C, LAYER_D):
        print(f'layer {layer}')
        print()
        circuits_ = [
            random_rotations_between_grid_interaction_layers_circuit(
                qubits=qubits,
                depth=max_cycles,
                two_qubit_op_factory=lambda a, b, _: two_qubit_gate(a, b),
                pattern=[layer],
                single_qubit_gates=SINGLE_QUBIT_GATES,
                add_final_single_qubit_layer=False,
                seed=prng) for _ in range(num_circuits)
        ]
        active_qubit_pairs = [
            pair for pair in coupled_qubit_pairs if pair in layer
        ]
        xeb_pairs = collections.defaultdict(list)
        for depth in cycles:
            print(f'depth {depth}')
            truncated_circuits = [circuit[:2 * depth] for circuit in circuits_]
            trial_results = []
            for truncated_circuit in truncated_circuits:
                truncated_circuit.append(ops.measure(*qubits, key='m'))
                trial_result = sampler.run(truncated_circuit,
                                           repetitions=repetitions)
                trial_results.append(trial_result)
            for qubit_pair in active_qubit_pairs:
                print(f'pair {qubit_pair}')
                print()
                fidelity = _get_xeb_fidelity(qubit_pair, truncated_circuits,
                                             trial_results)
                xeb_pairs[qubit_pair].append(CrossEntropyPair(depth, fidelity))
        for qubit_pair, xeb_pair in xeb_pairs.items():
            xeb_results[qubit_pair] = CrossEntropyResult(
                data=xeb_pair, repetitions=repetitions)

    return xeb_results


def _get_xeb_fidelity(qubit_pair, circuits_, trial_results) -> float:
    a, b = qubit_pair
    numerator = 0
    denominator = 0
    for circuit, trial_result in zip(circuits_, trial_results):
        measurement_qubits = circuit[-1].operations[0].qubits
        # Get the measurement indices of this qubit pair
        qubit_indices = [
            measurement_qubits.index(a),
            measurement_qubits.index(b)
        ]
        # Get the measurements of this qubit pair
        restricted_measurements = trial_result.measurements['m'][:,
                                                                 qubit_indices]
        # Convert length-2 bitstrings to integers
        restricted_measurements_ints = (2 * restricted_measurements[:, 0] +
                                        restricted_measurements[:, 1])
        # Get the circuit restricted to this qubit pair
        restricted_circuit = circuits.Circuit(
            op for op in circuit[:-1].all_operations()
            if not set(op.qubits).isdisjoint(qubit_pair))
        # Compute the theoretical probabilities
        amplitudes = sim.final_wavefunction(restricted_circuit,
                                            qubit_order=qubit_pair)
        probabilities = np.abs(amplitudes)**2
        # Compute the values needed for fidelity calculation
        experimental_value = 4 * np.mean(
            probabilities[restricted_measurements_ints])
        theoretical_value = 4 * np.sum(probabilities**2)
        numerator += (experimental_value - 1) * (theoretical_value - 1)
        denominator += (theoretical_value - 1)**2
    return numerator / denominator


def _coupled_qubit_pairs(qubits: List['cirq.GridQubit'],
                        ) -> List[Tuple['cirq.GridQubit', 'cirq.GridQubit']]:
    pairs = []
    qubit_set = set(qubits)
    for qubit in qubits:

        def add_pair(neighbor: 'cirq.GridQubit'):
            if neighbor in qubit_set:
                pairs.append((qubit, neighbor))

        add_pair(devices.GridQubit(qubit.row, qubit.col + 1))
        add_pair(devices.GridQubit(qubit.row + 1, qubit.col))

    return pairs
