import pytest
from quantem.pauli_checks import postselect_counts
from qiskit.quantum_info import hellinger_fidelity
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator,noise

class Testpostselection:
    @pytest.fixture
    def ideal_test_circ(self):
        test_qc_base = QuantumCircuit(3)
        test_qc_base.h(0)
        test_qc_base.x(1)
        test_qc_base.cx(0,1)
        test_qc_base.cx(0,1)
        test_qc_base.h(0)
        test_qc_base.measure_active()

        return test_qc_base
    
    @pytest.fixture
    def pcs_test_circ(self):
        test_qc = QuantumCircuit(3)
        test_qc.h(0)
        test_qc.h(2)
        test_qc.cx(2,1)
        test_qc.x(1)
        test_qc.cx(0,1)
        test_qc.cx(0,1)
        test_qc.h(0)
        test_qc.cx(2,1)
        test_qc.h(2)
        test_qc.measure_active()
        return test_qc
    
    @pytest.fixture
    def simulated_counts(self,pcs_test_circ,ideal_test_circ):
        gate_err_x = noise.depolarizing_error(0.4, 1)
        noise_model_x = noise.NoiseModel()
        noise_model_x.add_all_qubit_quantum_error(gate_err_x,instructions=['x'] )
        noisy_sim = AerSimulator(noise_model=noise_model_x) # noisy simulator with depolarizing noise on x gate
        ideal_sim = AerSimulator() # ideal sim

        ideal_counts = ideal_sim.run(ideal_test_circ,seed_simulator=1, shots=100).result().get_counts()
        noisy_counts = noisy_sim.run(ideal_test_circ,seed_simulator=1, shots=100).result().get_counts()
        noisy_counts_pcs = noisy_sim.run(pcs_test_circ, seed_simulator=1, shots=100).result().get_counts()

        return ideal_counts,noisy_counts,noisy_counts_pcs
    
    def test_final_counts(self,simulated_counts):
        ideal_counts,_,pcs_counts = simulated_counts
        final_counts = postselect_counts(pcs_counts,['+1'],num_ancillas=1)
        assert isinstance(final_counts,dict)
        assert len(final_counts) < len(pcs_counts)
        # for key in final_counts.keys():
        #     assert len(key) == len(ideal_counts[0])

    def test_fidelity(self,simulated_counts):
        ideal_counts,noisy_counts,pcs_counts = simulated_counts
        unmitigated_fidelity= hellinger_fidelity(ideal_counts,noisy_counts)
        post_pcs_counts = postselect_counts(pcs_counts,['+1'],num_ancillas=1)
        mitigated_fidelity = hellinger_fidelity(ideal_counts,post_pcs_counts)
        assert mitigated_fidelity > unmitigated_fidelity

    def test_positive_polarity(self,simulated_counts):
        _,_,pcs_counts = simulated_counts
        final_counts = postselect_counts(pcs_counts,['+1'],num_ancillas=1)
        expected_counts = {}
        for key,count in pcs_counts.items():
            if '1' not in key[:1]:
                expected_counts[key[1:]] = count
        assert final_counts == expected_counts

