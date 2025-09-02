
"""
Module for automatically applying PCS to arbitrary circuits(To-be-implemented) and extracting counts.
"""

import cirq
import matplotlib.pyplot as plt
import math
'''
Getting the counts from simulation result in terms of base-d representation
'''
def big_endian_dits_to_base(dits) -> str:
        return ''.join(str(e) for e in dits)

def get_counts(sampled_circuit: cirq.study.result.ResultDict, key: str) -> dict:
    res = sampled_circuit.histogram(key=key, fold_func=lambda v: big_endian_dits_to_base(v))
    return dict(res)


def plot_histogram(readouts: dict):
    """
Plotting measurement values for PCS experiments
"""
    data = readouts
    states = list(data.keys())
    counts = list(data.values())
    total = sum(counts)

    plt.bar(states, counts)
    plt.ylim(0, total)
    plt.xlabel("States")
    plt.ylabel("Counts")
    plt.title(total)
    plt.show()

def postselection(readouts: dict,ancillas: int) -> dict:
    """
    Post-selection function for PCS experiments
    """
    filtered_readouts = {state: count for state, count in readouts.items() if state.endswith('0'*ancillas)}
    return filtered_readouts

'''
Hellinger fidelity code
'''
# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
def hellinger_fidelity(dist_p: dict, dist_q: dict) -> float:

    p_sum = sum(dist_p.values())
    q_sum = sum(dist_q.values())

    p_normed = {}
    for key, val in dist_p.items():
        p_normed[key] = val / p_sum

    q_normed = {}
    for key, val in dist_q.items():
        q_normed[key] = val / q_sum

    total = 0
    for key, val in p_normed.items():
        if key in q_normed:
            total += (math.sqrt(val) - math.sqrt(q_normed[key])) ** 2
            del q_normed[key]
        else:
            total += val
    total += sum(q_normed.values())

    dist = math.sqrt(total) / math.sqrt(2)
    return (1 - dist**2) ** 2