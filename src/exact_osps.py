

# reuse code in this project for some experiments in the follow-up work on exact OSPS

from time import time

import numpy as np

from src.physical.quantum import GDP, GDP_LOSH, GDP_LOSL
from src.sps.solver import test_DPSolver, test_EPPSolver, test_GRDSolver, test_TreeSolver
from src.utils.tools import test_edges_gen


def run(repeat=100):
    np.random.seed(42)
    gate = GDP_LOSH # 0.75
    # gate = GDP_LOSL # 0.5
    # gate = GDP      # 1
    edge_lengths = [2, 3, 5, 7, 10, 13, 15,]
    # edges_list = [test_edges_gen(length, (0.8, 0.95)) for length in edge_lengths]
    fths = [0.9, 0.95, 0.99]
    
    solvers = {
        'EPP': test_EPPSolver,
        'FGER': test_GRDSolver,
        'TREE': test_TreeSolver,
    }
    # resulting costs
    results = {}
    times = {}
    # solvers
    for fth in fths:
        print(f"running for fth = {fth}")
        results[fth] = {}
        times[fth] = {}
        for _ in range(repeat):

            edges_list = [test_edges_gen(length, (0.7, 0.95)) for length in edge_lengths]
            for length, edges in zip(edge_lengths, edges_list):
                if length not in results[fth]:
                    results[fth][length] = {}
                    times[fth][length] = {}

                for solver_name, solver_func in solvers.items():
                    if solver_name not in results[fth][length]:
                        results[ fth][length][solver_name] = 0
                        times[fth][length][solver_name] = 0

                    start = time()
                    f, costs = solver_func(edges, gate, fth, 1e6)
                    end = time()
                    cost = sum(costs)
                    results[fth][length][solver_name] += cost / repeat
                    times[fth][length][solver_name] += (end - start) / repeat


    # dump results
    import pickle
    with open('baseline_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open('baseline_times.pkl', 'wb') as f:
        pickle.dump(times, f)

    # draw result costs of different solvers
    # edge num as x, cost as y, each solver as a line
    from src.utils.tools import draw_lines
    for fth in fths:
        x = edge_lengths
        ys = []
        labels = []
        markers = ['o', 's', '^']
        for solver_name in solvers.keys():
            y = [results[fth][length][solver_name] for length in edge_lengths]
            ys.append(y)
            labels.append(solver_name)
        draw_lines(
            x, ys,
            xlabel='Path Length (Number of Edges)',
            ylabel='Average Cost',
            labels=labels,
            markers=markers,
            xscale='linear',
            yscale='linear',
            filename=f'exact_osps_fth_{int(fth*100)}.png'
        )
        
    # draw result times of different solvers
    for fth in fths:
        x = edge_lengths
        ys = []
        labels = []
        markers = ['o', 's', '^']
        for solver_name in solvers.keys():
            y = [times[fth][length][solver_name] for length in edge_lengths]
            ys.append(y)
            labels.append(solver_name)
        draw_lines(
            x, ys,
            xlabel='Chain Length',
            ylabel='Average Time (seconds)',
            labels=labels,
            markers=markers,
            xscale='linear',
            yscale='linear',
            filename=f'exact_osps_time_fth_{int(fth*100)}.png'
        )


if __name__ == "__main__":
    run()