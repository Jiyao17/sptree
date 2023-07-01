
import physical.quantum as qu
from sps.solver import test_TreeSolver, test_DPSolver, test_GRDSolver, test_EPPSolver

from solver import test_QuNetOptim


def test_path_solvers():
    import numpy as np
    np.random.seed(0)
    fids = np.random.random(100)
    fid_lower_bound = 0.5
    fid_range = 0.5
    fids = fids * fid_range + fid_lower_bound
    # fids = [0.9] * 10
    # print(fids)
    fth = 0.9
    edge_num = 10
    edges = {
        (i, i+1): fids[i] for i in range(edge_num)
    }
    print(fids[:edge_num])

    op = qu.GDP
    cost_cap = 100
    print("-----------------")
    # f, allocs = test_TreeSolver(edges, op, fth, cost_cap, True)
    f, allocs = test_TreeSolver(edges, op, fth, cost_cap, False)
    print("Tree:")
    print(f, sum(allocs), allocs)

    print("-----------------")
    f, allocs = test_GRDSolver(edges, op, fth, cost_cap, )
    print("GRD:")
    print(f, sum(allocs), allocs)

    print("-----------------")
    f, allocs = test_EPPSolver(edges, op, fth, cost_cap, )
    print("EPP:")
    print(f, sum(allocs), allocs)

    print("-----------------")
    f, allocs = test_DPSolver(edges, op, budget=int(np.ceil(sum(allocs))))
    print("DP:")
    print(f, )


if __name__ == '__main__':

    test_QuNetOptim()

