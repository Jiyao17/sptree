
import numpy as np

import physical.quantum as qu
from sps.solver import test_TreeSolver, test_DPSolver, test_GRDSolver, test_EPPSolver

from solver import test_QuNetOptim


def test_path_solvers():
    import numpy as np
    np.random.seed(0)
    fids = np.random.random(10)
    fid_lower_bound = 0.95
    fid_range = 0
    fids = fids * fid_range + fid_lower_bound
    # fids = [0.9] * 10
    # print(fids)
    fth = 0.95
    edge_num = 2
    edges = {
        (i, i+1): fids[i] for i in range(edge_num)
    }
    print(fids[:edge_num])

    op = qu.GWP
    cost_cap = 100000000
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

    # print("-----------------")
    # f, allocs = test_DPSolver(edges, op, budget=int(np.ceil(sum(allocs))))
    # print("DP:")
    # print(f, )

def test_theorem(exp_num = 10000):
    import numpy as np

    gate = qu.GDP

    for i in range(exp_num):
        fs = np.random.random(4)
        cs = np.random.random(4)
        fs = 0.5 + fs * 0.5
        cs = 1 + np.floor(cs * 100)

        f1, f2, f3, f4 = fs
        c1, c2, c3, c4 = cs

        f_spp, f_pss = 0, 0
        p_spp, p_pss = 0, 0

        f13, p13 = gate.swap(f1, f3)
        p13 = f13
        c13 = (c1 + c3) / p13
        f24, p24 = gate.swap(f2, f4)
        p24 = f24
        c24 = (c2 + c4) / p24
        f_pss, p_pss = gate.purify(f13, f24)
        c_pss = (c13 + c24) / p_pss

        f12, p12 = gate.purify(f1, f2)
        c12 = (c1 + c2) / p12
        f34, p34 = gate.purify(f3, f4)
        c34 = (c3 + c4) / p34
        f_spp, p_spp = gate.swap(f12, f34)
        p_spp = f_spp
        c_spp = (c12 + c34) / p_spp

        assert f_spp >= f_pss
        assert c_spp <= c_pss

        print(f_spp, f_pss, c_spp, c_pss)


if __name__ == '__main__':

    test_path_solvers()
    # test_QuNetOptim()
    # test_theorem()
    # import random
    # import time
    # random.seed(0)
    # f = 0.5 + random.random() * 0.5
    # fs: list = [f] * 10
    # cs: list = [1] * 10

    # print(fs)

    # c_final = 0
    # random.seed(time.time())
    # while len(fs) > 1:
    #     indices = list(range(len(fs)))
    #     random.shuffle(indices)
    #     fs = [fs[i] for i in indices]
    #     cs = [cs[i] for i in indices]
        
    #     f1, f2 = fs.pop(0), fs.pop(0)
    #     c1, c2 = cs.pop(0), cs.pop(0)

    #     f, p = qu.GDP.purify(f1, f2)
    #     c = (c1 + c2) / p

    #     fs.append(f)
    #     cs.append(c)

    # print(fs, cs)
