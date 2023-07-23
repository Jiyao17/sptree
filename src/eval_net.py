
import time

import numpy as np

import physical.quantum as qu
from sps.solver import SolverType
from physical.topology import ATT
from solver import test_QuNetOptim
from utils.tools import draw_lines


def test_dp_tp_by_user(fth, exp_num):
    solver_type=SolverType.TREE, 
    gate=qu.GWH,
    topology=ATT(),
    node_memory=100,
    edge_capacity=(26, 35),
    edge_fidelity=(0.7, 0.95),
    user_pair_num=50,
    path_num=5,
    req_num_range=(1, 10),
    req_fid_range=(0.99, 0.99),

    req_fids = [0.7, 0.8, 0.9, 0.99, 0.999]
    tps = np.zeros((len(req_fids), 3))
    times = np.zeros((len(req_fids), 3))

    for i, fth in enumerate(req_fids):
        tree_tp, grdy_tp, epp_tp = 0, 0, 0
        tree_time, grdy_time, epp_time = 0, 0, 0
        for i in range(exp_num):
            start = time.time()
            ObjVal = test_QuNetOptim(solver_type, gate, topology,
                node_memory, edge_capacity, edge_fidelity,
                user_pair_num, path_num,
                req_num_range, req_fid_range)
            tree_time += time.time() - start
            tree_tp += ObjVal

            start = time.time()
            ObjVal = test_QuNetOptim(SolverType.GRD, gate, topology,
                node_memory, edge_capacity, edge_fidelity,
                user_pair_num, path_num,
                req_num_range, req_fid_range)
            grdy_time += time.time() - start
            grdy_tp += ObjVal

            start = time.time()
            ObjVal = test_QuNetOptim(SolverType.EPP, gate, topology,
                node_memory, edge_capacity, edge_fidelity,
                user_pair_num, path_num,
                req_num_range, req_fid_range)
            epp_time += time.time() - start
            epp_tp += ObjVal

        tree_tp, grdy_cost, epp_cost = tree_tp / exp_num, grdy_cost / exp_num, epp_cost / exp_num
        tree_time, grdy_time, epp_time = tree_time / exp_num, grdy_time / exp_num, epp_time / exp_num

        tps[req_fids.index(fth)] = [tree_tp, grdy_tp, epp_tp]
        times[req_fids.index(fth)] = [tree_time, grdy_time, epp_time]


    x = req_fids
    ys = [tps[:, 0], tps[:, 1], tps[:, 2]]
    labels = ["Tree", "GRDY", "EPP"]
    xlabel = "Fidelity Threshold"
    ylabel = "Throughput"
    filename = "../data/dp_net_tp_.png"
    draw_lines(x, ys, labels, xlabel, ylabel, filename)

    ys = [times[:, 0], times[:, 1], times[:, 2]]
    ylabel = "Time (s)"
    filename = "../data/dp_net_time.png"
    draw_lines(x, ys, labels, xlabel, ylabel, filename)



if __name__ == '__main__':



    test_QuNetOptim()




