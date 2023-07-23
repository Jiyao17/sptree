
import time

import numpy as np
import networkx as nx

import physical.quantum as qu
from physical.network import QuNet, QuNetTask
from sps.solver import SolverType
from physical.topology import ATT, IBM, RandomPAG, RandomGNP
from solver import test_QuNetOptim
from utils.tools import draw_lines




def test_dp_tp_by_fth(exp_num):
    gate=qu.GDP
    # topology=ATT()
    # topology=RandomGNP(100, 0.1)
    topology = RandomPAG(150, 2)
    node_memory=100
    edge_capacity=(300, 301)
    # edge_capacity=(100, 101)
    edge_fidelity=(0.7, 0.95)
    user_pair_num=30
    path_num=3
    req_num_range=(10, 10)
    # req_fid_range=(0.99, 0.99)

    # req_fids = [0.7, 0.8, 0.9, 0.99, 0.999]
    error = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # error = [1e-1, 1e-3, 1e-5]
    req_fids = [ 1 - n for n in error]
    tps = np.zeros((len(req_fids), 3))
    times = np.zeros((len(req_fids), 3))

    qunet = QuNet(topology, gate)
    qunet.net_gen(node_memory, edge_capacity, edge_fidelity)
    task = QuNetTask(qunet)
    task.set_user_pairs(user_pair_num)
    task.set_up_paths(path_num)
    for i, fth in enumerate(req_fids):
        task.workload_gen(req_num_range, (fth, fth))

        tree_tp, grdy_tp, epp_tp = 0, 0, 0
        tree_time, grdy_time, epp_time = 0, 0, 0
        for j in range(exp_num):
            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.TREE)
            tree_time += time.time() - start
            tree_tp += ObjVal

            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.GRD)
            grdy_time += time.time() - start
            grdy_tp += ObjVal

            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.EPP)
            epp_time += time.time() - start
            epp_tp += ObjVal

        tree_tp, grdy_tp, epp_tp = tree_tp / exp_num, grdy_tp / exp_num, epp_tp / exp_num
        tree_time, grdy_time, epp_time = tree_time / exp_num, grdy_time / exp_num, epp_time / exp_num

        tps[i] = [tree_tp, grdy_tp, epp_tp]
        times[i] = [tree_time, grdy_time, epp_time]

        print("fidelity {} done".format(fth))

    x = error
    ys = [tps[:, 0], tps[:, 1], tps[:, 2]]
    labels = ["Tree", "GRDY", "EPP"]
    xlabel = "Infidelity Threshold"
    ylabel = "Throughput"
    filename = "../data/dp_net_tp_.png"
    draw_lines(x, ys, labels, xlabel, ylabel, xscale='log', xreverse=True, filename=filename)

    ys = [times[:, 0], times[:, 1], times[:, 2]]
    ylabel = "Time (s)"
    filename = "../data/dp_net_time.png"
    draw_lines(x, ys, labels, xlabel, ylabel, xscale='log', xreverse=True, filename=filename)

def test_wn_tp_by_noise(exp_num):
    solver_type=SolverType.TREE
    # gate=qu.GWP
    topology=ATT()
    node_memory=100
    edge_capacity=(500, 501)
    edge_fidelity=(0.95, 1.0)
    user_pair_num=10
    path_num=3
    req_num_range=(10, 10)
    req_fid_range=(0.99, 0.99)

    # req_fids = [0.7, 0.8, 0.9, 0.99, 0.999]
    error = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    req_fids = [ 1 - n for n in error]
    tps = np.zeros((len(req_fids), 3))
    times = np.zeros((len(req_fids), 3))

    for i, fth in enumerate(req_fids):
        tree_tp, grdy_tp, epp_tp = 0, 0, 0
        tree_time, grdy_time, epp_time = 0, 0, 0
        for j in range(exp_num):
            start = time.time()
            ObjVal = test_QuNetOptim(solver_type, gate, topology,
                node_memory, edge_capacity, edge_fidelity,
                user_pair_num, path_num,
                req_num_range, (fth, fth))
            tree_time += time.time() - start
            tree_tp += ObjVal

            start = time.time()
            ObjVal = test_QuNetOptim(SolverType.GRD, gate, topology,
                node_memory, edge_capacity, edge_fidelity,
                user_pair_num, path_num,
                req_num_range, (fth, fth))
            grdy_time += time.time() - start
            grdy_tp += ObjVal

            start = time.time()
            ObjVal = test_QuNetOptim(SolverType.EPP, gate, topology,
                node_memory, edge_capacity, edge_fidelity,
                user_pair_num, path_num,
                req_num_range, (fth, fth))
            epp_time += time.time() - start
            epp_tp += ObjVal

        tree_tp, grdy_tp, epp_tp = tree_tp / exp_num, grdy_tp / exp_num, epp_tp / exp_num
        tree_time, grdy_time, epp_time = tree_time / exp_num, grdy_time / exp_num, epp_time / exp_num

        tps[i] = [tree_tp, grdy_tp, epp_tp]
        times[i] = [tree_time, grdy_time, epp_time]

        print("fidelity {} done".format(fth))

    x = error
    ys = [tps[:, 0], tps[:, 1], tps[:, 2]]
    labels = ["Tree", "GRDY", "EPP"]
    xlabel = "Infidelity Threshold"
    ylabel = "Throughput"
    filename = "../data/dp_net_tp_.png"
    draw_lines(x, ys, labels, xlabel, ylabel, xscale='log', xreverse=True, filename=filename)

    ys = [times[:, 0], times[:, 1], times[:, 2]]
    ylabel = "Time (s)"
    filename = "../data/dp_net_time.png"
    draw_lines(x, ys, labels, xlabel, ylabel, xscale='log', xreverse=True, filename=filename)



if __name__ == '__main__':

    test_dp_tp_by_fth(10)




