
import time
import copy

import numpy as np
import networkx as nx

import physical.quantum as qu
from physical.network import QuNet, QuNetTask
from sps.solver import SolverType
from physical.topology import ATT, IBM, RandomPAG, RandomGNP
from solver import test_QuNetOptim
from utils.tools import draw_lines




def test_dp_tp_by_fth(size, exp_num):
    gate=qu.GDP
    if size == 'small':
        topology=ATT()
    elif size == 'medium':
        topology = RandomGNP(50, 0.1)
    elif size == 'large':
        topology=RandomPAG(100, 2)

    node_memory=(100, 101)
    edge_capacity=(50, 51)
    # edge_capacity=(100, 101)
    edge_fidelity=(0.7, 0.95)
    if size == 'small':
        user_pair_num=13
    elif size == 'medium':
        user_pair_num=25
    elif size == 'large':
        user_pair_num = 50
    path_num=5
    req_num_range=(10, 10)
    # req_fid_range=(0.99, 0.99)

    # req_fids = [0.7, 0.8, 0.9, 0.99, 0.999]
    # error = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    error = [1e-1, 1e-2, 1e-3, 1e-4]
    req_fids = [ 1 - n for n in error]
    labels = ["TREE", "GRDY", "EPP", "NESTED_F", "NESTED_C", "DP-1.2", "DP-1.3"]
    
    tps = np.zeros((len(req_fids), len(labels)))
    times = np.zeros((len(req_fids), len(labels)))

    qunet = QuNet(topology, gate)
    qunet.net_gen(node_memory, edge_capacity, edge_fidelity)
    task = QuNetTask(qunet)
    task.set_user_pairs(user_pair_num)
    task.set_up_paths(path_num)

    for i, fth in enumerate(req_fids):
        task.workload_gen(req_num_range, (fth, fth))

        for j in range(exp_num):
            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.TREE)
            tps[i, 0] += ObjVal
            times[i, 0] += time.time() - start

            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.GRD)
            tps[i, 1] += ObjVal
            times[i, 1] += time.time() - start


            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.EPP)
            tps[i, 2] += ObjVal
            times[i, 2] += time.time() - start

            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.NESTED_F)
            tps[i, 3] += ObjVal
            times[i, 3] += time.time() - start

            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.NESTED_C)
            tps[i, 4] += ObjVal
            times[i, 4] += time.time() - start

            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.DP, 1.2)
            tps[i, 5] += ObjVal
            times[i, 5] += time.time() - start

            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.DP, 1.3)
            tps[i, 6] += ObjVal
            times[i, 6] += time.time() - start

        tps[i] /= exp_num
        times[i] /= exp_num

        print("fidelity {} done".format(fth))

    x = error
    ys = [tps[:, 0], tps[:, 1], tps[:, 2], tps[:, 3], tps[:, 4], tps[:, 5], tps[:, 6]]
    xlabel = "Infidelity Threshold"
    ylabel = "Throughput"
    markers = ['o', 's', 'v', 'x', 'd', 'p', '*']
    filename = "../data/net/dp_net_tp_{}.png".format(size)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, xscale='log', xreverse=True, filename=filename)

    ys = [times[:, 0], times[:, 1], times[:, 2], times[:, 3], times[:, 4], times[:, 5], times[:, 6]]
    ylabel = "Time (s)"
    filename = "../data/net/dp_net_time_{}.png".format(size)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, xscale='log', xreverse=True, filename=filename)


def create_task(
        topology=ATT(),
        gate=qu.GWP,
        node_memory=(50, 100),
        edge_capacity=(300, 301),
        edge_fidelity=(0.95, 1),
        user_pair_num=10,
        path_num=3,
        req_num_range=(10, 10),
        fth=0.99
    ):

    qunet = QuNet(topology, gate)
    qunet.net_gen(node_memory, edge_capacity, edge_fidelity)
    task = QuNetTask(qunet)
    task.set_user_pairs(user_pair_num)
    task.set_up_paths(path_num)
    task.workload_gen(req_num_range, (fth, fth))

    return task


def test_wn_tp_by_fth(size, exp_num):
    # gate=qu.GDP
    if size == 'small':
        topology=ATT()
    elif size == 'medium':
        topology=RandomGNP(50, 0.1)
    elif size == 'large':
        topology = RandomPAG(100, 2)
    node_memory=(100, 101)
    edge_capacity=(50, 51)
    # edge_capacity=(100, 101)
    edge_fidelity=(0.95, 1)
    if size == 'small':
        user_pair_num=13
    elif size == 'medium':
        user_pair_num=25
    elif size == 'large':
        user_pair_num = 50
    path_num=5
    req_num_range=(10, 10)
    # req_fid_range=(0.99, 0.99)

    gates = [qu.GWP, qu.GWH, qu.GWM, qu.GWL]
    # req_fids = [0.7, 0.8, 0.9, 0.99, 0.999]
    # error = [1e-1, 7.5e-2, 5e-2, 2.5e-2, 1e-2]
    error = [0.125, 0.1, 0.075, 0.05, 0.025]
    # error = [1e-1, 1e-3, 1e-5]
    req_fids = [ 1 - n for n in error]
    tps = np.zeros((len(req_fids), len(gates)))
    times = np.zeros((len(req_fids), len(gates)))

    for i, fth in enumerate(req_fids):
        p_tp, h_tp, m_tp, l_tp = 0, 0, 0, 0
        p_time, h_time, m_time, l_time = 0, 0, 0, 0
        for j in range(exp_num):
            task = create_task(topology, qu.GWP, 
                node_memory, edge_capacity, edge_fidelity, 
                user_pair_num, path_num, 
                req_num_range, fth)
            task1, task2, task3 = copy.deepcopy(task), copy.deepcopy(task), copy.deepcopy(task)
            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.TREE)
            p_time += time.time() - start
            p_tp += ObjVal

            task1.qunet.gate = qu.GWH
            start = time.time()
            ObjVal = test_QuNetOptim(task1, SolverType.TREE)
            h_time += time.time() - start
            h_tp += ObjVal

            task2.qunet.gate = qu.GWM
            start = time.time()
            ObjVal = test_QuNetOptim(task2, SolverType.TREE)
            m_time += time.time() - start
            m_tp += ObjVal


            task3.qunet.gate = qu.GWL
            start = time.time()
            ObjVal = test_QuNetOptim(task3, SolverType.TREE)
            l_time += time.time() - start
            l_tp += ObjVal


        p_tp, h_tp, m_tp, l_tp = p_tp / exp_num, h_tp / exp_num, m_tp / exp_num, l_tp / exp_num
        p_time, h_time, m_time, l_time = p_time / exp_num, h_time / exp_num, m_time / exp_num, l_time / exp_num

        tps[i] = [p_tp, h_tp, m_tp, l_tp]
        times[i] = [p_time, h_time, m_time, l_time]

        print("fidelity {} done".format(fth))

    x = error
    ys = tps.T
    labels = ["P", "H", "M", "L"]
    xlabel = "Infidelity Threshold"
    ylabel = "Throughput"
    markers = ['o', 's', 'v', 'x']
    filename = "../data/net/wn_net_tp_{}.png".format(size)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, xreverse=True, filename=filename)

    ys = times.T
    ylabel = "Time (s)"
    filename = "../data/net/wn_net_time_{}.png".format(size)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, xreverse=True, filename=filename)


def test_wn_tp_by_methods(size, exp_num, gate=qu.GWP):
    # gate=qu.GDP
    if gate == qu.GWP:
        gate_desc = "P"
    elif gate == qu.GWH:
        gate_desc = "H"
    elif gate == qu.GWM:
        gate_desc = "M"
    elif gate == qu.GWL:
        gate_desc = "L"

    if size == 'small':
        topology=ATT()
    elif size == 'medium':
        topology=RandomGNP(50, 0.1)
    elif size == 'large':
        topology = RandomPAG(100, 2)
    node_memory=(100, 101)
    edge_capacity=(50, 51)
    # edge_capacity=(100, 101)
    edge_fidelity=(0.95, 1)
    if size == 'small':
        user_pair_num=13
    elif size == 'medium':
        user_pair_num=25
    elif size == 'large':
        user_pair_num = 50
    path_num=5
    req_num_range=(10, 10)
    # req_fid_range=(0.99, 0.99)

    # req_fids = [0.7, 0.8, 0.9, 0.99, 0.999]
    # error = [1e-1, 7.5e-2, 5e-2, 2.5e-2, 1e-2]
    error = [0.125, 0.1, 0.075, 0.05]
    # error = [0.15, 0.125, 0.1]
    labels = ["TREE", "NESTED-F", "NESTED-C", "DP-1.7", "DP-1.8"]
    labels = ["TREE", "NESTED-F", "NESTED-C"]

    # error = [1e-1, 1e-3, 1e-5]
    req_fids = [ 1 - n for n in error]
    tps = np.zeros((len(req_fids), len(labels)))
    times = np.zeros((len(req_fids), len(labels)))

    for i, fth in enumerate(req_fids):
        for j in range(exp_num):
            task = create_task(topology, gate, 
                node_memory, edge_capacity, edge_fidelity, 
                user_pair_num, path_num, 
                req_num_range, fth)

            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.TREE)
            times[i, 0] += time.time() - start
            tps[i, 0] += ObjVal


            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.NESTED_F)
            times[i, 1] += time.time() - start
            tps[i, 1] += ObjVal


            start = time.time()
            ObjVal = test_QuNetOptim(task, SolverType.NESTED_C)
            times[i, 2] += time.time() - start
            tps[i, 2] += ObjVal

            # start = time.time()
            # ObjVal = test_QuNetOptim(task, SolverType.DP, 1.7)
            # times[i, 3] += time.time() - start
            # tps[i, 3] += ObjVal

            # start = time.time()
            # ObjVal = test_QuNetOptim(task, SolverType.DP, 1.8)
            # times[i, 4] += time.time() - start
            # tps[i, 4] += ObjVal



        tps[i] /= exp_num
        times[i] /= exp_num

        print("fidelity {} done".format(fth))


    x = error
    ys = tps.T
    xlabel = "Infidelity Threshold"
    ylabel = "Throughput"
    markers = ['o', 's', 'v', 'x', 'd', 'p']
    filename = "../data/net/wn_net_tp_{}_{}.png".format(size, gate_desc)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, xreverse=True, filename=filename)

    ys = times.T
    ylabel = "Time (s)"
    filename = "../data/net/wn_net_time_{}_{}.png".format(size, gate_desc)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, xreverse=True, filename=filename)



if __name__ == '__main__':

    # avg_edge_num = 0
    # for i in range(100):
    #     topology = RandomGNP(50, 0.1)
    #     avg_edge_num += len(topology.edges)
    # print(avg_edge_num / 100)




    # test_dp_tp_by_fth('small', 1)
    # test_dp_tp_by_fth('medium', 1)
    # test_dp_tp_by_fth('large', 1)

    # test_wn_tp_by_fth('small', 100)
    # test_wn_tp_by_fth('medium', 100)
    # test_wn_tp_by_fth('large', 1)

    # test_wn_tp_by_methods('small', 1)
    # test_wn_tp_by_methods('medium', 1)
    # test_wn_tp_by_methods('large', 1)

    # test_wn_tp_by_methods('small', 40, qu.GWP)
    # test_wn_tp_by_methods('medium', 20, qu.GWP)
    # test_wn_tp_by_methods('large', 20, qu.GWP)


    # test_wn_tp_by_methods('small', 20, qu.GWH)
    # test_wn_tp_by_methods('medium', 20, qu.GWH)
    test_wn_tp_by_methods('large', 20, qu.GWH)

    # test_wn_tp_by_methods('small', 20, qu.GWM)
    # test_wn_tp_by_methods('medium', 10, qu.GWM)
    test_wn_tp_by_methods('large', 10, qu.GWM)

    # test_wn_tp_by_methods('small', 20, qu.GWL)
    # test_wn_tp_by_methods('medium', 20, qu.GWL)
    # test_wn_tp_by_methods('large', 20, qu.GWL)
