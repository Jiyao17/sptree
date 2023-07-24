
import time

import numpy as np
import matplotlib.pyplot as plt

import physical.quantum as qu
from sps.solver import test_TreeSolver, test_GRDSolver, test_EPPSolver
from utils.tools import test_edges_gen, draw_lines


# SEED = 0
def test_dp_sys(fth, exp_num):
    op = qu.GDP
    cost_cap = 10000
    edge_num_range = range(5, 26, 5)

    costs = np.zeros((len(edge_num_range), 3))
    times = np.zeros((len(edge_num_range), 3))
    for edge_num in edge_num_range:
        tree_cost, grdy_cost, epp_cost = 0, 0, 0
        tree_time, grdy_time, epp_time = 0, 0, 0

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, 0.7, 0.25)

            start = time.time()
            f, allocs = test_TreeSolver(edges, op, fth, cost_cap, False)
            tree_time += time.time() - start
            tree_cost += sum(allocs)

            start = time.time()
            f, allocs = test_GRDSolver(edges, op, fth, cost_cap, )
            grdy_time += time.time() - start
            grdy_cost += sum(allocs)

            start = time.time()
            f, allocs = test_EPPSolver(edges, op, fth, cost_cap, )
            epp_time += time.time() - start
            epp_cost += sum(allocs)

        tree_cost, grdy_cost, epp_cost = tree_cost / exp_num, grdy_cost / exp_num, epp_cost / exp_num
        tree_time, grdy_time, epp_time = tree_time / exp_num, grdy_time / exp_num, epp_time / exp_num

        costs[edge_num_range.index(edge_num)] = [tree_cost, grdy_cost, epp_cost]
        times[edge_num_range.index(edge_num)] = [tree_time, grdy_time, epp_time]

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    ys = [costs[:, 0], costs[:, 1], costs[:, 2]]
    labels = ["Tree", "GRDY", "EPP"]
    xlabel = "Number of edges"
    ylabel = "Cost"
    filename = "../data/dp_cost_f={}.png".format(fth)
    draw_lines(x, ys, labels, xlabel, ylabel, filename=filename)

    ys = [times[:, 0], times[:, 1], times[:, 2]]
    ylabel = "Time (s)"
    filename = "../data/dp_time_f={}.png".format(fth)
    draw_lines(x, ys, labels, xlabel, ylabel, filename=filename)

def comp_wn_sys(fth, exp_num):
    op = qu.GWP
    cost_cap = 10000
    edge_num_range = range(2, 16, 2)

    costs = np.zeros((len(edge_num_range), 3))
    times = np.zeros((len(edge_num_range), 3))
    for edge_num in edge_num_range:
        tree_cost, grdy_cost, epp_cost = 0, 0, 0
        tree_time, grdy_time, epp_time = 0, 0, 0

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, 0.99, 0.01)

            start = time.time()
            f, allocs = test_TreeSolver(edges, op, fth, cost_cap, False)
            tree_time += time.time() - start
            tree_cost += sum(allocs)

            start = time.time()
            f, allocs = test_GRDSolver(edges, op, fth, cost_cap, )
            grdy_time += time.time() - start
            grdy_cost += sum(allocs)

            start = time.time()
            f, allocs = test_EPPSolver(edges, op, fth, cost_cap, )
            epp_time += time.time() - start
            epp_cost += sum(allocs)

        tree_cost, grdy_cost, epp_cost = tree_cost / exp_num, grdy_cost / exp_num, epp_cost / exp_num
        tree_time, grdy_time, epp_time = tree_time / exp_num, grdy_time / exp_num, epp_time / exp_num

        costs[edge_num_range.index(edge_num)] = [tree_cost, grdy_cost, epp_cost]
        times[edge_num_range.index(edge_num)] = [tree_time, grdy_time, epp_time]

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    ys = [costs[:, 0], costs[:, 1], costs[:, 2]]
    labels = ["Tree", "GRDY", "EPP"]
    xlabel = "Number of edges"
    ylabel = "Cost"
    filename = "../data/wn_cost_f={}.png".format(fth)
    draw_lines(x, ys, labels, xlabel, ylabel, filename)

    ys = [times[:, 0], times[:, 1], times[:, 2]]
    ylabel = "Time (s)"
    title = "Time Comparison"
    filename = "../data/wn_time_f={}.png".format(fth)
    draw_lines(x, ys, labels, xlabel, ylabel, filename)



def test_wn(fth, gate, edge_num_range:list, fid_base, fid_range, exp_num, cost_cap):

    costs = np.zeros((len(edge_num_range), 1))
    times = np.zeros((len(edge_num_range), 1))
    for edge_num in edge_num_range:
        tree_cost = 0
        tree_time = 0

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, fid_base, fid_range)

            start = time.time()
            f, allocs = test_TreeSolver(edges, gate, fth, cost_cap, False)
            tree_time += time.time() - start
            tree_cost += sum(allocs)

        tree_cost = tree_cost / exp_num
        tree_time = tree_time / exp_num

        costs[edge_num_range.index(edge_num)] = [tree_cost, ]
        times[edge_num_range.index(edge_num)] = [tree_time, ]

        # print("edge num {} done".format(edge_num))

    y_cost = costs[:, 0]
    y_time = times[:, 0]
    return y_cost, y_time

def test_wn_sys(exp_num):
    gate = qu.GWP
    cost_cap = 10000
    edge_num_range = range(2, 16, 2)

    fid_lb = 0.95
    fid_range = 0.05

    fths = [0.7, 0.8, 0.9]
    y_costs = []
    y_times = []
    for fth in fths:
        y_cost, y_time = test_wn(fth, gate, edge_num_range, fid_lb, fid_range, exp_num, cost_cap)
        y_costs.append(y_cost)
        y_times.append(y_time)

        print("fth {} done".format(fth))

    x = edge_num_range
    ys = y_costs
    labels = ["f={}".format(fth) for fth in fths]
    xlabel = "Number of edges"
    ylabel = "Cost"
    filename = "../data/wn_cost.png"
    draw_lines(x, ys, labels, xlabel, ylabel, filename)

    ys = y_times
    ylabel = "Time (s)"
    filename = "../data/wn_time.png"
    draw_lines(x, ys, labels, xlabel, ylabel, filename)



if __name__ == '__main__':
    # test_dp_sys(0.8, 100)
    test_dp_sys(0.9, 100)
    test_dp_sys(0.99, 100)
    test_dp_sys(0.9999, 100)

    # test_wn_sys(5)
    # comp_wn_sys(0.8, 5)

