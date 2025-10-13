
import time

import numpy as np
import matplotlib.pyplot as plt

import physical.quantum as qu
from sps.solver import test_TreeSolver, test_GRDSolver, test_EPPSolver, test_DPSolver
from utils.tools import test_edges_gen, draw_lines
from sps.spt import MetaTree

def draw_times(
    x, ys,
    xlabel, ylabel,
    labels:list, markers,
    xscale='linear', yscale='linear',
    xreverse=False, yreverse=False,
    xlim=None, ylim=None,
    filename='pic.png',
    stops=None,
    ):
    plt.figure()
    plt.rc('font', size=20)
    plt.subplots_adjust(0.18, 0.16, 0.95, 0.96)
    
    for y, label, marker in zip(ys, labels, markers):
        if stops is not None:
            stop = stops[labels.index(label)]
            x = x[:stop]
            y = y[:stop]
        # find first y > ylim[1]
        if ylim is not None:
            idx = np.where(y > ylim[1])[0]
            if len(idx) > 0:
                y = y[:idx[0]]
                x = x[:idx[0]]
        plt.plot(x, y, label=label, marker=marker, markerfacecolor='none', markersize=10)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if xreverse:
        plt.gca().invert_xaxis()
    if yreverse:
        plt.gca().invert_yaxis()
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    # plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)

    


# SEED = 0
def test_dp_sys(fth, exp_num):
    op = qu.GDP
    cost_cap = 1e5
    edge_num_range = range(5, 26, 5)
    # edge_num_range = range(7, 10, 1)

    labels = ["TREE",
        "GRDY-LL", "GRDY-LB", "GRDY-BL", "GRDY-BB",
        "EPP-LL", "EPP-LB", "EPP-BL", "EPP-BB",
        ]


    costs = np.zeros((len(edge_num_range), len(labels)))
    times = np.zeros((len(edge_num_range), len(labels)))
    for edge_num in edge_num_range:
        edge_cost = np.zeros((len(labels)))
        edge_time = np.zeros((len(labels)))

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, (0.7, 0.95))

            start = time.time()
            f, allocs = test_TreeSolver(edges, op, fth, cost_cap, False)
            edge_time[0] += time.time() - start
            edge_cost[0] += sum(allocs)

            start = time.time()
            f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            edge_time[1] += time.time() - start
            edge_cost[1] += sum(allocs)

            # start = time.time()
            # f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
            #                         MetaTree.Shape.LINKED, MetaTree.Shape.BALANCED)
            # edge_time[2] += time.time() - start
            # edge_cost[2] += sum(allocs)

            start = time.time()
            f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.BALANCED, MetaTree.Shape.LINKED)
            edge_time[3] += time.time() - start
            edge_cost[3] += sum(allocs)

            start = time.time()
            f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.BALANCED, MetaTree.Shape.BALANCED)
            edge_time[4] += time.time() - start
            edge_cost[4] += sum(allocs)

            start = time.time()
            f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            edge_time[5] += time.time() - start
            edge_cost[5] += sum(allocs)

            start = time.time()
            f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.LINKED, MetaTree.Shape.BALANCED)
            edge_time[6] += time.time() - start
            edge_cost[6] += sum(allocs)

            start = time.time()
            f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.BALANCED, MetaTree.Shape.LINKED)
            edge_time[7] += time.time() - start
            edge_cost[7] += sum(allocs)

            start = time.time()
            f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.BALANCED, MetaTree.Shape.BALANCED)
            edge_time[8] += time.time() - start
            edge_cost[8] += sum(allocs)


        costs[edge_num_range.index(edge_num)] = edge_cost / exp_num
        times[edge_num_range.index(edge_num)] = edge_time / exp_num

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    ys = costs.T
    xlabel = "Hop Number"
    ylabel = "Cost"
    markers = ['o', 's', 'v', 'd', 'p', 'x', 'h', '>', '<']
    filename = "../data/path/dp_path_cost_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename)

    ys = times.T
    ylabel = "Time (s)"
    filename = "../data/path/dp_path_time_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename)

def test_wn_sys_full(fth, exp_num):
    op = qu.GWP
    cost_cap = 1e5
    edge_num_range = range(2, 11, 2)
    # edge_num_range = range(7, 10, 1)

    labels = ["TREE",
        "GRDY-LL", "GRDY-LB", "GRDY-BL", "GRDY-BB",
        "EPP-LL", "EPP-LB", "EPP-BL", "EPP-BB"
        ]


    costs = np.zeros((len(edge_num_range), len(labels)))
    times = np.zeros((len(edge_num_range), len(labels)))
    for edge_num in edge_num_range:
        edge_cost = np.zeros((len(labels)))
        edge_time = np.zeros((len(labels)))

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, (0.95, 1))

            start = time.time()
            f, allocs = test_TreeSolver(edges, op, fth, cost_cap, False)
            edge_time[0] += time.time() - start
            edge_cost[0] += sum(allocs)

            # start = time.time()
            # f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
            #                         MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            # edge_time[1] += time.time() - start
            # edge_cost[1] += sum(allocs)

            # start = time.time()
            # f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
            #                         MetaTree.Shape.LINKED, MetaTree.Shape.BALANCED)
            # edge_time[2] += time.time() - start
            # edge_cost[2] += sum(allocs)

            # start = time.time()
            # f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
            #                         MetaTree.Shape.BALANCED, MetaTree.Shape.LINKED)
            # edge_time[3] += time.time() - start
            # edge_cost[3] += sum(allocs)

            # start = time.time()
            # f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
            #                         MetaTree.Shape.BALANCED, MetaTree.Shape.BALANCED)
            # edge_time[4] += time.time() - start
            # edge_cost[4] += sum(allocs)

            # start = time.time()
            # f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
            #                         MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            # edge_time[5] += time.time() - start
            # edge_cost[5] += sum(allocs)

            # start = time.time()
            # f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
            #                         MetaTree.Shape.LINKED, MetaTree.Shape.BALANCED)
            # edge_time[6] += time.time() - start
            # edge_cost[6] += sum(allocs)

            # start = time.time()
            # f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
            #                         MetaTree.Shape.BALANCED, MetaTree.Shape.LINKED)
            # edge_time[7] += time.time() - start
            # edge_cost[7] += sum(allocs)

            start = time.time()
            f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.BALANCED, MetaTree.Shape.BALANCED)
            edge_time[8] += time.time() - start
            edge_cost[8] += sum(allocs)


        costs[edge_num_range.index(edge_num)] = edge_cost / exp_num
        times[edge_num_range.index(edge_num)] = edge_time / exp_num

        print("edge num {} done".format(edge_num))

        x = edge_num_range
        ys = costs.T
        xlabel = "Hop Number"
        ylabel = "Cost"
        markers = ['o', 's', 'v', 'd', 'p', 'x', 'h', '>', '<']
        filename = "/home/ljy/projects/sptree/data/path/wn_path_cost_f={}.png".format(fth)
        draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename)

        ys = times.T
        ylabel = "Time (s)"
        filename = "/home/ljy/projects/sptree/data/path/wn_path_time_f={}.png".format(fth)
        draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename)

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

def test_wn(fth, gate, edge_num_range:list, fid_range, exp_num, cost_cap):

    costs = np.zeros((len(edge_num_range)))
    times = np.zeros((len(edge_num_range)))
    for i, edge_num in enumerate(edge_num_range):
        tree_cost = 0
        tree_time = 0

        for j in range(exp_num):
            edges = test_edges_gen(edge_num, fid_range)

            start = time.time()
            f, allocs = test_TreeSolver(edges, gate, fth, cost_cap, False)
            tree_time += time.time() - start
            tree_cost += sum(allocs)

        tree_cost = tree_cost / exp_num
        tree_time = tree_time / exp_num

        costs[i] = [tree_cost, ]
        times[i] = [tree_time, ]

        # print("edge num {} done".format(edge_num))

    y_cost = costs[:, 0]
    y_time = times[:, 0]
    return y_cost, y_time

def test_wn_sys(fth, exp_num):
    # gate = qu.GWP
    cost_cap = 1e4
    edge_num_range = range(2, 11, 1)


    fid_range = (0.95, 1)

    gates = [qu.GWP, qu.GWH, qu.GWM, qu.GWL]
    # gates = [qu.GWP, qu.GWH, qu.GWM, ]
    y_costs = np.zeros((len(edge_num_range), len(gates)))
    y_times = np.zeros((len(edge_num_range), len(gates)))

    for i, edge_num in enumerate(edge_num_range):
        edges = test_edges_gen(edge_num, fid_range)

        for _ in range(exp_num):
            for j, gate in enumerate(gates):
                start = time.time()
                f, allocs = test_TreeSolver(edges, gate, fth, cost_cap, False)
                y_costs[i, j] += sum(allocs)
                y_times[i, j] += time.time() - start
                
        y_costs[i] = y_costs[i] / exp_num
        y_times[i] = y_times[i] / exp_num

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    # ys = [y_costs[:, 0], y_costs[:, 1], y_costs[:, 2], y_costs[:, 3]]
    ys = y_costs.T
    labels = ["P", "H", "M", "L"]
    xlabel = "Hop Number"
    ylabel = "Cost"
    markers = ['o', 's', 'v', 'x']
    filename = "../data/path/wn_path_cost_f={}.png".format(fth)
    # if (fth < 0.8):
    #     draw_lines(x, ys, labels, xlabel, ylabel, filename=filename)
    # else:
    draw_lines(x, ys, xlabel, ylabel, labels, markers,
        xlim=None, ylim=(0, 1e3),
        yscale='log', filename=filename,
        )

    # ys = [y_times[:, 0], y_times[:, 1], y_times[:, 2], y_times[:, 3]]
    ys = y_times.T
    ylabel = "Time (s)"
    filename = "../data/path/wn_path_time_f={}.png".format(fth)
    # if (fth < 0.8):
    #     draw_lines(x, ys, labels, xlabel, ylabel, filename=filename)
    # else:
    stops = []
    for y in y_costs.T:
        idx = np.where(y > 1e3)[0]
        if len(idx) > 0:
            stops.append(idx[0])
        else:
            stops.append(len(y))
    draw_times(x, ys, xlabel, ylabel, labels, markers,
        ylim=(0, 1),
        yscale='log', filename=filename, stops=stops,
        )

def test_wn_others(fth, exp_num):
    # gate = qu.GWP
    cost_cap = 1e10
    fth = 0.7
    fid_range = (fth, fth)

    gate = qu.GWP

    edges = test_edges_gen(3, fid_range)

    f, allocs = test_GRDSolver(edges, gate, fth, cost_cap)
    # f, allocs = test_EPPSolver(edges, gate, fth, cost_cap)

    print(f, sum(allocs))

                

if __name__ == '__main__':
    # test_dp_sys(0.8, 10)
    # test_dp_sys(0.9, 10)
    test_dp_sys(0.99, 10)
    # test_dp_sys(0.9999, 10)

    # test_wn_sys_full(0.8, 10)
    # test_wn_sys_full(0.85, 10)
    # test_wn_sys_full(0.9, 10)
    # test_wn_sys_full(0.99, 10)

    # test_wn_others(0.99, 1)

    # test_wn_sys(0.85, 100)
    # test_wn_sys(0.9, 100)
    # test_wn_sys(0.95, 100)

