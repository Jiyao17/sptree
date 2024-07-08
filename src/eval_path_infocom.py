
import time

import numpy as np
import matplotlib.pyplot as plt

import physical.quantum as qu
from sps.solver import test_TreeSolver, test_GRDSolver, test_EPPSolver, test_DPSolver, test_NestedSolver
from utils.tools import test_edges_gen, draw_lines, draw_2y_lines
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
def test_binary_all_time(fth, exp_num):
    op = qu.GDP
    cost_cap = 1e5
    edge_num_range = range(2, 26, 1)
    # edge_num_range = range(4, 11, 2)

    labels = ["TREE", "GRDY", "EPP", "NESTED-F", "NESTED-C", "DP-1.2", "DP-1.3"]
    # labels = ["TREE", "GRDY", "EPP" ]

    NESTED_EDGE_NUMS = edge_num_range
    
    dp_edge_limit = 6
    if fth <= 0.99:
        dp_edge_limit = 8
    if fth <= 0.9:
        dp_edge_limit = 11



    costs = np.zeros((len(edge_num_range), len(labels)))
    times = np.zeros((len(edge_num_range), len(labels)))
    fids = np.zeros((len(edge_num_range), len(labels)))
    for edge_num in edge_num_range:
        test_cost = np.zeros((len(labels)))
        test_time = np.zeros((len(labels)))
        test_fid = np.zeros((len(labels)))

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, (0.7, 0.95))

            start = time.time()
            f, allocs = test_TreeSolver(edges, op, fth, cost_cap, False)
            test_time[0] += time.time() - start
            test_cost[0] += sum(allocs)
            test_fid[0] += f
            tree_budget = np.ceil(sum(allocs)).astype(int)

            start = time.time()
            f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            test_time[1] += time.time() - start
            test_cost[1] += sum(allocs)
            test_fid[1] += f

            start = time.time()
            f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            test_time[2] += time.time() - start
            test_cost[2] += sum(allocs)
            test_fid[2] += f

            if edge_num in NESTED_EDGE_NUMS:
                start = time.time()
                f, allocs = test_NestedSolver(edges, op, budget=tree_budget, M_option='floor')
                test_time[3] += time.time() - start
                test_cost[3] += sum(allocs)
                test_fid[3] += f

                start = time.time()
                f, allocs = test_NestedSolver(edges, op, budget=tree_budget, M_option='ceil')
                test_time[4] += time.time() - start
                test_cost[4] += sum(allocs)
                test_fid[4] += f
            else:
                test_time[3] += np.nan
                test_cost[3] += np.nan
                test_fid[3] += np.nan

                test_time[4] += np.nan
                test_cost[4] += np.nan
                test_fid[4] += np.nan

            if  edge_num < dp_edge_limit:
                start = time.time()
                f, allocs = test_DPSolver(edges, op, int(tree_budget*1.2), eps=0)
                test_time[5] += time.time() - start
                test_cost[5] += sum(allocs)
                test_fid[5] += f

                start = time.time()
                f, allocs = test_DPSolver(edges, op, int(tree_budget*1.3), eps=0)
                test_time[6] += time.time() - start
                test_cost[6] += sum(allocs)
                test_fid[6] += f
            else:
                test_time[5] += np.nan
                test_cost[5] += np.nan
                test_fid[5] += np.nan

                test_time[6] += np.nan
                test_cost[6] += np.nan
                test_fid[6] += np.nan
                

            # start = time.time()
            # f, allocs = test_DPSolver(edges, op, int(dp_budget*1.5))
            # test_time[setting_idx] += time.time() - start
            # test_cost[setting_idx] += int(dp_budget*1.5)
            # test_fid[setting_idx] += f
            # setting_idx += 1

            # start = time.time()
            # f, allocs = test_DPSolver(edges, op, dp_budget*2)
            # test_time[setting_idx] += time.time() - start
            # test_cost[setting_idx] += dp_budget*2
            # test_fid[setting_idx] += f
            # setting_idx += 1
                                      


        costs[edge_num_range.index(edge_num)] = test_cost / exp_num
        times[edge_num_range.index(edge_num)] = test_time / exp_num
        fids[edge_num_range.index(edge_num)] = test_fid / exp_num

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    ys = costs.T
    xlabel = "Hop Number"
    ylabel = "Cost"
    markers = ['o', 's', 'v', 'd', 'p', 'x', 'h', '>', '<',]
    filename = "../data/path/dp_path_cost_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename)
    # draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, yscale='log')

    ys = times.T
    ylabel = "Time (s)"
    filename = "../data/path/dp_path_time_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, yscale='log')

    ys = fids.T
    ylabel = "Fidelity"
    filename = "../data/path/dp_path_fid_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, ylim=(0.5, 1))

def test_binary_cost_1(fth, exp_num):
    op = qu.GDP_LOSM
    cost_cap = 1e5
    edge_num_range = range(2, 16, 1)
    # edge_num_range = range(4, 11, 2)

    labels = ["TREE", "GRDY", "EPP" ]

    costs = np.zeros((len(edge_num_range), len(labels)))
    times = np.zeros((len(edge_num_range), len(labels)))
    fids = np.zeros((len(edge_num_range), len(labels)))
    for edge_num in edge_num_range:
        test_cost = np.zeros((len(labels)))
        test_time = np.zeros((len(labels)))
        test_fid = np.zeros((len(labels)))

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, (0.7, 0.95))

            start = time.time()
            f, allocs = test_TreeSolver(edges, op, fth, cost_cap, False)
            test_time[0] += time.time() - start
            test_cost[0] += sum(allocs)
            test_fid[0] += f
            tree_budget = np.ceil(sum(allocs)).astype(int)

            start = time.time()
            f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            test_time[1] += time.time() - start
            test_cost[1] += sum(allocs)
            test_fid[1] += f

            start = time.time()
            f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            test_time[2] += time.time() - start
            test_cost[2] += sum(allocs)
            test_fid[2] += f

        costs[edge_num_range.index(edge_num)] = test_cost / exp_num
        times[edge_num_range.index(edge_num)] = test_time / exp_num
        fids[edge_num_range.index(edge_num)] = test_fid / exp_num

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    ys = costs.T
    xlabel = "Hop Number"
    ylabel = "Cost"
    markers = ['o', 's', 'v', 'd', 'p', 'x', 'h', '>', '<']
    filename = "data/path/dp_path_cost_f={}_1.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, yscale='log')
    # draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename)

    ys = times.T
    ylabel = "Time (s)"
    filename = "data/path/dp_path_time_f={}_1.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, yscale='log')

    ys = fids.T
    ylabel = "Fidelity"
    filename = "data/path/dp_path_fid_f={}_1.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, ylim=(0.5, 1))

def test_binary_cost_all(fth, exp_num, gate=qu.GDP):
    if gate == qu.GDP:
        gate_desc = "P"
    elif gate == qu.GDP_LOSH:
        gate_desc = "H"
    elif gate == qu.GDP_LOSM:
        gate_desc = "M"
    elif gate == qu.GDP_LOSL:
        gate_desc = "L"

    cost_cap = 1e5
    edge_num_range = range(2, 26, 1)
    # edge_num_range = range(4, 11, 2)

    dp_edge_limit = 6
    if fth <= 0.99:
        dp_edge_limit = 8
    if fth <= 0.9:
        dp_edge_limit = 10

    # labels = ["TREE", "NESTED-F", "NESTED-C", "DP-1.2", "DP-1.3"]
    labels = ["TREE", "GRDY", "EPP", "DP-1.2", "DP-1.3", "NESTED-F", "NESTED-C", ]

    NESTED_EDGE_NUMS = edge_num_range

    costs = np.zeros((len(edge_num_range), len(labels)))
    times = np.zeros((len(edge_num_range), len(labels)))
    fids = np.zeros((len(edge_num_range), len(labels)))
    for edge_num in edge_num_range:
        test_cost = np.zeros((len(labels)))
        test_time = np.zeros((len(labels)))
        test_fid = np.zeros((len(labels)))

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, (0.7, 0.95))

            start = time.time()
            f, allocs = test_TreeSolver(edges, gate, fth, cost_cap, False)
            test_time[0] += time.time() - start
            test_cost[0] += sum(allocs)
            test_fid[0] += f
            tree_budget = np.ceil(sum(allocs)).astype(int)

            start = time.time()
            f, allocs = test_GRDSolver(edges, gate, fth, cost_cap,
                                    MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            test_time[1] += time.time() - start
            test_cost[1] += sum(allocs)
            test_fid[1] += f

            start = time.time()
            f, allocs = test_EPPSolver(edges, gate, fth, cost_cap,
                                    MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            test_time[2] += time.time() - start
            test_cost[2] += sum(allocs)
            test_fid[2] += f

            if  edge_num < dp_edge_limit:
                start = time.time()
                f, allocs = test_DPSolver(edges, gate, int(tree_budget*1.2), eps=0)
                test_time[3] += time.time() - start
                test_cost[3] += sum(allocs)
                test_fid[3] += f

                start = time.time()
                f, allocs = test_DPSolver(edges, gate, int(tree_budget*1.3), eps=0)
                test_time[4] += time.time() - start
                test_cost[4] += sum(allocs)
                test_fid[4] += f
            else:
                test_time[3] += np.nan
                test_cost[3] += np.nan
                test_fid[3] += np.nan

                test_time[4] += np.nan
                test_cost[4] += np.nan
                test_fid[4] += np.nan

            if edge_num in NESTED_EDGE_NUMS:
                start = time.time()
                f, allocs = test_NestedSolver(edges, gate, budget=tree_budget, M_option='floor')
                test_time[5] += time.time() - start
                test_cost[5] += sum(allocs)
                test_fid[5] += f

                start = time.time()
                f, allocs = test_NestedSolver(edges, gate, budget=tree_budget, M_option='ceil')
                test_time[6] += time.time() - start
                test_cost[6] += sum(allocs)
                test_fid[6] += f
            else:
                test_time[5] += np.nan
                test_cost[5] += np.nan
                test_fid[5] += np.nan

                test_time[6] += np.nan
                test_cost[6] += np.nan
                test_fid[6] += np.nan


                                      


        costs[edge_num_range.index(edge_num)] = test_cost / exp_num
        times[edge_num_range.index(edge_num)] = test_time / exp_num
        fids[edge_num_range.index(edge_num)] = test_fid / exp_num

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    ys = costs.T
    xlabel = "Hop Number"
    ylabel = "Cost"
    markers = ['o', 's', 'v', 'd', 'p', 'x', 'h', '>', '<']
    filename = "data/path/dp_path_cost_f={}_noise={}.png".format(fth, gate_desc)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, yscale='log')

    ys1 = costs[:, :5].T
    ys2 = costs[:, 5:].T
    filename = "data/path/dp_path_cost_f={}_noise={}_2y.png".format(fth, gate_desc)
    draw_2y_lines(x, ys1, ys2, xlabel, 
        y1_label=ylabel, y2_label=ylabel,
        line1_labels=labels[:5], line2_labels=labels[5:],
        line1_markers=markers[:5], line2_markers=markers[5:],
        xscale='linear', y1_scale='linear', y2_scale='log',
        xreverse=False, y1_reverse=False, y2_reverse=False,
        xlim=None, y1_lim=None, y2_lim=None,
        filename=filename,
        )

    ys = times.T
    ylabel = "Time (s)"
    filename = "data/path/dp_path_time_f={}_noise={}.png".format(fth, gate_desc)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, yscale='log')

    ys = fids.T
    ylabel = "Fidelity"
    filename = "data/path/dp_path_fid_f={}_noise={}.png".format(fth, gate_desc)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, ylim=(0.5, 1))

def test_wn_cost_1(fth, exp_num):
    op = qu.GWH
    cost_cap = 1e5
    edge_num_range = range(2, 17, 1)
    # edge_num_range = range(4, 11, 2)

    labels = ["TREE", "GRDY", "EPP", "NESTED_F", "NESTED_C", "DP"]

    NESTED_EDGE_NUMS = edge_num_range


    costs = np.zeros((len(edge_num_range), len(labels)))
    times = np.zeros((len(edge_num_range), len(labels)))
    fids = np.zeros((len(edge_num_range), len(labels)))
    for edge_num in edge_num_range:
        test_cost = np.zeros((len(labels)))
        test_time = np.zeros((len(labels)))
        test_fid = np.zeros((len(labels)))

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, (0.95, 1))

            start = time.time()
            f, allocs = test_TreeSolver(edges, op, fth, cost_cap, False)
            test_time[0] += time.time() - start
            test_cost[0] += sum(allocs)
            test_fid[0] += f
            tree_budget = np.ceil(sum(allocs)).astype(int)

            # if edge_num <= 2:
            #     start = time.time()
            #     f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
            #                             MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            #     test_time[1] += time.time() - start
            #     test_cost[1] += sum(allocs)
            #     test_fid[1] += f
            # else:
            #     test_time[1] += np.nan
            #     test_cost[1] += np.nan
            #     test_fid[1] += np.nan

            # if edge_num <= 5:
            #     start = time.time()
            #     f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
            #                             MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            #     test_time[2] += time.time() - start
            #     test_cost[2] += sum(allocs)
            #     test_fid[2] += f
            # else:
            #     test_time[2] += np.nan
            #     test_cost[2] += np.nan
            #     test_fid[2] += np.nan


            if edge_num in NESTED_EDGE_NUMS:
                start = time.time()
                f, allocs = test_NestedSolver(edges, op, budget=tree_budget, M_option='floor')
                test_time[3] += time.time() - start
                test_cost[3] += sum(allocs)
                test_fid[3] += f

                start = time.time()
                f, allocs = test_NestedSolver(edges, op, budget=tree_budget, M_option='ceil')
                test_time[4] += time.time() - start
                test_cost[4] += sum(allocs)
                test_fid[4] += f
            else:
                test_time[3] += np.nan
                test_cost[3] += np.nan
                test_fid[3] += np.nan

                test_time[4] += np.nan
                test_cost[4] += np.nan
                test_fid[4] += np.nan

            if  edge_num <= 6:
                start = time.time()
                f, allocs = test_DPSolver(edges, op, tree_budget)
                test_time[5] += time.time() - start
                test_cost[5] += tree_budget
                test_fid[5] += f
            else:
                test_time[5] += np.nan
                test_cost[5] += np.nan
                test_fid[5] += np.nan

            # start = time.time()
            # f, allocs = test_DPSolver(edges, op, int(dp_budget*1.5))
            # test_time[setting_idx] += time.time() - start
            # test_cost[setting_idx] += int(dp_budget*1.5)
            # test_fid[setting_idx] += f
            # setting_idx += 1

            # start = time.time()
            # f, allocs = test_DPSolver(edges, op, dp_budget*2)
            # test_time[setting_idx] += time.time() - start
            # test_cost[setting_idx] += dp_budget*2
            # test_fid[setting_idx] += f
            # setting_idx += 1
                                      


        costs[edge_num_range.index(edge_num)] = test_cost / exp_num
        times[edge_num_range.index(edge_num)] = test_time / exp_num
        fids[edge_num_range.index(edge_num)] = test_fid / exp_num

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    ys = costs.T
    xlabel = "Hop Number"
    ylabel = "Cost"
    markers = ['o', 's', 'v', 'd', 'p', 'x', 'h', '>', '<']
    filename = "../data/path/wn_path_cost_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, yscale='log')

    ys = times.T
    ylabel = "Time (s)"
    filename = "../data/path/wn_path_time_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename)

    ys = fids.T
    ylabel = "Fidelity"
    filename = "../data/path/wn_path_fid_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, ylim=(0.5, 1))

def test_wn_cost_1(fth, exp_num):
    op = qu.GWH
    cost_cap = 1e5
    edge_num_range = range(2, 17, 1)
    # edge_num_range = range(4, 11, 2)

    labels = ["TREE", "GRDY", "EPP", "NESTED", "DP-1", "DP-2", "DP-3"]
    labels = ["TREE", "GRDY", "EPP", "NESTED_F", "NESTED_C", "DP"]
    labels = ["TREE", "NESTED_F", "NESTED_C", "DP"]

    NESTED_EDGE_NUMS = edge_num_range


    costs = np.zeros((len(edge_num_range), len(labels)))
    times = np.zeros((len(edge_num_range), len(labels)))
    fids = np.zeros((len(edge_num_range), len(labels)))
    for edge_num in edge_num_range:
        test_cost = np.zeros((len(labels)))
        test_time = np.zeros((len(labels)))
        test_fid = np.zeros((len(labels)))

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, (0.95, 1))

            start = time.time()
            f, allocs = test_TreeSolver(edges, op, fth, cost_cap, False)
            test_time[0] += time.time() - start
            test_cost[0] += sum(allocs)
            test_fid[0] += f
            tree_budget = np.ceil(sum(allocs)).astype(int)

            # if edge_num <= 2:
            #     start = time.time()
            #     f, allocs = test_GRDSolver(edges, op, fth, cost_cap,
            #                             MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            #     test_time[1] += time.time() - start
            #     test_cost[1] += sum(allocs)
            #     test_fid[1] += f
            # else:
            #     test_time[1] += np.nan
            #     test_cost[1] += np.nan
            #     test_fid[1] += np.nan

            # if edge_num <= 5:
            #     start = time.time()
            #     f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
            #                             MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            #     test_time[2] += time.time() - start
            #     test_cost[2] += sum(allocs)
            #     test_fid[2] += f
            # else:
            #     test_time[2] += np.nan
            #     test_cost[2] += np.nan
            #     test_fid[2] += np.nan


            if edge_num in NESTED_EDGE_NUMS:
                start = time.time()
                f, allocs = test_NestedSolver(edges, op, budget=tree_budget, M_option='floor')
                test_time[3] += time.time() - start
                test_cost[3] += sum(allocs)
                test_fid[3] += f

                start = time.time()
                f, allocs = test_NestedSolver(edges, op, budget=tree_budget, M_option='ceil')
                test_time[4] += time.time() - start
                test_cost[4] += sum(allocs)
                test_fid[4] += f
            else:
                test_time[3] += np.nan
                test_cost[3] += np.nan
                test_fid[3] += np.nan

                test_time[4] += np.nan
                test_cost[4] += np.nan
                test_fid[4] += np.nan

            if  edge_num <= 6:
                start = time.time()
                f, allocs = test_DPSolver(edges, op, tree_budget)
                test_time[5] += time.time() - start
                test_cost[5] += tree_budget
                test_fid[5] += f
            else:
                test_time[5] += np.nan
                test_cost[5] += np.nan
                test_fid[5] += np.nan

            # start = time.time()
            # f, allocs = test_DPSolver(edges, op, int(dp_budget*1.5))
            # test_time[setting_idx] += time.time() - start
            # test_cost[setting_idx] += int(dp_budget*1.5)
            # test_fid[setting_idx] += f
            # setting_idx += 1

            # start = time.time()
            # f, allocs = test_DPSolver(edges, op, dp_budget*2)
            # test_time[setting_idx] += time.time() - start
            # test_cost[setting_idx] += dp_budget*2
            # test_fid[setting_idx] += f
            # setting_idx += 1
                                      


        costs[edge_num_range.index(edge_num)] = test_cost / exp_num
        times[edge_num_range.index(edge_num)] = test_time / exp_num
        fids[edge_num_range.index(edge_num)] = test_fid / exp_num

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    ys = costs.T
    xlabel = "Hop Number"
    ylabel = "Cost"
    markers = ['o', 's', 'v', 'd', 'p', 'x', 'h', '>', '<']
    filename = "../data/path/wn_path_cost_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, yscale='log')

    ys = times.T
    ylabel = "Time (s)"
    filename = "../data/path/wn_path_time_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename)

    ys = fids.T
    ylabel = "Fidelity"
    filename = "../data/path/wn_path_fid_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, ylim=(0.5, 1))

def test_wn_sys_full_2(fth, exp_num):
    op = qu.GWP
    cost_cap = 1e5
    edge_num_range = range(2, 15, 1)
    # edge_num_range = range(4, 11, 2)
    dp_edge_limit = 7
    if fth <= 0.9:
        dp_edge_limit = 9
    if fth <= 0.85:
        dp_edge_limit = 11

    labels = ["TREE", "GRDY", "EPP", "NESTED", "DP-1", "DP-2", "DP-3"]
    labels = ["TREE", "GRDY", "EPP", "NESTED_F", "NESTED_C", "DP"]
    labels = ["TREE", "NESTED_F", "NESTED_C", "DP-1.3", "DP-1.4"]

    NESTED_EDGE_NUMS = edge_num_range


    costs = np.zeros((len(edge_num_range), len(labels)))
    times = np.zeros((len(edge_num_range), len(labels)))
    fids = np.zeros((len(edge_num_range), len(labels)))
    for edge_num in edge_num_range:
        test_cost = np.zeros((len(labels)))
        test_time = np.zeros((len(labels)))
        test_fid = np.zeros((len(labels)))

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, (0.95, 1))

            start = time.time()
            f, allocs = test_TreeSolver(edges, op, fth, cost_cap, False)
            test_time[0] += time.time() - start
            test_cost[0] += sum(allocs)
            test_fid[0] += f
            tree_budget = np.ceil(sum(allocs)).astype(int)

            if edge_num <= 30:
                if edge_num in NESTED_EDGE_NUMS:
                    start = time.time()
                    f, allocs = test_NestedSolver(edges, op, budget=tree_budget, M_option='floor')
                    test_time[1] += time.time() - start
                    test_cost[1] += sum(allocs)
                    test_fid[1] += f

                    start = time.time()
                    f, allocs = test_NestedSolver(edges, op, budget=tree_budget, M_option='ceil')
                    test_time[2] += time.time() - start
                    test_cost[2] += sum(allocs)
                    test_fid[2] += f
                else:
                    test_time[1] += np.nan
                    test_cost[1] += np.nan
                    test_fid[1] += np.nan

                    test_time[2] += np.nan
                    test_cost[2] += np.nan
                    test_fid[2] += np.nan
            else:
                test_time[1] += np.nan
                test_cost[1] += np.nan
                test_fid[1] += np.nan

                test_time[2] += np.nan
                test_cost[2] += np.nan
                test_fid[2] += np.nan

            if  edge_num < dp_edge_limit:
                start = time.time()
                f, allocs = test_DPSolver(edges, op, int(tree_budget*1.3), eps=0)
                test_time[3] += time.time() - start
                test_cost[3] += sum(allocs)
                test_fid[3] += f

                start = time.time()
                f, allocs = test_DPSolver(edges, op, int(tree_budget*1.4), eps=0)
                test_time[4] += time.time() - start
                test_cost[4] += sum(allocs)
                test_fid[4] += f
            else:
                test_time[3] += np.nan
                test_cost[3] += np.nan
                test_fid[3] += np.nan

                test_time[4] += np.nan
                test_cost[4] += np.nan
                test_fid[4] += np.nan

            # start = time.time()
            # f, allocs = test_DPSolver(edges, op, int(dp_budget*1.5))
            # test_time[setting_idx] += time.time() - start
            # test_cost[setting_idx] += int(dp_budget*1.5)
            # test_fid[setting_idx] += f
            # setting_idx += 1

            # start = time.time()
            # f, allocs = test_DPSolver(edges, op, dp_budget*2)
            # test_time[setting_idx] += time.time() - start
            # test_cost[setting_idx] += dp_budget*2
            # test_fid[setting_idx] += f
            # setting_idx += 1
                                      


        costs[edge_num_range.index(edge_num)] = test_cost / exp_num
        times[edge_num_range.index(edge_num)] = test_time / exp_num
        fids[edge_num_range.index(edge_num)] = test_fid / exp_num

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    ys = costs.T
    xlabel = "Hop Number"
    ylabel = "Cost"
    markers = ['o', 's', 'v', 'd', 'p', 'x', 'h', '>', '<']
    filename = "../data/path/wn_path_cost_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, yscale='log')

    ys = times.T
    ylabel = "Time (s)"
    filename = "../data/path/wn_path_time_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename)

    ys = fids.T
    ylabel = "Fidelity"
    filename = "../data/path/wn_path_fid_f={}.png".format(fth)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, ylim=(0.5, 1))

def test_wn_sys_full_3(fth, exp_num, gate=qu.GWH):
    if gate == qu.GWP:
        gate_desc = "P"
    elif gate == qu.GWH:
        gate_desc = "H"
    elif gate == qu.GWM:
        gate_desc = "M"
    elif gate == qu.GWL:
        gate_desc = "L"



    cost_cap = 1e5
    edge_num_range = range(2, 15, 1)
    # edge_num_range = range(4, 11, 2)
    dp_edge_limit = 7
    if fth <= 0.9:
        dp_edge_limit = 9
    if fth <= 0.85:
        dp_edge_limit = 11

    labels = ["TREE", "GRDY", "EPP", "NESTED", "DP-1", "DP-2", "DP-3"]
    labels = ["TREE", "GRDY", "EPP", "NESTED_F", "NESTED_C", "DP"]
    labels = ["TREE", "NESTED_F", "NESTED_C", "DP-1.8", "DP-1.9"]
    labels = ["TREE", "NESTED-F", "NESTED-C"]
    NESTED_EDGE_NUMS = edge_num_range


    costs = np.zeros((len(edge_num_range), len(labels)))
    times = np.zeros((len(edge_num_range), len(labels)))
    fids = np.zeros((len(edge_num_range), len(labels)))
    for edge_num in edge_num_range:
        test_cost = np.zeros((len(labels)))
        test_time = np.zeros((len(labels)))
        test_fid = np.zeros((len(labels)))

        for i in range(exp_num):
            edges = test_edges_gen(edge_num, (0.95, 1))

            start = time.time()
            f, allocs = test_TreeSolver(edges, gate, fth, cost_cap, False)
            test_time[0] += time.time() - start
            test_cost[0] += sum(allocs)
            test_fid[0] += f
            tree_budget = np.ceil(sum(allocs)).astype(int)

            if edge_num <= 30:
                if edge_num in NESTED_EDGE_NUMS:
                    start = time.time()
                    f, allocs = test_NestedSolver(edges, gate, budget=tree_budget, M_option='floor')
                    test_time[1] += time.time() - start
                    test_cost[1] += sum(allocs)
                    test_fid[1] += f

                    start = time.time()
                    f, allocs = test_NestedSolver(edges, gate, budget=tree_budget, M_option='ceil')
                    test_time[2] += time.time() - start
                    test_cost[2] += sum(allocs)
                    test_fid[2] += f
                else:
                    test_time[1] += np.nan
                    test_cost[1] += np.nan
                    test_fid[1] += np.nan

                    test_time[2] += np.nan
                    test_cost[2] += np.nan
                    test_fid[2] += np.nan
            else:
                test_time[1] += np.nan
                test_cost[1] += np.nan
                test_fid[1] += np.nan

                test_time[2] += np.nan
                test_cost[2] += np.nan
                test_fid[2] += np.nan

            # if  edge_num < dp_edge_limit:
            #     start = time.time()
            #     f, allocs = test_DPSolver(edges, gate, int(tree_budget*1.7), eps=0)
            #     test_time[3] += time.time() - start
            #     test_cost[3] += sum(allocs)
            #     test_fid[3] += f

            #     start = time.time()
            #     f, allocs = test_DPSolver(edges, gate, int(tree_budget*1.8), eps=0)
            #     test_time[4] += time.time() - start
            #     test_cost[4] += sum(allocs)
            #     test_fid[4] += f
            # else:
            #     test_time[3] += np.nan
            #     test_cost[3] += np.nan
            #     test_fid[3] += np.nan

            #     test_time[4] += np.nan
            #     test_cost[4] += np.nan
            #     test_fid[4] += np.nan

            # start = time.time()
            # f, allocs = test_DPSolver(edges, op, int(dp_budget*1.5))
            # test_time[setting_idx] += time.time() - start
            # test_cost[setting_idx] += int(dp_budget*1.5)
            # test_fid[setting_idx] += f
            # setting_idx += 1

            # start = time.time()
            # f, allocs = test_DPSolver(edges, op, dp_budget*2)
            # test_time[setting_idx] += time.time() - start
            # test_cost[setting_idx] += dp_budget*2
            # test_fid[setting_idx] += f
            # setting_idx += 1
                                      


        costs[edge_num_range.index(edge_num)] = test_cost / exp_num
        times[edge_num_range.index(edge_num)] = test_time / exp_num
        fids[edge_num_range.index(edge_num)] = test_fid / exp_num

        print("edge num {} done".format(edge_num))

    x = edge_num_range
    ys = costs.T
    xlabel = "Hop Number"
    ylabel = "Cost"
    markers = ['o', 's', 'v', 'd', 'p', 'x', 'h', '>', '<']
    filename = "../data/path/wn_path_cost_f={}_noise={}.png".format(fth, gate_desc)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, yscale='log')

    ys = times.T
    ylabel = "Time (s)"
    filename = "../data/path/wn_path_time_f={}_noise={}.png".format(fth, gate_desc)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename)

    ys = fids.T
    ylabel = "Fidelity"
    filename = "../data/path/wn_path_fid_f={}noise={}.png".format(fth, gate_desc)
    draw_lines(x, ys, xlabel, ylabel, labels, markers, filename=filename, ylim=(0.5, 1))

def test_wn_sys_full_old(fth, exp_num):
    op = qu.GWP
    cost_cap = 1e5
    edge_num_range = range(2, 11, 1)
    # edge_num_range = range(7, 10, 1)

    labels = ["TREE", "GRDY", "EPP", "NESTED", "DP-1"]


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

            start = time.time()
            f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
                                    MetaTree.Shape.LINKED, MetaTree.Shape.LINKED)
            edge_time[5] += time.time() - start
            edge_cost[5] += sum(allocs)

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

            # start = time.time()
            # f, allocs = test_EPPSolver(edges, op, fth, cost_cap,
            #                         MetaTree.Shape.BALANCED, MetaTree.Shape.BALANCED)
            # edge_time[8] += time.time() - start
            # edge_cost[8] += sum(allocs)


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
    edge_num_range = range(3, 4, 1)


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
    # fth = 0.7
    fid_range = (fth, fth)

    gate = qu.GWP

    edges = test_edges_gen(2, fid_range)

    # f, allocs = test_GRDSolver(edges, gate, fth, cost_cap)
    f, allocs = test_EPPSolver(edges, gate, fth, cost_cap)

    print(f, sum(allocs))

                

if __name__ == '__main__':
    # test_dp_sys(0.8, 10)

    # test_binary_all_time(0.9, 10)
    # test_binary_all_time(0.99, 10)
    # test_binary_all_time(0.9999, 10)

    # test_binary_cost_1(0.9, 20)
    # test_binary_cost_1(0.99, 20)
    # test_binary_cost_1(0.9999, 20)

    test_binary_cost_all(0.9, 10)
    test_binary_cost_all(0.99, 10)
    test_binary_cost_all(0.9999, 10)


    # test_wn_sys_full(0.8, 10)
    # test_wn_sys_full_2(0.85, 20)
    # test_wn_sys_full_2(0.9, 20)
    # test_wn_sys_full_2(0.925, 20)
    # test_wn_sys_full_2(0.95, 20)
    # test_wn_sys_full_2(0.975, 10)
    # test_wn_sys_full_2(0.99, 10)

    # test_wn_sys_full_3(0.9, 20, qu.GWH)
    # test_wn_sys_full_3(0.925, 20, qu.GWH)
    # test_wn_sys_full_3(0.95, 20, qu.GWH)

    # test_wn_sys_full_3(0.9, 20, qu.GWM)
    # test_wn_sys_full_3(0.925, 20, qu.GWM)
    # test_wn_sys_full_3(0.95, 20, qu.GWM)

    # test_wn_sys_full_3(0.9, 20, qu.GWL)
    # test_wn_sys_full_3(0.925, 20, qu.GWL)
    # test_wn_sys_full_3(0.95, 20, qu.GWL)


    # import random
    # fth = random.uniform(0.7, 1)
    # print(fth)
    # test_wn_others(fth, 1)

    # test_wn_sys(0.85, 100)
    # test_wn_sys(0.9, 100)
    # test_wn_sys(0.95, 100)

