
import numpy as np
import matplotlib.pyplot as plt


def test_edges_gen(edge_num, fid_range):
    # np.random.seed(0)
    fids = np.random.uniform(fid_range[0], fid_range[1], edge_num)
    
    edges = {
        (i, i+1): fids[i] for i in range(edge_num)
    }

    return edges

def draw_lines(
    x, ys,
    xlabel, ylabel,
    labels, markers,
    xscale='linear', yscale='linear',
    xreverse=False, yreverse=False,
    xlim=None, ylim=None,
    filename='pic.png',
    ):
    plt.figure()
    plt.rc('font', size=20)
    plt.subplots_adjust(0.18, 0.16, 0.95, 0.96)
    
    for y, label, marker in zip(ys, labels, markers):
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

    

def draw_bars(
    x, ys,
    labels, xlabel, ylabel,
    xscale='linear', yscale='linear',
    xreverse=False, yreverse=False,
    filename='pic.png'):
    plt.figure()
    plt.rc('font', size=20)
    plt.subplots_adjust(0.18, 0.16, 0.95, 0.96)

    for y, label in zip(ys, labels):
        # bar 
        plt.bar(x, y, label=label)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if xreverse:
        plt.gca().invert_xaxis()
    if yreverse:
        plt.gca().invert_yaxis()
    # plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    
