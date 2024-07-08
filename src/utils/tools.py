
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
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    # plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)

    plt.close()


def draw_2y_lines(
    x, ys1, ys2,
    xlabel, y1_label, y2_label,
    line1_labels, line2_labels,
    line1_markers, line2_markers,
    xscale='linear', y1_scale='linear', y2_scale='linear',
    xreverse=False, y1_reverse=False, y2_reverse=False,
    xlim=None, y1_lim=None, y2_lim=None,
    filename='pic.png',
    ):
    # two y-axis in one figure
    fig, ax1 = plt.subplots()


    plt.rc('font', size=16)
    plt.subplots_adjust(0.18, 0.16, 0.96, 0.96)

    ax2 = ax1.twinx()

    for y1, label, marker in zip(ys1, line1_labels, line1_markers):
        # find first y > ylim[1]
        if y1_lim is not None:
            idx = np.where(y1 > y1_lim[1])[0]
            if len(idx) > 0:
                y1 = y1[:idx[0]]
                x = x[:idx[0]]
        ax1.plot(x, y1, label=label, marker=marker, markerfacecolor='none', markersize=10)
    
    for y2, label, marker in zip(ys2, line2_labels, line2_markers):
        # find first y > ylim[1]
        if y2_lim is not None:
            idx = np.where(y2 > y2_lim[1])[0]
            if len(idx) > 0:
                y2 = y2[:idx[0]]
                x = x[:idx[0]]
        ax2.plot(x, y2, label=label, marker=marker, markerfacecolor='none', markersize=10)
    
    ax1.set_xlabel(xlabel, fontsize=20)
    ax1.set_ylabel(y1_label, fontsize=20)
    ax2.set_ylabel(y2_label, fontsize=20)
    ax1.set_xscale(xscale)
    ax1.set_yscale(y1_scale)
    ax2.set_yscale(y2_scale)
    if xreverse:
        ax1.invert_xaxis()
    if y1_reverse:
        ax1.invert_yaxis()
    if y2_reverse:
        ax2.invert_yaxis()
    if xlim is not None:
        ax1.set_xlim(*xlim)
    if y1_lim is not None:
        ax1.set_ylim(*y1_lim)
    if y2_lim is not None:
        ax2.set_ylim(*y2_lim)
    # plt.title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True)

    fig.tight_layout()

    plt.savefig(filename)

    plt.close()



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
    
