
import numpy as np
import matplotlib.pyplot as plt


def test_edges_gen(edge_num, fid_lb, fid_range):
    # np.random.seed(0)
    fids = np.random.random(edge_num)
    fids = fids * fid_range + fid_lb
    
    edges = {
        (i, i+1): fids[i] for i in range(edge_num)
    }

    return edges

def draw_lines(x, ys, labels, xlabel, ylabel, filename):
    plt.figure()
    plt.rc('font', size=20)
    plt.subplots_adjust(0.18, 0.16, 0.95, 0.96)

    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    # plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    
