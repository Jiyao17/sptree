

# an implementation of the Algo. 2 in paper:
# From Entanglement Purification Scheduling to  Fidelity-constrained Multi-Flow Routing


# takes the graph and SD pairs as input
# outputs the SPS for up to k paths between each SD pair

import numpy as np
import networkx as nx

from src.physical.quantum import GWH, GWM, OpType
from .spt import Node, Leaf, Branch
from .solver import TreeSolver, ExpAllocType, SPST

from src.physical.network import BufferedNode, Edge

def EPS(N, f_e, f_th, df=0.01, dk=0.01, round=False):
    """
    Algorithm 1: Entanglement Purification Scheduling
    df is the fidelity increment step
    dk is the percentage increment step
    """
    # minimal achievable fidelity with n leaves
    tao = np.ones(N+1)
    tao *= f_e

    leaf = Node((0, 1), f_e)
    L = [(1, f_e, 1, leaf)]

    for i in range(2, N + 1):
        for k in range(1, i//2 + 1):
            f, p = gate.purify(tao[k], tao[i-k])
            if tao[i] < f:
                tao[i] = f

    # find the smallest N_p such that tao[N_p] >= f_th
    N_p = np.where(tao >= f_th)[0][0]

    for i in range(1, min(N, 2*(N_p - 1)) + 1):
        # generate indices of all pairs of elements in L
        pairs = [(idx1, idx2) for idx1 in range(len(L)) for idx2 in range(idx1+1, len(L))]
        if len(pairs) == 0:
            pairs = [(0, 0)]
        for idx1, idx2 in pairs:
            b1, f1, k1, T1 = L[idx1]
            b2, f2, k2, T2 = L[idx2]

            b3 = b1 + b2
            if b3 <= min(N, 2*(N_p - 1)):
                f, p = gate.purify(f1, f2)
                if round:
                    f_hat = np.ceil(f / df) * df
                    k3_hat = np.ceil(p * min(k1, k2) / dk) * dk
                else:
                    f_hat = f
                    k3_hat = p * min(k1, k2)
                T3 = Branch((0, 1), f, None, T1, T2, op=OpType.PURIFY, prob=p)

                l3 = (b3, f_hat, k3_hat, T3)
                add_flag = True
                for l in L:
                    b, f, k, T = l
                    if b <= b3 and f >= f_hat and k >= k3_hat:
                        add_flag = False
                        break
                if add_flag:
                    L.append(l3)

    # filter L to only keep those with f >= f_th
    L = [l for l in L if l[1] >= f_th]
    # find the entry with max k/b in L
    idx = np.argmax([k/b for b, f, k, T in L])
    return L[idx]


def FCMCP(G_p, s_p, t_p, psi0, phi0):
    pass

def FCMCP_wrapper(G: nx.Graph, s, t, f_th):
    """
    G=(V,E)
    """
    vps = set(G.nodes()) - {s, t}
    Gp = nx.Graph()

    # each nodes in V - {s, t} is split Qv nodes
    vmap: 'dict[int, list[tuple[int, int]]]' = {}
    for v in vps:
        obj: BufferedNode = G.nodes[v]['obj']
        vmap[v] = []
        for i in range(1, obj.storage): # (1, Qv-1)
            Gp.add_node((v, i))
            vmap[v].append((v, i))
    # add s and t
    objs: BufferedNode = G.nodes[s]['obj']
    objt: BufferedNode = G.nodes[t]['obj']
    for i in range(1, objs.storage + 1):
        Gp.add_node((s, i))
    for i in range(0, objt.storage):
        Gp.add_node((t, i))
    # virtual s and t
    Gp.add_node((s, -1))
    Gp.add_node((t, -1))

    # add edges
    for u, v in G.edges():
        obj: Edge = G.edges[u, v]['obj']
        Qv = G.nodes[v]['obj'].storage
        cap = obj.capacity
        for ui, i in vmap[u]:
            for vj, j in vmap[v]:
                if Qv - j <= i and Qv - j <= cap:
                    Gp.add_edge(ui, vj, cost=Qv-j)
    
    # add edges from virtual s to (s, i)
    for i in range(1, objs.storage + 1):
        Gp.add_edge((s, -1), (s, i), cost=0, fid=1)
    # add edges from (t, i) to virtual t
    for i in range(0, objt.storage):
        Gp.add_edge((t, i), (t, -1), cost=0, fid=1)

def test_EPS():
    gate = GWH_D
    
    N = 10
    f_e = 0.9
    f_th = 0.95

    b, f, k, T = EPS(N, f_e, f_th)
    print(f"b: {b}, f: {f}, k: {k}")

    alloc = ExpAllocType({})
    TreeSolver._traverse(T, 1, alloc)
    
    print(alloc)

    SPST.print_tree(T)


if __name__ == "__main__":
    gate = GWM
    
    N = 27
    f_e = 0.85
    f_th = 0.95

    edges = {(0, 1): f_e,}
    tree = TreeSolver(edges, gate)
    tree.solve(fth=f_th, cost_cap=1000)
    result = tree.report()
    print(result)

    N = np.ceil(sum(result[1].values())).astype(int)

    N = np.ceil(N*1.5).astype(int)
    b, f, k, T = EPS(N, f_e, f_th)
    print(f"b: {b}, f: {f}, k: {k}")

    alloc = ExpAllocType({})
    TreeSolver._traverse(T, 1, alloc)
    
    print(alloc)

    SPST.print_tree(T)

    # result = gate.balanced_purify([f_e]*4)
    # print(result)

    # result = gate.seq_purify([f_e]*4)
    # print(result)

