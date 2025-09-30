

# an implementation of the Algo. 2 in paper:
# From Entanglement Purification Scheduling to  Fidelity-constrained Multi-Flow Routing


# takes the graph and SD pairs as input
# outputs the SPS for up to k paths between each SD pair

import heapq
from collections import defaultdict

import numpy as np
import networkx as nx

from src.physical.quantum import GWP, GWH, GWM, OpType
from .spt import Node, Leaf, Branch
from .solver import TreeSolver, ExpAllocType, SPST

from src.physical.network import BufferedNode, Edge

def EPS(N, f_e, f_th, gate=GWP, df=0.01, dk=0.01, round=False):
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


def is_dominating(l1, l2):
    """
    check if l1 dominates l2
    return:
        2: strictly dominates
        1: dominates
        0: not dominates
    """
    c1, phi_hat1, psi_b1, psi_hatl1, path1 = l1
    c2, phi_hat2, psi_b2, psi_hatl2, path2 = l2
    if c1 <= c2 and phi_hat1 >= phi_hat2 and psi_hatl1 >= psi_hatl2:
        if c1 < c2 or phi_hat1 > phi_hat2 or psi_hatl1 > psi_hatl2:
            return 2
        else:
            return 1
    else:
        return 0

def FCMCP(G: nx.Graph, G_p: nx.Graph, s_p, t_p, psi0, phi0, Rk= 5, dphi=0.01, dpsi=0.01):
    
    Tao = []
    L = defaultdict(list)
    Omega = defaultdict(list)
    for v in G_p.nodes():
        if v == s_p:
            l = (0, 0, float('inf'), float('inf'), [s_p[0], ])
            # l = (0, 0, float('inf'), float('inf'), [s_p[0], ])

            L[v] = [l, ]
            heapq.heappush(Tao, (0, l, v))
        else:
            L[v] = []
        Omega[v] = []

    while len(Tao) > 0:
        c, l, ui = heapq.heappop(Tao)
        if l not in Omega[ui]:
            Omega[ui].append(l)
            if ui == t_p:
                break
            c, phi_hat, psi_b, psi_hat, path = l
            for vj in G_p.neighbors(ui):
                u, i = ui
                v, j = vj
                Qv = G.nodes[v]['obj'].storage
                if Qv - j <= 0:
                    continue
                # print(u, v)
                f_e = G_p.edges[ui, vj]['fid']
                if v not in path or len(path) == 1:
                # if v not in path:
                    ub = np.floor((phi_hat - phi0) / dphi).astype(int)
                    for k in range(1, ub + 1):
                        phi_hp = phi_hat - k * dphi
                        # f_e = G_p.edges[ui, vj]['fid']
                        f_th = np.pow(np.e, (phi_hp*3 + 1)) / 4
                        if f_th <= f_e:
                            b, f, xi, T = 1, f_e, 1.0, Node((u, v), f_e)
                        else:
                            b, f, xi, T = EPS(Qv - j, f_e, f_th)
                        # print(xi, b, Qv, j)
                        psi_e = np.log((xi/b) * (Qv - j))
                        psi_vj = np.log(G.nodes[v]['obj'].swap_prob)
                        # if psi_hat == float('inf'):
                        #     psi_hat = psi_e
                        # if psi_b == float('inf'):
                        #     psi_b = psi_e
                        if psi_e <= psi_b:
                            temp = (psi_vj + psi_hat + psi_e - psi_b) / dpsi
                            psi_hp = np.ceil(temp).astype(int) * dpsi
                        else:
                            temp = (psi_vj + psi_hat) / dpsi
                            psi_hp = np.ceil(temp).astype(int) * dpsi

                        if psi_hp is np.nan:
                            psi_hp = float('inf')

                        if psi_hp >= psi0:
                            # Qj = G.nodes[j]['obj'].storage
                            Qj = Qv
                            l_p = (c + Qj-j, phi_hp, min(psi_b, psi_e), psi_hp, path + [v, ])

                            ls = L[vj]
                            n_dominating = 0
                            for l2 in ls:
                                if is_dominating(l2, l_p) > 0:
                                    n_dominating += 1
                            if n_dominating < Rk:
                                L[vj].append(l_p)
                                Omega[ui].append(l_p)
                                heapq.heappush(Tao, (c + Qj-j, l_p, vj))

    if len(L[t_p]) == 0:
        return []
    else:
        # sort L[t_p] by cost and return the top Rk entries
        L_t = list(L[t_p])
        L_t.sort(key=lambda x: x[0])
        return L_t[:Rk]

def FCMCP_wrapper(G: nx.Graph, s, t, f_th):
    """
    G=(V,E)
    """
    vps = set(G.nodes()) - {s, t}
    # undirected graph 
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
    vmap[s] = []
    vmap[t] = []
    for i in range(1, objs.storage + 1):
        Gp.add_node((s, i))
        vmap[s].append((s, i))
    for i in range(0, objt.storage):
        Gp.add_node((t, i))
        vmap[t].append((t, i))

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
                    Gp.add_edge((ui, i), (vj, j), cost=Qv-j, fid=obj.fid)
    
    # add edges from virtual s to (s, i)
    for i in range(1, objs.storage + 1):
        Gp.add_edge((s, -1), (s, i), cost=0, fid=1)
    # add edges from (t, i) to virtual t
    for i in range(0, objt.storage):
        Gp.add_edge((t, i), (t, -1), cost=0, fid=1)

    return Gp

def test_EPS():
    gate = GWP
    
    N = 10
    f_e = 0.9
    f_th = 0.95

    b, f, k, T = EPS(N, f_e, f_th)
    print(f"b: {b}, f: {f}, k: {k}")

    alloc = ExpAllocType({})
    TreeSolver._traverse(T, 1, alloc)
    
    print(alloc)

    SPST.print_tree(T)

def test_FCMCP():
    import matplotlib.pyplot as plt
    G = nx.Graph()
    nodes = [0, 1, 2, 3]
    storage = [2, 3, 3, 2]
    for n, s in zip(nodes, storage):
        G.add_node(n, obj=BufferedNode(n, 1, storage=s))
    edges = [ (0, 1, (0.98, 10)), (1, 2, (0.96, 10)), (2, 3, (0.99, 10)), ]
    for u, v, (fid, cap) in edges:
        G.add_edge(u, v, obj=Edge(u, v, fid, cap))

    # nodes = [0, 1]
    # storage = [5] * 3
    # for n, s in zip(nodes, storage):
    #     G.add_node(n, obj=BufferedNode(n, 1, storage=s))
    # edges = [ (0, 1, (0.93, 10)), ]
    # for u, v, (fid, cap) in edges:
    #     G.add_edge(u, v, obj=Edge(u, v, fid, cap))
    # draw and save G to file
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.savefig("G.png")
    plt.close()

    s = 0
    t = 3
    f_th = 0.6
    tp = 1
    Gp = FCMCP_wrapper(G, s, t, f_th)
    # draw and save Gp to file
    pos = nx.spring_layout(Gp)
    nx.draw(Gp, pos, with_labels=True)
    plt.savefig("Gp.png")
    plt.close()
    psi0 = np.log(tp)
    phi0 = np.log((4*f_th - 1) / 3)
    paths = FCMCP(G, Gp, (s, -1), (t, -1), psi0=psi0, phi0=phi0, Rk=5)
    
    print(paths)


if __name__ == "__main__":
    # test_EPS()
    test_FCMCP()