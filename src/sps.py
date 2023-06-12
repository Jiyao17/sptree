
from copy import deepcopy

import numpy as np

from network import QuNet, QuNetTask, StaticPath, EdgeTuple
from quantum import Operation, EntType, MeasureAccu


class GRDYSolver():
    # greedy solver
    def __init__(self, 
            edges: dict[EdgeTuple, float],
            op: Operation = Operation()
            ) -> None:
        self.edges = deepcopy(edges)
        self.op = op
        
        self.edge_num = len(edges)
        self.fids = list(edges.values())
        self.alloc = { edge_tuple: 0 for edge_tuple in edges.keys() }

    def swap_purify(self, fids, allocs) -> float:

        pfids = np.zeros(len(fids))
        for i in range(self.edge_num):
            pfids[i] = self.op.seq_purify([fids[i]] * allocs[i])

        f = self.op.seq_swap(pfids)

        return f
    
    def solve(self, fth: int) -> 'tuple[float, list]':
        allocs = [1] * self.edge_num
        f = self.swap_purify(self.fids, allocs)
        while f < fth:
            fm = 0
            em = 0
            for i in range(self.edge_num):
                allocs[i] += 1
                f_new = self.swap_purify(self.fids, allocs)
                if f_new > fm:
                    fm = f_new
                    em = i

                allocs[i] -= 1

            allocs[em] += 1
            f = fm

        return f, allocs


def test_GRDYSolver(edges: dict[EdgeTuple, float], fth: float=0.9, ):
    solver = GRDYSolver(edges, Operation())
    f, allocs = solver.solve(fth=fth)
    # print(f)
    # print(allocs, sum(allocs))

    return f, sum(allocs)
