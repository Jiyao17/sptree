
import numpy as np

from physical.network import EdgeTuple
import physical.quantum as qu

class OSPSDP():
    # dynamic programming solver
    # for OSPS problem

    def __init__(self, 
            edges: dict[EdgeTuple, float],
            op: qu.Operation=qu.DOPP,
            budget: int = 100,
            ) -> None:
        self.edges = edges
        self.op = op
        self.budget = budget

        self.edge_num = len(edges)
        self.fids = list(edges.values())
        self.alloc = {}
    
    def solve(self, budget: int) -> float:
        # fidelity of optimal solutions
        self.mat = np.zeros((self.edge_num, self.edge_num+1, budget+1), dtype=np.float64)

        # init dp matrix
        for i in range(self.edge_num):
            self.mat[i][i+1][1] = self.fids[i]

        # the two outer loops iterate over all possible path fragments
        for length in range(2, self.edge_num + 1):
            for i in range(self.edge_num - length + 1):
                # solve path [i, j)
                j = i + length
                # got all subpaths of length len
                # now try all possible budget for each subpath
                for c in range(length, budget-(self.edge_num-length)+1):
                    max_fid = 0 
                    # max_fid_budget = 0
                    op = qu.OpType.SWAP

                    # if swap is the optimal choice at this point
                    # try all possible path split
                    for m in range(i+1, j):
                        # try all possible budget split
                        for bl in range(m-i, c-(j-m)+1):
                            for br in range(j-m, c-bl+1):
                                fid_left = self.mat[i][m][bl]
                                fid_right = self.mat[m][j][br]
                                if fid_left == 0 or fid_right == 0:
                                    # infeasible subpath
                                    continue
                                fid, prob = self.op.swap(fid_left, fid_right)
                                current_budget = np.ceil((bl + br) / prob)
                                if current_budget > c:
                                    # infeasible budget
                                    pass
                                elif fid > max_fid:
                                    op = qu.OpType.SWAP
                                    max_fid = fid
                                    # max_fid_budget = current_budget
                                    
                                    
                    # if purify is the optimal choice at this point
                    # try all possible budget split
                    for bl in range(length, c-length+1):
                        for br in range(length, c-bl+1):
                            if self.mat[i][j][bl] == 0 or self.mat[i][j][br] == 0:
                                # infeasible budget split
                                continue
                        fid, prob = self.op.purify(self.mat[i][j][bl], self.mat[i][j][br])
                        current_budget = np.ceil((bl + br) / prob)
                        if current_budget > c:
                            # infeasible budget
                            pass
                        elif fid > max_fid:
                            op = qu.OpType.PURIFY
                            max_fid = fid
                            # max_fid_budget = current_budget

                    self.mat[i][j][c] = max_fid

        
        f = self.mat[0][self.edge_num][budget]
        return f


def test_OSPS_DP() -> float:
    # np.random.seed(0)
    fids = np.random.random(100)
    fid_lower_bound = 0.95
    fid_range = 0.05
    fids = fids * fid_range + fid_lower_bound
    # fids = [0.9] * 10
    # print(fids)
    fth = 0.85
    edge_num = 5
    edges = {
        (i, i+1, 0): fids[i] for i in range(edge_num)
    }
    # print(fids[:edge_num])

    op = qu.WOPL
    budget = 50
    solver = OSPSDP(edges, op, budget)
    f = solver.solve(budget)

    return f
