
from abc import ABC, abstractmethod
from typing import NewType
from copy import deepcopy
from enum import Enum

import numpy as np

from physical.network import EdgeTuple
import physical.quantum as qu
from .spt import SPST, Node, Branch


AllocType = NewType('AllocType', dict[EdgeTuple, qu.BudgetType])
ExpAllocType = NewType('ExpAllocType', dict[EdgeTuple, qu.ExpCostType])

class SolverType(Enum):
    TREE = 1
    GRD = 2
    EPP = 3

class Solver(ABC):
    def __init__(self, edges: 'dict[EdgeTuple, qu.FidType]', gate: qu.Gate) -> None:
        self.edges = edges
        self.gate = gate

        self.edge_num = len(edges)
        self.fids = list(edges.values())
        self.alloc: ExpAllocType = {}

    @abstractmethod
    def solve(self, fid_th: qu.FidType) -> 'tuple[qu.FidType, ExpAllocType]':
        pass

    @abstractmethod
    def report(self) -> None:
        """
        report the final fidelity and allocation
        """
        pass


class TreeSolver(Solver):

    @staticmethod
    def _traverse(node: Node, cost_factor: 'qu.ExpCostType', alloc: ExpAllocType):
        if node.is_leaf():
            if node.edge_tuple not in alloc:
                alloc[node.edge_tuple] = node.cost * cost_factor
            else:
                alloc[node.edge_tuple] += node.cost * cost_factor
        else:
            assert isinstance(node, Branch)
            cost_factor /= node.prob
            TreeSolver._traverse(node.left, cost_factor, alloc)
            TreeSolver._traverse(node.right, cost_factor, alloc)

    def __init__(self, edges: dict[EdgeTuple, qu.FidType], gate: qu.Gate) -> None:
        super().__init__(edges, gate)

        self.tree = SPST(edges, gate)
        self.tree.build_sst()

    def solve(self, fth: qu.FidType=0.90, cost_cap=0, attr: str='adjust_eff'):
        """
        Parameters
        ----------
        attr : str, attribute used to rank candidate nodes for purifications
            - 'adjust_eff' : adjusted efficiency
        """

        while self.tree.root.fid < fth and self.tree.root.cost < cost_cap:
            self.tree.grad(self.tree.root)
            self.tree.calc_efficiency(self.tree.root)
            node = SPST.find_max(self.tree.root, attr=attr)
            # calculate the cost of purifying the node
            pf, prob = self.gate.purify(node.fid, node.fid)
            dc = node.cost*node.grad_cn + (pf-node.fid)*node.grad_cf
            if self.tree.root.cost + dc > cost_cap:
                # infeasible budget
                break
            # purify the node
            node = self.tree.purify(node)
            # update fidelity of all ancestors
            self.tree.backward(node)

        return self.tree.root

    def report(self) -> 'tuple[qu.FidType, ExpAllocType]':
        
        alloc = ExpAllocType({})
        TreeSolver._traverse(self.tree.root, 1, alloc)

        return self.tree.root.fid, alloc


class DPSolver(Solver):
    # dynamic programming solver
    # for OSPS problem

    def __init__(self, 
            edges: dict[EdgeTuple, float],
            gate: qu.Gate=qu.GDP,
            ) -> None:
        super().__init__(edges, gate)

    
    def solve(self, budget: int):
        # fidelity of optimal solutions
        self.mat = np.zeros((self.edge_num, self.edge_num+1, budget+1), dtype=np.float64)

        # init dp matrix
        for i in range(self.edge_num):
            # edge i = [i, i+1]
            self.mat[i][i+1][1] = self.fids[i]

        # the two outer loops iterate over all possible path fragments
        for length in range(2, self.edge_num + 1):
            for i in range(self.edge_num - length + 1):
                # solve path [i, j]
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
                                # max between mat[i][m][m-i] and mat[i][m][br]
                                fid_left = self.mat[i][m][bl]
                                fid_right = self.mat[m][j][br]
                                if fid_left == 0 or fid_right == 0:
                                    # infeasible subpath
                                    continue
                                fid, prob = self.gate.swap(fid_left, fid_right)
                                expected_budget = (bl + br) / prob
                                if expected_budget > c:
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
                            fid, prob = self.gate.purify(self.mat[i][j][bl], self.mat[i][j][br])
                            expected_budget = (bl + br) / prob
                            if expected_budget > c:
                                # infeasible budget
                                pass
                            elif fid > max_fid:
                                op = qu.OpType.PURIFY
                                max_fid = fid
                                # print(f"purify: {i}, {j}, {bl}, {br}")
                                # max_fid_budget = current_budget

                    self.mat[i][j][c] = max_fid

    def report(self) -> 'tuple[qu.FidType, ExpAllocType]':
        # find the optimal allocation
        # from the dp matrix
        # self.alloc = {}
        # self._report(self.edge_num, self.budget)
        
        
        f = self.mat[0][self.edge_num][-1]
        return f, self.alloc


class GRDSolver(Solver):
    def __init__(self, edges: dict[EdgeTuple, qu.FidType], gate: qu.Gate) -> None:
        super().__init__(edges, gate)

    def purify_on_edge(self, fids: list[qu.FidType], allocs: list[int]) -> 'list[qu.FidType]':
        pfids = []
        for i in range(len(fids)):
            pfids.append(self.gate.seq_purify([fids[i]] * allocs[i]))
        return pfids
    
    def swap_purify(self, fids: list[qu.FidType], allocs: list[int]) -> qu.FidType:
        pfids = self.purify_on_edge(fids, allocs)
        return self.gate.seq_swap(pfids)

    def solve(self, fth: qu.FidType, cost_cap: qu.ExpCostType):
        """
        cost_cap is only used to stop the algorithm
        when it is unable to reach fth
        so it does not matter if it is not the accurate expected cost
        """
        # init allocation
        for edge in self.edges:
            self.alloc[edge] = 1
        pfids = self.purify_on_edge(self.fids, list(self.alloc.values()))
        f = self.gate.seq_swap(pfids)
        while f < fth and sum(self.alloc.values()) <= cost_cap:
            f_max = 0
            edge_max = 0
            for edge in self.edges:
                self.alloc[edge] += 1
                pfids = self.purify_on_edge(self.fids, list(self.alloc.values()))
                f_new = self.gate.seq_swap(pfids)
                if f_new > f_max:
                    f_max = f_new
                    edge_max = edge

                self.alloc[edge] -= 1

            self.alloc[edge_max] += 1
            f = f_max
    
    def report(self) -> 'tuple[qu.FidType, ExpAllocType]':
        # introduce probability of success
        psts: 'list[Node]' = []
        for i, edge in enumerate(self.edges):
            num = int(np.ceil(self.alloc[edge]))
            pst = SPST.build_pst(self.gate, edge, self.fids[i], num,)
            psts.append(pst)

        pfids = [ pst.fid for pst in psts ]
        pedges = { edge: pfids[i] for i, edge in enumerate(self.edges) }
        tree = SPST(pedges, self.gate)
        costs = [ pst.cost for pst in psts ]
        tree.build_sst(costs=costs)

        alloc = ExpAllocType({})
        TreeSolver._traverse(tree.root, 1, alloc)

        # pfids = self.purify_on_edge(self.fids, list(self.alloc.values()))
        # pedges = { edge: pfids[i] for i, edge in enumerate(self.edges) }
        # tree = SPST(pedges, self.gate)
        # tree.build_sst(costs=list(self.alloc.values()))

        # alloc = ExpAllocType({})
        # TreeSolver._traverse(tree.root, 1, alloc)
        return tree.root.fid, alloc


class EPPSolver(Solver):
    def __init__(self, edges: dict[EdgeTuple, qu.FidType], gate: qu.Gate) -> None:
        super().__init__(edges, gate)

    def purify_on_edge(self, fids: list[qu.FidType], allocs: list[int]) -> 'list[qu.FidType]':
        pfids = []
        for i in range(len(fids)):
            pfids.append(self.gate.seq_purify([fids[i]] * allocs[i]))
        return pfids
    
    def swap_purify(self, fids: list[qu.FidType], allocs: list[int]) -> qu.FidType:
        pfids = self.purify_on_edge(fids, allocs)
        return self.gate.seq_swap(pfids)

    def find_critical_edge(self, fids: list[qu.FidType]) -> 'int':
        # find the edge with the highest gradient
        grads = [0] * len(fids)
        for i in range(len(fids)):
            grads[i] = self.gate.seq_swap_grad(fids, i)
        
        max_grad_idx = np.argmax(grads)
        return max_grad_idx

    def solve(self, fth: qu.FidType, cost_cap: qu.ExpCostType) -> Node:
        """
        cost_cap is only used to stop the algorithm
        when it is unable to reach fth
        so it does not matter if it is not the accurate expected cost
        """
        # allocate purification to edges
        N = len(self.edges) - 1
        # Equ. 11
        min_alloc = ((3*N-1)**2 + 1) / (3*N)**2 if N > 0 else 1
        min_alloc = int(np.ceil(min_alloc))
        self.alloc = { edge: min_alloc for edge in self.edges }

        fid = self.swap_purify(self.fids, list(self.alloc.values()))

        while fid < fth and sum(self.alloc.values()) <= cost_cap:
            # find the edge with the highest gradient
            pfids = self.purify_on_edge(self.fids, list(self.alloc.values()))
            max_grad_idx = self.find_critical_edge(pfids)
            edge = list(self.edges.keys())[max_grad_idx]
            self.alloc[edge] += 1
            fid = self.swap_purify(self.fids, list(self.alloc.values()))

    
    def report(self) -> 'tuple[qu.FidType, ExpAllocType]':
        # introduce probability of success
        psts: 'list[Node]' = []
        for i, edge in enumerate(self.edges):
            num = int(np.ceil(self.alloc[edge]))
            pst = SPST.build_pst(self.gate, edge, self.fids[i], num,)
            psts.append(pst)

        pfids = [ pst.fid for pst in psts ]
        pedges = { edge: pfids[i] for i, edge in enumerate(self.edges) }
        tree = SPST(pedges, self.gate)
        costs = [ pst.cost for pst in psts ]
        tree.build_sst(costs=costs)

        alloc = ExpAllocType({})
        TreeSolver._traverse(tree.root, 1, alloc)

        return tree.root.fid, alloc





def test_DPSolver(edges, op, budget) -> float:
    solver = DPSolver(edges, op)
    solver.solve(budget)
    f, alloc = solver.report()

    return f, list(alloc.values())


def test_TreeSolver(edges, gate, fth=0.9, cost_cap=1000, print_tree=False):

    solver = TreeSolver(edges, gate)
    solver.solve(fth, cost_cap)
    f, alloc = solver.report()

    # print(f, solver.tree.root.cost)
    # print(sum(alloc.values()), list(alloc.values()))
    if print_tree:
        SPST.print_tree(solver.tree.root)

    return f, list(alloc.values())
    

def test_GRDSolver(edges, op, fth=0.9, cost_cap=1000):
    solver = GRDSolver(edges, op)
    solver.solve(fth, cost_cap)
    f, alloc = solver.report()

    return f, list(alloc.values())


def test_EPPSolver(edges, op, fth=0.9, cost_cap=1000):
    solver = EPPSolver(edges, op)
    solver.solve(fth, cost_cap)
    f, alloc = solver.report()

    return f, list(alloc.values())
