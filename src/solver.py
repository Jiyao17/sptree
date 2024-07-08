
import copy
from abc import ABC, abstractmethod
import time

import numpy as np

import gurobipy as gp
from gurobipy import GRB

from physical.network import QuNet, QuNetTask, Edge, EdgeTuple, NodeID, NodePair, StaticPath, BufferedNode
import physical.quantum as qu
from sps.solver import test_TreeSolver, test_GRDSolver, test_EPPSolver, test_DPSolver, test_NestedSolver, SolverType
from physical.topology import ATT



class ObjType:
    FEASIBILITY = 0
    MAX_THROUGHPUT = 1
    MIN_LATENCY = 2


class Optimizer(ABC):

    def __init__(self, task: QuNetTask) -> None:
        self.task = copy.deepcopy(task)
        # dump all parameters needed by optimizer to a dict
        self.params = {}
        self.net = copy.deepcopy(task.net)

        self.model = gp.Model("LP")

    @abstractmethod
    def import_params(self, ) -> dict:
        pass

    @abstractmethod
    def add_constrs(self, cs: str='') -> None:
        pass
    
    @abstractmethod
    def optimize(self, objective: ObjType=ObjType.FEASIBILITY) -> float:
        pass


class QuNetOptim:
    tree_budge: dict = {}
    def __init__(self, task: QuNetTask,) -> None:
        self.task = copy.deepcopy(task)
        self.qunet = copy.deepcopy(task.qunet)
        
        self.params = {}
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        self.model = gp.Model("QuNetOptim", env=env)

    def import_params(self) -> None:
        self.params = {}

        # all edges, [EdgeTuple]
        self.params['E']: 'list[EdgeTuple]' = []
        # capacity of each edge, {e: capacity}
        self.params['EC']: 'dict[EdgeTuple, qu.ExpCostType]' = {}
        for (u, v, obj) in self.task.qunet.net.edges.data('obj'):
            edge_tuple: EdgeTuple = (u, v)
            self.params['E'].append(edge_tuple)
            self.params['EC'][edge_tuple] = obj.capacity
        self.E = self.params['E']
        self.EC = self.params['EC']
        # all nodes in the network, [NodeID]
        self.params['N']: 'list[NodeID]' = copy.deepcopy(self.task.qunet.net.nodes)
        # node capacity , {node: capacity}
        self.params['NC']: 'dict[NodeID, int]' = {}
        for node in self.task.qunet.net.nodes:
            node_obj: BufferedNode = self.task.qunet.net.nodes[node]['obj']
            self.params['NC'][node] = node_obj.storage
        self.N = self.params['N']
        self.NC = self.params['NC']
        # user pairs, [(src, dst)]
        self.params['K']: 'list[NodePair]' = copy.deepcopy(self.task.user_pairs)
        self.K = self.params['K']
        # paths between user pairs, {k: [path]}
        self.params['P']: 'dict[NodePair, list[StaticPath]]' = copy.deepcopy(self.task.up_paths)
        self.P = self.params['P']

        # workload of user pair, {(k, t): load}
        self.params['W']: 'dict[NodePair, int]' = copy.deepcopy(self.task.workload)
        self.W = self.params['W']
        # fidelity threshold of user pair, {(k, t): fidelity}
        self.params['F']: 'dict[NodePair, float]' = copy.deepcopy(self.task.fid_req)
        self.F = self.params['F']

        # path length
        self.params['Pl']: 'dict[StaticPath, int]' = {}
        for np, paths in self.P.items():
            for path in paths:
                self.params['Pl'][path] = len(path)
        self.Pl = self.params['Pl']

        # purification allocations
        # indices = []
        self.params['x']: 'dict[tuple[NodePair, StaticPath], GRB.INTEGER]' = {}
        self.x = self.params['x']
        for k in self.K:
            for p in self.P[k]:
                    self.x[k, p] = self.model.addVar(vtype=GRB.INTEGER, name=f"x_{k}_{p}")



        return self.params

    def path_prep(self, solver_type: SolverType, arg=None) -> 'tuple[int, int]':
        # path allocations
        self.params['A']: 'dict[tuple[NodePair, StaticPath, EdgeTuple], qu.ExpCostType]' = {}
        self.A = self.params['A']

        gate = self.qunet.gate

        total_pair_num = 0
        total_path_num = 0
        for k in self.K:
            for p in self.P[k]:
                edges: 'dict[EdgeTuple, qu.FidType]' = {}
                cost_cap = 0
                for e in p:
                    edge: Edge = self.task.qunet.net.edges[e]['obj']
                    edges[e] = edge.fid
                    cost_cap += edge.capacity
                
                # print(f"fid: {fids}")
                fth = self.F[k]
                if solver_type == SolverType.TREE:
                    f, allocs = test_TreeSolver(edges, gate, fth, cost_cap)
                    QuNetOptim.tree_budge[(k, p)] = sum(allocs)
                elif solver_type == SolverType.GRD:
                    f, allocs = test_GRDSolver(edges, gate, fth, cost_cap,
                                )
                elif solver_type == SolverType.EPP:
                    f, allocs = test_EPPSolver(edges, gate, fth, cost_cap)
                elif solver_type == SolverType.DP:
                    f, allocs = test_DPSolver(edges, gate, int(QuNetOptim.tree_budge[(k, p)]*arg))
                    if f < fth:
                        # not qualified, disable this path
                        allocs = [1000000] * len(allocs)
                elif solver_type == SolverType.NESTED_F:
                    f, allocs = test_NestedSolver(edges, gate, int(QuNetOptim.tree_budge[(k, p)])+1, 'floor')
                    if f < fth:
                        allocs = [1000000] * len(allocs)
                elif solver_type == SolverType.NESTED_C:
                    f, allocs = test_NestedSolver(edges, gate, int(QuNetOptim.tree_budge[(k, p)])+1, 'ceil')
                    if f < fth:
                        allocs = [1000000] * len(allocs)
                for i, e in enumerate(p):
                    self.A[k, p, e] = allocs[i]

                total_pair_num += sum(allocs)
                total_path_num += 1

        return total_pair_num, total_path_num

    def add_constrs(self, cs: str='') -> None:
        # edge capacity
        for k in self.K:
            for path in self.P[k]:
                for i, e in enumerate(path):
                    if e in self.EC:
                        edge_capacity = self.EC[e]
                    else:
                        edge_capacity = self.EC[(e[1], e[0])]
                    self.model.addConstr(
                        gp.quicksum(
                            self.x[k, path] * self.A[k, path, e] for e in path) 
                            <= edge_capacity,
                        name=f"EC_{k}_{path}_{i}")
                    
        # memory capacity
        for node in self.N:
            mem_usage = 0
            adj_edges = self.task.qunet.net.edges(node, data=False)
            for k in self.K:
                for path in self.P[k]:
                    for i, e in enumerate(path):
                        if e in adj_edges or (e[1], e[0]) in adj_edges:
                            mem_usage += self.x[k, path] * self.A[k, path, e]
            self.model.addConstr(
                mem_usage <= self.NC[node],
                name=f"NC_{node}")
            
        # request fulfillment
        for k in self.K:
            for path in self.P[k]:
                self.model.addConstr(
                    gp.quicksum(self.x[k, path] for path in self.P[k]) 
                        <= self.W[k],
                    name=f"RF_{k}_{path}")

    def optimize(self, objective: ObjType=ObjType.MAX_THROUGHPUT) -> float:
        if objective == ObjType.FEASIBILITY:
            pass
        elif objective == ObjType.MAX_THROUGHPUT:
            tp = 0
            for k in self.K:
                for path in self.P[k]:
                    tp += self.x[k, path]
            self.model.setObjective(tp, GRB.MAXIMIZE)
        elif objective == ObjType.MIN_LATENCY:
            pass
        else:
            raise ValueError(f"objective type {objective} not supported")

        self.model.optimize()
        return self.model.objVal
    

def test_QuNetOptim(
        task: QuNetTask,
        solver_type=SolverType.TREE, 
        arg=None,
        ):


    optm = QuNetOptim(task)
    optm.import_params()
    pair_num, path_num = optm.path_prep(solver_type, arg)
    # print(f"Total pair number: {pair_num}, total path number: {path_num}")

    optm.add_constrs()
    # optm.optimize(ObjType.FEASIBILITY)
    optm.optimize(ObjType.MAX_THROUGHPUT)
    # optm.optimize(ObjType.MIN_LATENCY)

    return optm.model.objVal


if __name__ == "__main__":
    test_QuNetOptim()
