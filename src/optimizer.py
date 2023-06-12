
import copy
from abc import ABC, abstractmethod

import gurobipy as gp
from gurobipy import GRB

import numpy as np

from network import NodeID, NodePair, EdgeTuple, StaticPath
from network import BufferedNode, Edge, QuNetTask, QuNet
from quantum import EntType, MeasureAccu, Operation


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
    def __init__(self, task: QuNetTask,) -> None:
        self.task = copy.deepcopy(task)
        self.qunet = copy.deepcopy(task.qunet)
        
        self.params = {}
        self.model = gp.Model("QuNetOptim")

    def import_params(self) -> None:
        self.params = {}

        # all edges, [EdgeTuple]
        self.params['E']: 'list[EdgeTuple]' = []
        # capacity of each edge, {e: capacity}
        self.params['EC']: 'dict[EdgeTuple, int]' = {}
        for (u, v, obj) in self.task.qunet.net.edges.data('obj'):
            edge_tuple: EdgeTuple = (u, v, 0)
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

    def path_prep(self,) -> 'tuple[int, int]':
        # path allocations
        self.params['A']: 'dict[tuple[NodePair, StaticPath, EdgeTuple], int]' = {}
        self.A = self.params['A']

        total_pair_num = 0
        total_path_num = 0
        for k in self.K:
            for p in self.P[k]:
                fids = {}
                for e in p:
                    edge: Edge = self.task.qunet.net.edges[e[:2]]['obj']
                    fids[e] = edge.fidelity
                    
                # print(f"fid: {fids}")
                fth = self.F[k]
                allocs = [1] * len(p)
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
                        edge_capacity = self.EC[(e[1], e[0], 0)]
                    self.model.addConstr(
                        gp.quicksum(
                            self.x[k, path] * self.A[k, path, e] for e in path) 
                            <= edge_capacity,
                        name=f"EC_{k}_{path}_{i}")
                    
        # memory capacity
        for node in self.N:
            mem_usage = 0
            adj_edges = self.task.qunet.net.edges(node, data=True)
            for k in self.K:
                for path in self.P[k]:
                    for i, e in enumerate(path):
                        if e[:2] in adj_edges or (e[1], e[0]) in adj_edges:
                            mem_usage += self.x[k, path] * self.A[k, path, e]
            self.model.addConstr(
                mem_usage <= self.NC[node],
                name=f"NC_{node}")

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
    

def test_QuNetOptim():
    np.random.seed(0)

    qunet = QuNet()
    qunet.net_gen(node_memory=100, edge_capacity=(100, 101), edge_fidelity=(0.9, 1.0))
    
    task = QuNetTask(qunet)
    task.set_user_pairs(10)
    # print("All user pairs: \n", task.user_pairs)
    task.set_up_paths(5)
    # print("All rpaths between selected user pairs: \n", task.up_paths)
    task.workload_gen(fid_range=(0.75, 0.75), request_range=(100, 100))
    # QuNet.draw(qunet.net, "qunet.png")



    optm = QuNetOptim(task)
    optm.import_params()
    # pair_num, path_num = optm.path_prep(Solver=GRDYSolver, system_type=SystemType.DEPOLARIZED)
    # pair_num, path_num = optm.path_prep(Solver=OSPSDP, system_type=SystemType.DEPOLARIZED)
    # pair_num, path_num = optm.path_prep(Solver=EPPSolver, system_type=SystemType.DEPHASED)
    pair_num, path_num = optm.path_prep()
    print(f"Total pair number: {pair_num}, total path number: {path_num}")

    optm.add_constrs()
    # optm.optimize(ObjType.FEASIBILITY)
    optm.optimize(ObjType.MAX_THROUGHPUT)
    # optm.optimize(ObjType.MIN_LATENCY)


if __name__ == '__main__':

    np.random.seed(0)

    test_QuNetOptim()






