
from enum import Enum
import copy
from typing import NewType

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from .topology import _RealTopo, ATT, IBM
import physical.quantum as qu


# A Quantum Overlay Network (QON) is a directed multi-graph (MultiDiGraph), therefore:
# each node is uniquely defined by node_id (int)
# each edge is uniquely defined by (src, dst)
# each node/edge has an associated object (Node/Edge)
# that contains all the information about the node/edge
NodeID = NewType('NodeID', int)
KeyID = NewType('KeyID', int)
NodePair = NewType('NodePair', tuple[NodeID, NodeID])
EdgeTuple = NewType('EdgeTuple', tuple[NodeID, NodeID])
MultiEdgeTuple = NewType('MultiEdgeTuple', tuple[NodeID, NodeID, KeyID])
StaticPath = NewType('StaticPath', tuple[EdgeTuple])
Path = NewType('Path', list[EdgeTuple])



class NodeType(Enum):
    BUFFERED = 1
    BUFFLESS = 2

    REPEATER = 100


class Node:
    def __init__(self, node_id: NodeID, node_type: NodeType,):
        self.node_id = node_id
        self.node_type = node_type

    def __str__(self):
        return str(self.node_type) + str(self.node_id)


class BufflessNode(Node):
    def __init__(self, node_id: NodeID):
        super().__init__(node_id, NodeType.BUFFLESS)


class BufferedNode(Node):
    def __init__(self, node_id: NodeID, storage: int=0):
        super().__init__(node_id, NodeType.BUFFERED)

        self.storage = storage


class Edge:
    def __init__(self, src_id: NodeID, dst_id: NodeID, 
            fidelity: float=1.0, capacity: int=1400,
            ):
        self.src_node = src_id
        self.dst_node = dst_id
        self.fid = fidelity
        self.capacity = capacity
        self.edge_tuple: EdgeTuple = (src_id, dst_id)
        
    def __str__(self):
        desc = f'Edge {self.edge_tuple}: '
        desc += f'(fid, cap) = ({self.fid}, capacity={self.capacity}) '
        return desc


class QuNet:
    """
    Basic Quantum Network (QN) class
    """

    @staticmethod
    def draw(net: nx.Graph, filename=None):
        # save the graph to file
        # or show it on screen if filename == None
        pos = nx.spring_layout(net)


        nx.draw(net, pos, with_labels=True)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    @staticmethod
    def disjoint_paths(net: nx.Graph, src: NodeID, dst: NodeID, path_num: int=5):
        """
        Find at most k disjoint paths from src to dst
        If no enough paths, return all paths found
        """
        paths: list[StaticPath] = []
        # make a deep copy of the graph when processing each user pair 
        # to find disjoint paths
        tnet = copy.deepcopy(net)
        for i in range(path_num):
            try:
                path_nodes: list[NodeID] = nx.shortest_path(tnet, src, dst)
            except nx.NetworkXNoPath:
                break

            path = []
            # remove the edges in the path
            for i in range(len(path_nodes)-1):
                edge_tuple = (path_nodes[i], path_nodes[i+1])
                path.append(edge_tuple)
                tnet.remove_edge(edge_tuple[0], edge_tuple[1])

            paths.append(tuple(path))

        return paths

    def __init__(self, topology: _RealTopo=ATT(), gate: qu.Gate=qu.GDP):
        self.topology = topology
        self.gate = gate

        # real network, without virtual edges among QMs
        self.net = nx.Graph()
        self.nodes = list(topology.nodes)
        self.adjacency = topology.adjacency

    def net_gen(self,
                node_memory=100,
                edge_capacity=(26, 35),
                edge_fidelity=(0.7, 0.95),
        ):
        """
        Generate the real network according to
        topology and given parameters
        storage: storage capacity of each nodes
        capacity: capacity of each edge, must have same shape as adjacency
        fidelity: fidelity of each edge, must have same shape as adjacency
        """

        for i, node_id in enumerate(self.nodes):
            obj=BufferedNode(node_id, node_memory)
            self.net.add_node(node_id, obj=obj)
        
        for i, edge in enumerate(self.topology.edges):
            cap = np.random.randint(edge_capacity[0], edge_capacity[1])
            fid = np.random.uniform(edge_fidelity[0], edge_fidelity[1])
            key = 0
            edge_tuple = (edge[0], edge[1])
            obj = Edge(*edge_tuple, fid, cap)
            self.net.add_edge(edge[0], edge[1], obj=obj)


class Task:
    def __init__(self, net):
        pass
        

class QuNetTask:
    def __init__(self, qunet: QuNet, ):
        self.qunet = qunet
        self.net = qunet.net
        
        self.user_pairs: 'list[NodePair]' = []
        # all real paths between user pairs
        self.up_paths: 'dict[NodePair, list[StaticPath]]' = {}
        # set in workload_gen()
        self.workload: 'dict[NodePair, int]' = {}
        self.fid_req: 'dict[NodePair, float]' = {}

    def set_user_pairs(self, pair_num=6, method='random'):
        # all possible edge user pairs
        user_pairs: list[NodePair] = []
        EUs = self.qunet.nodes
        for i in range(len(EUs)):
            for j in range(i+1, len(EUs)):
                user_pairs.append((EUs[i], EUs[j]))
            
        if pair_num > len(user_pairs):
            raise ValueError('pair_num must be <= the number of all possible user pairs')

        # selected user pairs
        if method == 'random':
            up_indices = np.random.choice(len(user_pairs), pair_num, replace=False)
            self.user_pairs = [user_pairs[idx] for idx in up_indices]

    def set_up_paths(self, path_num=3):
        """
        find disjoint (by edge) real & virtual paths for each user pair
        """
        for user_pair in self.user_pairs:
            paths = QuNet.disjoint_paths(self.qunet.net, *user_pair, path_num)
            self.up_paths[user_pair] = []
            for path in paths:
                self.up_paths[user_pair].append(path)
        
    def workload_gen(self, request_range=(100, 100), fid_range=(0.8, 0.8)):
        
        for i, user_pair in enumerate(self.user_pairs):
            # generate a random load & fid requirement for each user pair
            if request_range[0] == request_range[1]:
                self.workload[(user_pair)] = request_range[0]
            else:
                self.workload[(user_pair)] = np.random.randint(*request_range)

            if fid_range[0] == fid_range[1]:
                self.fid_req[(user_pair)] = fid_range[0]
            else:
                self.fid_req[(user_pair)] = np.random.uniform(*fid_range)


def test_QuNet():
    # draw an example graph

    np.random.seed(0)

    qunet = QuNet(topology=ATT(),)
    qunet.net_gen()
    
    task = QuNetTask(qunet)
    task.set_user_pairs(5)
    print("All user pairs: \n", task.user_pairs)
    task.set_up_paths()
    # print("All rpaths between selected user pairs: \n", task.up_paths)
    task.workload_gen()



    QuNet.draw(qunet.net, "qunet.png")


if __name__ == "__main__":
    
    test_QuNet()
    