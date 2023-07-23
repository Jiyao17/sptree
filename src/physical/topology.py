
import os
from abc import ABC, abstractmethod

import numpy as np
import networkx as nx


class Topology(ABC):
    def __init__(self) -> None:
        self.nodes: 'set[int]' = set()
        self.edges: 'set[tuple[int]]' = set()
        # adjacency matrix
        self.adjacency: np.ndarray

    @abstractmethod
    def topo_analyze(self, ) -> None:
        pass


class _RealTopo(Topology):
    def __init__(self, filename):
        self.filename = filename

        f = open(filename)
        self._lines = f.readlines()[1:]
        self._lines = [line.strip() for line in self._lines]
        self._lines = [line.split() for line in self._lines]
        self._lines = [[int(line[0]), int(line[1]), float(line[2])] for line in self._lines]

        self.nodes: 'set[int]' = set()
        self.edges: 'set[tuple[int]]' = set()
        # adjacency matrix
        self.adjacency: np.ndarray

        # get the three attributes above
        self.topo_analyze()
    
    def topo_analyze(self):
        """
        Analyze topology:
        get vertices, edges and adjacency matrix.
        """

        # get vertices and edges
        for line in self._lines:
            self.nodes.add(line[0] - 1)
            self.nodes.add(line[1] - 1)

            if (line[1], line[0], line[2]) not in self.edges:
                self.edges.add((line[0] - 1, line[1] - 1, line[2]))

        # get adjacency matrix
        self.adjacency = np.zeros((len(self.nodes), len(self.nodes)))
        for edge in self.edges:
            # edge[0] and edge[1] are vertices
            # edge[2] is the capacity of the edge
            self.adjacency[edge[0], edge[1]] = edge[2]
            self.adjacency[edge[1], edge[0]] = edge[2]


class ATT(_RealTopo):
    def __init__(self, ):
        att_file = 'raw_topo/ATT.txt'
        filename = os.path.join(os.path.dirname(__file__), att_file)
        super().__init__(filename)


class IBM(_RealTopo):
    def __init__(self, ):
        ibm_file = 'raw_topo/IBM.txt'
        filename = os.path.join(os.path.dirname(__file__), ibm_file)
        super().__init__(filename)


class RandomTopo(Topology):
    def __init__(self,):
        super().__init__()

        self.net: nx.Graph = None

    @abstractmethod
    def net_gen(self) -> nx.Graph:
        pass

    def topo_analyze(self,):
        self.nodes = set(self.net.nodes)
        self.edges = set(self.net.edges)

        self.adjacency = np.zeros((len(self.nodes), len(self.nodes)))
        for edge in self.edges:
            self.adjacency[edge[0], edge[1]] = 1
            self.adjacency[edge[1], edge[0]] = 1


class RandomPAG(RandomTopo):
    def __init__(self, n, m):
        super().__init__()

        self.n = n
        self.m = m

        self.net_gen()
        self.topo_analyze()
        
    def net_gen(self):
        self.net: nx.Graph = nx.barabasi_albert_graph(self.n, self.m)




class RandomGNP(RandomTopo):
    def __init__(self, n, p):
        super().__init__()

        self.n = n
        self.p = p

        self.net_gen()
        self.topo_analyze()

    def net_gen(self):
        self.net: nx.Graph = nx.fast_gnp_random_graph(self.n, self.p)
