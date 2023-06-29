

from copy import deepcopy
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


from physical.network import EdgeTuple
import physical.quantum as qu


class Node:
    node_id = -1
    @staticmethod
    def new_id():
        Node.node_id += 1
        return Node.node_id

    def __init__(self, edge_tuple: EdgeTuple, fidelity: float,
                parent=None, left=None, right=None, node_id=None) -> None:
        # node info
        if node_id is None:
            node_id = Node.new_id()
        else:
            assert node_id > Node.node_id
            self.node_id = node_id
        self.edge_tuple: EdgeTuple = edge_tuple
        self.fidelity: float = fidelity

        # tree structure
        self.parent: Node = parent
        self.left: Node = left
        self.right: Node = right

        # optimization info
        self.cost: float = 1
        self.grad: float = 1
        self.efficiency: float = 1
        self.adjust: float = 1
        self.adjust_eff: float = 1
        self.test_rank_attr: float = 1

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def is_root(self) -> bool:
        return self.parent is None

    def __str__(self) -> str:
        s = f'{self.node_id} {self.edge_tuple}: '
        if self.is_root():
            s += f'f={self.fidelity}, '
        # keep 2 decimal places
        else:
            s += f'f={self.fidelity:.2f}, '

        # s += f'g={self.grad:.5f}, e={self.efficiency:.5f}, a={self.adjust:.5f}, ae={self.adjust_eff:.5f}'

        return s


class Leaf(Node):
    """
    Leaf Node in the SPS Tree
    """

    def __init__(self, edge_tuple: EdgeTuple, fidelity: float,
                    parent: Node,) -> None:
        super().__init__(edge_tuple, fidelity, parent, None, None)


    def __str__(self) -> str:
        # keep 2 decimal places
        # return f'Leaf ' + super().__str__()
        return "L"


class Branch(Node):
    """
    Branch Node in the SPS Tree
    """

    def __init__(self, edge_tuple: EdgeTuple, fidelity: float, 
                parent: Node, left: Node, right: Node,
                op: qu.OpType.SWAP) -> None:
        super().__init__(edge_tuple, fidelity, parent, left, right)

        self.op: qu.OpType = op
    
    def __str__(self) -> str:
        # keep 2 decimal places
        # return f'Branch ' + super().__str__()
        return self.op.name[0]


class MetaTree(ABC):
    class Shape(Enum):
        BALANCED = 0
        LINKED = 1
        REVERSE_LINKED = 2
        RANDOM = 3
    
    @staticmethod
    def grad(node: Node) -> float:
        pass

    def __init__(self, leaves: 'dict[EdgeTuple, float]', op: 'qu.Operation'=qu.DOPP) -> None:
        self.leaves = leaves
        self.op = op

        self.edges = list(self.leaves.keys())
        self.fids = list(self.leaves.values())

        self.root = None

    @abstractmethod
    def build_tree(self, shape=Shape.BALANCED) -> Node:
        pass


class SPST(MetaTree):
    """
    Swap Purification Scheme Tree
    SPST is a binary tree becasue:
    both swap and purification are binary operators
    """

    def __init__(self, leaves: 'dict[EdgeTuple, float]', op: 'qu.Operation'=qu.DOPP) -> None:
        super().__init__(leaves, op)

        self.root = self.build_tree()

    def build_tree(self, shape=MetaTree.Shape.BALANCED) -> Node:
        """
        Build the SPST
        """
        def _build_balanced(leaves: 'list[Node]') -> Node:
            current_nodes: 'list[Node]' = leaves
            next_nodes: 'list[Node]' = []
            # merge nodes round by round
            while len(current_nodes) >= 1:
                if len(current_nodes) == 1:
                    return current_nodes.pop()
                
                # merge nodes in current round
                while len(current_nodes) >= 2:
                    node1, node2 = current_nodes.pop(0), current_nodes.pop(0)
                    f1, f2 = node1.fidelity, node2.fidelity
                    f, p = self.op.swap(f1, f2)
                    edge = (node1.edge_tuple[0], node2.edge_tuple[1])
                    new_node = Branch(edge, f, None, node1, node2, qu.OpType.SWAP)
                    node1.parent = new_node
                    node2.parent = new_node
                    next_nodes.append(new_node)
                    # print(f'New Node {new_id} = {node1} + {node2}')
                    # print(f'Fidelity {f} = swap({f1}, {f2})')

                # one node left, add it to the next round directly
                if len(current_nodes) == 1:
                    next_nodes.append(current_nodes.pop())
                
                current_nodes, next_nodes = next_nodes, []

        # sort the edges by fidelity
        # edges = sorted(self.leaves.items(), key=lambda x: x[1], reverse=True)
        leaves = [Leaf(edge, fidelity, None) for edge, fidelity 
                    in zip(self.edges, self.fids)]

        if shape == MetaTree.Shape.BALANCED:
            root = _build_balanced(leaves)
        else:
            raise NotImplementedError('shape not implemented')

        return root


def test_SST():
    from physical.quantum import Operation, EntType, HW
    wsys_n = Operation(EntType.WERNER, qu.HWM)
    wsys = Operation(EntType.WERNER)
    dsys = Operation(EntType.DEPHASED)
    op = wsys_n
    
    f1, f2, f3, f4 = 0.9, 0.9, 0.9, 0.9

    f12, p12 = op.swap(f1, f2)
    n12 = 2/p12
    f34, p34 = op.swap(f3, f4)
    n34 = 2/p34

    f, p1234 = op.swap(f12, f34)
    n = (n12 + n34) / p1234
    print(f, n)

    f12, p12 = op.swap(f1, f2)
    n12 = 2/p12
    f123, p123 = op.swap(f12, f3)
    n123 = (n12 + 1) / p123
    f, p = op.swap(f123, f4)
    n = (n123 + 1) / p
    print(f, n)


    ns = 0
    p = op.hw.prob_swap
    simu = 100
    import numpy as np
    ns = np.zeros((simu, simu))
    for m in range(0, simu):
        for n in range(0, simu):
            ns[m, n] = (1-p12)**m*p12 * (1-p34)**n*p34 * ((m+1)*2 + (n+1)*2)
    print(sum(ns.flatten()) / p1234)

    en = (2/p12 + 2/p34) / p1234
    print(en)

def test_PST():
    from physical.quantum import Operation, EntType, HW
    wsys_n = Operation(EntType.WERNER, qu.HWM)
    wsys = Operation(EntType.WERNER)
    dsys = Operation(EntType.DEPHASED)
    op = dsys
    
    f1, f2, f3, f4 = 0.9, 0.9, 0.9, 0.9

    f12, p12 = op.purify(f1, f2)
    n12 = 2/p12
    f34, p34 = op.purify(f3, f4)
    n34 = 2/p34

    f, p1234 = op.purify(f12, f34)
    n = (n12 + n34) / p1234
    print(f, n)

    f12, p12 = op.purify(f1, f2)
    n12 = 2/p12
    f123, p123 = op.purify(f12, f3)
    n123 = (n12 + 1) / p123
    f, p = op.purify(f123, f4)
    n = (n123 + 1) / p
    print(f, n)

    p = op.hw.prob_swap
    simu = 100
    import numpy as np
    ns = np.zeros((simu, simu))
    for m in range(0, simu):
        for n in range(0, simu):
            ns[m, n] = (1-p12)**m*p12 * (1-p34)**n*p34 * ((m+1)*2 + (n+1)*2)
    print(sum(ns.flatten()) / p1234)

    en = (2/p12 + 2/p34) / p1234
    print(en)

    f = np.prod([f1, f2, f3, f4])/(np.prod([f1, f2, f3, f4]) + np.prod([1-f1, 1-f2, 1-f3, 1-f4]))
    print(f)

def test_SPP_PSS():
    from physical.quantum import Operation, EntType, HW
    import numpy as np

    wsys_n = Operation(EntType.WERNER, qu.HWM)
    wsys = Operation(EntType.WERNER)
    dsys = Operation(EntType.DEPHASED)
    op = wsys
    
    f1, f2, f3, f4 = np.random.rand(4) * 0.5 + 0.5
    n1, n2, n3, n4 = np.round(np.random.rand(4) * 10) + 1

    f12, p12 = op.purify(f1, f2)
    n12 = (n1 + n2)/p12
    f34, p34 = op.purify(f3, f4)
    n34 = (n3 + n4)/p34

    f, p1234 = op.swap(f12, f34)
    C_SPP = (n12 + n34) / p1234
    print(f, C_SPP)

    f13, p13 = op.swap(f1, f3)
    n13 = (n1 + n3)/p13
    f24, p24 = op.swap(f2, f4)
    n24 = (n2 + n4)/p24

    f, p1234 = op.purify(f13, f24)
    C_PSS = (n13 + n24) / p1234
    print(f, C_PSS)

    # assert C_SPP <= C_PSS

if __name__ == '__main__':
    # test_SST()
    test_PST()
    # for i in range(100):
    #     test_SPP_PSS()
    
