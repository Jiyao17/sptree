

from copy import deepcopy
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


from src.physical.network import EdgeTuple
import src.physical.quantum as qu


class Node:
    node_id = -1
    @staticmethod
    def new_id():
        Node.node_id += 1
        return Node.node_id

    def __init__(self, edge_tuple: EdgeTuple, fid: qu.FidType,
                parent=None, left=None, right=None, node_id=None) -> None:
        # node info
        if node_id is None:
            node_id = Node.new_id()
        else:
            assert node_id > Node.node_id
            self.node_id = node_id
        self.edge_tuple: EdgeTuple = edge_tuple
        self.fid: qu.FidType = fid

        # tree structure
        self.parent: Node = parent
        self.left: Node = left
        self.right: Node = right

        # optimization info
        self.cost: qu.ExpCostType = 1
        self.grad_f: qu.FidType = 1
        self.grad_cn: qu.ExpCostType = 1
        self.grad_cf: qu.ExpCostType = 1
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
            s += f'f={self.fid:.2f}, c={self.cost:.2f}, '
        # keep 2 decimal places
        else:
            s += f'f={self.fid:.2f}, c={self.cost:.2f}, '

        # s += f'g={self.grad:.5f}, e={self.efficiency:.5f}, a={self.adjust:.5f}, ae={self.adjust_eff:.5f}'

        return s


class Leaf(Node):
    """
    Leaf Node in the SPS Tree
    """

    def __init__(self, edge_tuple: EdgeTuple, fid: qu.FidType,
                    parent: Node,) -> None:
        super().__init__(edge_tuple, fid, parent, None, None)


    def __str__(self) -> str:
        # keep 2 decimal places
        # return f'Leaf ' + super().__str__()
        return "L"


class Branch(Node):
    """
    Branch Node in the SPS Tree
    """

    def __init__(self, edge_tuple: EdgeTuple, fid: qu.FidType, 
                parent: Node, left: Node, right: Node,
                op: qu.OpType, prob: float) -> None:
        super().__init__(edge_tuple, fid, parent, left, right)

        self.op: qu.OpType = op
        self.prob: float = prob

        self.cost = (self.left.cost + self.right.cost) / self.prob
    
    def __str__(self) -> str:
        # keep 2 decimal places
        return f'Branch ' + super().__str__()
        # return self.op.name[0]


class MetaTree(ABC):
    class Shape(Enum):
        BALANCED = 0
        LINKED = 1
        REVERSE_LINKED = 2
        RANDOM = 3
        OPT_ST = 4
    
    @staticmethod
    def grad(node: Node) -> float:
        pass

    def __init__(self, leaves: 'dict[EdgeTuple, float]', op: 'qu.Gate'=qu.GDP) -> None:
        self.leaves = leaves
        self.gate = op

        self.edges = list(self.leaves.keys())
        self.fids = list(self.leaves.values())

        self.root = None

    @abstractmethod
    def build_sst(self, shape=Shape.BALANCED) -> Node:
        pass


class SPST(MetaTree):
    """
    Swap Purification Scheme Tree
    SPST is a binary tree becasue:
    both swap and purification are binary operators
    """

    @staticmethod
    def print_tree(root=None, indent=0):
        root: Node = root # type hinting here
        if root is None:
            return
        print('  ' * indent + str(root))
        SPST.print_tree(root.left, indent + 1)
        SPST.print_tree(root.right, indent + 1)

    @staticmethod
    def copy_subtree(node: Node) -> Node:
        """
        Be aware: all nodes in the copy keep the same node_id
        """
        if node is None:
            return None
        
        parent = node.parent
        node.parent = None
        # new_node = deepcopy(node)
        # node.parent = parent
        # lc = node.left
        # rc = node.right
        
        # prevent infinite recursion
        # node.parent = None
        # node.left = None
        # node.right = None
        new_node = deepcopy(node)
        node.parent = parent
        # node.left = lc
        # node.right = rc

        # new_node.left = SPST.copy_subtree(node.left)
        # new_node.right = SPST.copy_subtree(node.right)
        
        return new_node

    @staticmethod
    def find_max(node: Node, attr: str="adjust_eff", search_range=[Node]) -> Node:
        """
        find the node with max attr in the subtree
        attr: must be a member of Node
        """
        def in_range(node: Node, range: 'list') -> bool:
            for type in range:
                if isinstance(node, type):
                    return True
            return False

        if node is None:
            return None
        if node.is_leaf():
            if in_range(node, search_range):
                return node
            else:
                return None
        
        candidates = []
        if in_range(node, search_range):
            candidates.append(node)
        left = SPST.find_max(node.left, attr, search_range)
        if in_range(left, search_range):
            candidates.append(left)
        right = SPST.find_max(node.right, attr, search_range)
        if in_range(right, search_range):
            candidates.append(right)

        if len(candidates) == 0:
            return None
        
        max_node = candidates[0]
        for node in candidates:
            if getattr(node, attr) > getattr(max_node, attr):
                max_node = node        
        return max_node

        # if getattr(left, attr) > getattr(right, attr):
        #     return left
        # else:
        #     return right

    @staticmethod
    def build_pst(gate: qu.Gate, edge: EdgeTuple, fid: qu.FidType, num: int, 
            shape=MetaTree.Shape.LINKED, cost: qu.ExpCostType=1) -> Node:
        """
        Build the PST
        """
        # sort the edges by fidelity
        # edges = sorted(self.leaves.items(), key=lambda x: x[1], reverse=True)
        def _build_linked(leaves: 'list[Node]') -> Node:
            current_nodes: 'list[Node]' = leaves
            while len(current_nodes) > 1:
                node1, node2 = current_nodes.pop(0), current_nodes.pop(0)
                f, p = gate.purify(node1.fid, node2.fid)
                edge = deepcopy(node1.edge_tuple)
                new_node = Branch(edge, f, None, node1, node2, qu.OpType.PURIFY, p)
                new_node.cost = (node1.cost + node2.cost) / p
                node1.parent = new_node
                node2.parent = new_node
                current_nodes.insert(0, new_node)

            return current_nodes[0]

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
                    f1, f2 = node1.fid, node2.fid
                    f, p = gate.swap(f1, f2)
                    edge = (node1.edge_tuple[0], node2.edge_tuple[1])
                    new_node = Branch(edge, f, None, node1, node2, qu.OpType.PURIFY, p)
                    new_node.cost = (node1.cost + node2.cost) / p
                    node1.parent = new_node
                    node2.parent = new_node
                    next_nodes.append(new_node)
                    # print(f'New Node {new_id} = {node1} + {node2}')
                    # print(f'Fidelity {f} = swap({f1}, {f2})')

                # one node left, add it to the next round directly
                if len(current_nodes) == 1:
                    next_nodes.append(current_nodes.pop())
                
                current_nodes, next_nodes = next_nodes, []

        leaves = [Leaf(edge, fid, None) for i in range(num)]
        [setattr(leaf, 'cost', cost) for leaf in leaves]

        if shape == MetaTree.Shape.LINKED:
            root = _build_linked(leaves)
        elif shape == MetaTree.Shape.BALANCED:
            root = _build_balanced(leaves)
        else:
            raise NotImplementedError('shape not implemented')

        return root

    def __init__(self, leaves: 'dict[EdgeTuple, qu.FidType]', gate: 'qu.Gate'=qu.GDP) -> None:
        super().__init__(leaves, gate)

    def build_sst(self, shape=MetaTree.Shape.BALANCED, costs: 'list[qu.ExpCostType]'=None) -> Node:
        """
        Build the SST with initial leaves
        costs: overide costs is not None
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
                    f1, f2 = node1.fid, node2.fid
                    f, p = self.gate.swap(f1, f2)
                    edge = (node1.edge_tuple[0], node2.edge_tuple[1])
                    new_node = Branch(edge, f, None, node1, node2, qu.OpType.SWAP, p)
                    new_node.cost = (node1.cost + node2.cost) / p
                    node1.parent = new_node
                    node2.parent = new_node
                    next_nodes.append(new_node)
                    # print(f'New Node {new_id} = {node1} + {node2}')
                    # print(f'Fidelity {f} = swap({f1}, {f2})')

                # one node left, add it to the next round directly
                if len(current_nodes) == 1:
                    next_nodes.append(current_nodes.pop())
                
                current_nodes, next_nodes = next_nodes, []

        def _build_linked(leaves: 'list[Node]') -> Node:
            while len(leaves) > 1:
                node1, node2 = leaves.pop(0), leaves.pop(0)
                f, p = self.gate.swap(node1.fid, node2.fid)
                edge = deepcopy(node1.edge_tuple)
                new_node = Branch(edge, f, None, node1, node2, qu.OpType.SWAP, p)
                new_node.cost = (node1.cost + node2.cost) / p
                node1.parent = new_node
                node2.parent = new_node
                leaves.insert(0, new_node)

            return leaves[0]

        def _build_optimal(leaves: 'list[Node]',) -> Node:
            nodes = leaves
            # merge two adjacent nodes round by round
            while len(nodes) > 1:
                # find two adjacent nodes with minimal cost
                min_cost = float('inf')
                min_node_idx = None
                for i in range(len(nodes)-1):
                    cost = nodes[i].cost + nodes[i+1].cost
                    if cost < min_cost:
                        min_cost = cost
                        min_node_idx = i
                # merge the two nodes
                node1, node2 = nodes.pop(min_node_idx), nodes.pop(min_node_idx)
                f, p = self.gate.swap(node1.fid, node2.fid)
                edge = deepcopy(node1.edge_tuple)
                new_node = Branch(edge, f, None, node1, node2, qu.OpType.SWAP, p)
                new_node.cost = (node1.cost + node2.cost) / p
                node1.parent = new_node
                node2.parent = new_node
                nodes.insert(min_node_idx, new_node)

            return nodes[0]

        # sort the edges by fidelity
        # edges = sorted(self.leaves.items(), key=lambda x: x[1], reverse=True)
        leaves = [Leaf(edge, fidelity, None) for edge, fidelity 
                    in zip(self.edges, self.fids)]
        if costs is not None:
            for leaf, cost in zip(leaves, costs):
                leaf.cost = cost

        # shape = MetaTree.Shape.LINKED
        if shape == MetaTree.Shape.BALANCED:
            self.root = _build_balanced(leaves)
        elif shape == MetaTree.Shape.LINKED:
            self.root = _build_linked(leaves)
        elif shape == MetaTree.Shape.OPT_ST:
            self.root = _build_optimal(leaves)
        else:
            raise NotImplementedError('shape not implemented')

        return self.root

    def grad(self, node: Branch, grad_f: qu.FidType=1, 
            grad_cn: qu.ExpCostType=1, grad_cf: qu.ExpCostType=1) -> None:
        """
        Calculate the grads of all descendants, wrt the given node
        self_grad is the grad of the node itself (from its parent)
        """
        def grad_root():
            # f, c = node.fid, node.cost
            # gf, gcn, gcf = self.gate.purify_grad(f, f, c, c, 1)
            
            node.grad_f, node.grad_cn, node.grad_cf = 1, 1, 1
            self.grad(node.left, 1, 1, 1)
            self.grad(node.right, 1, 1, 1)
        
        def grad_branch():
            # calculate the grads of children
            f1, f2 = node.left.fid, node.right.fid
            c1, c2 = node.left.cost, node.right.cost
            if node.op == qu.OpType.SWAP:
                gf1, gcn1, gcf1 = self.gate.swap_grad(f1, f2, 1)
                gf1, gcn1, gcf1 = gf1*grad_f, gcn1*grad_cn, gcf1*grad_cf        
                gf2, gcn2, gcf2 = self.gate.swap_grad(f1, f2, 2)
                gf2, gcn2, gcf2 = gf2*grad_f, gcn2*grad_cn, gcf2*grad_cf
            elif node.op == qu.OpType.PURIFY:
                gf1, gcn1, gcf1 = self.gate.purify_grad(f1, f2, c1, c2, 1) 
                gf1, gcn1, gcf1 = gf1*grad_f, gcn1*grad_cn, gcf1*grad_cf
                gf2, gcn2, gcf2 = self.gate.purify_grad(f1, f2, c1, c2, 2) 
                gf2, gcn2, gcf2 = gf2*grad_f, gcn1*grad_cn, gcf1*grad_cf
            
            # update the grads of children
            node.left.grad_f = gf1
            node.left.grad_cn = gcn1
            node.left.grad_cf = gcf1
            node.right.grad_f = gf2
            node.right.grad_cn = gcn2
            node.right.grad_cf = gcf2
            # recursively update the grads of descendants
            self.grad(node.left, gf1, gcn1, gcf1)
            self.grad(node.right, gf2, gcn2, gcf2)

        if node is None or node.is_leaf():
            return
        
        if node.is_root():
            grad_root()
        else:
            grad_branch()

    def backward(self, node: Branch) -> None:
        """
        update fidelity of all ancestors (not including itself)
        backtrace from node to root
        """
        
        # cannot and shouldn't update fidelity of a leaf node
        assert isinstance(node, Branch), 'Must backtrace from a Branch'

        node = node.parent
        while node is not None:
            if node.op == qu.OpType.SWAP:
                node.fid, node.prob = self.gate.swap(node.left.fid, node.right.fid)
            elif node.op == qu.OpType.PURIFY:
                node.fid, node.prob = self.gate.purify(node.left.fid, node.right.fid)
            node.cost = (node.left.cost + node.right.cost) / node.prob
            
            node = node.parent

    def virtual_purify(self, node: Node) -> 'tuple[qu.FidType, qu.ExpCostType]':
        """
        backtrace the impact of an purification to the root
        return the fidelity and cost of the root
        """
        
        # get f, c after the purification
        old_f, old_c = node.fid, node.cost
        f, p = self.gate.purify(old_f, old_f)
        c = (old_c + old_c) / p
        while node.parent is not None:
            # process node.parent at each iteration
            if node == node.parent.left:
                fl, fr = f, node.parent.right.fid
                cl, cr = c, node.parent.right.cost
            else:
                fl, fr = node.parent.left.fid, f
                cl, cr = node.parent.left.cost, c
            node = node.parent
            assert isinstance(node, Branch), 'Must backtrace from a Branch'
            # -------prepare fl, fr, cl, cr above -------

            if node.op == qu.OpType.SWAP:
                f, p = self.gate.swap(fl, fr)
            elif node.op == qu.OpType.PURIFY:
                f, p = self.gate.purify(fl, fr)
            c = (cl + cr) / p

        return f, c

    def purify(self, node: Node) -> Branch:
        """
        purify a node in the tree
        return the new node (parent of the given node and node's copy)
        """
        parent = node.parent

        copy_node = self.copy_subtree(node)
        f1, f2 = node.fid, copy_node.fid
        fid, prob = self.gate.purify(f1, f2)
        edge = deepcopy(node.edge_tuple)
        new_node = Branch(edge, fid, parent, copy_node, node, qu.OpType.PURIFY, prob)

        copy_node.parent = new_node
        node.parent = new_node
        if parent is not None:
            if parent.left == node:
                parent.left = new_node
            else:
                parent.right = new_node
        else:
            self.root = new_node

        return new_node

    def calc_efficiency(self, node: Node):
        """
        Calculate the efficiency of all descendants, wrt the given node
        """
        if node is None:
            return

        # calculate the efficiency
        node.efficiency = node.grad_f / node.cost
        # calculate adjusted efficiency
        # virtual purify method
        # rf, rc = self.virtual_purify(node)
        # df = rf - self.root.fid
        # dc = rc - self.root.cost
        # grad method
        pf, p = self.gate.purify(node.fid, node.fid)
        df = (pf - node.fid)*node.grad_f
        dc = (pf - node.fid)*node.grad_cf + node.cost*node.grad_cn
        node.adjust_eff = df / dc

        # recursive on descendants
        self.calc_efficiency(node.left)
        self.calc_efficiency(node.right)


class SPT():
    def __init__(self, edges: 'dict[EdgeTuple, qu.FidType]', gate: 'qu.Gate'=qu.GDP) -> None:
        self.edges = edges
        self.gate = gate

        self.root = None

    def st_search(nodes: 'list[Node]'):
        pass

    def pt_search(node: 'Node', cost_bound: qu.ExpCostType,):
        pass



def test_SST():
    from physical.quantum import Gate, EntType, HWParam
    wsys_n = Gate(EntType.WERNER, qu.HWM)
    wsys = Gate(EntType.WERNER)
    dsys = Gate(EntType.BINARY)
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
    from physical.quantum import Gate, EntType, HWParam
    wsys_n = Gate(EntType.WERNER, qu.HWM)
    wsys = Gate(EntType.WERNER)
    dsys = Gate(EntType.BINARY)
    op = dsys
    
    f1, f2, f3, f4 = 0.9, 0.9, 0.9, 0.9

    f12, p12 = op.purify(f1, f2)
    n12 = 2/p12
    f34, p34 = op.purify(f3, f4)
    n34 = 2/p34

    f, p1234 = op.purify(f12, f34)
    n = (n12 + n34) / p1234
    print(f, n)
    n = 0
    n += 1/p1234 * n12 / p12 * (1 + 1)

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
    from physical.quantum import Gate, EntType, HWParam
    import numpy as np

    wsys_n = Gate(EntType.WERNER, qu.HWM)
    wsys = Gate(EntType.WERNER)
    dsys = Gate(EntType.BINARY)
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
    sn1 = 1 / p1234 / p12 * n1
    sn2 = 1 / p1234 / p12 * n2
    sn3 = 1 / p1234 / p34 * n3
    sn4 = 1 / p1234 / p34 * n4
    print(n1, n2, n3, n4)
    print(sn1, sn2, sn3, sn4)
    print(sum([sn1, sn2, sn3, sn4]))


    f13, p13 = op.swap(f1, f3)
    n13 = (n1 + n3)/p13
    f24, p24 = op.swap(f2, f4)
    n24 = (n2 + n4)/p24

    f, p1234 = op.purify(f13, f24)
    C_PSS = (n13 + n24) / p1234
    print(f, C_PSS)

    # assert C_SPP <= C_PSS

def test_grad():
    gate = qu.GDP
    f1, f2, f3, f4 = np.random.rand(4) * 0.25 + 0.7
    n1, n2, n3, n4 = np.round(np.random.rand(4) * 10) + 1

    f12, p12 = gate.purify(f1, f2)
    n12 = (n1 + n2)/p12

    gf, gcn, gcf = gate.purify_grad(f1, f2, n1, n2, 1)

    f11, p11 = gate.purify(f1, f1)
    n11 = (n1 + n1)/p11

    df = (f11 - f1) * gf
    dc = (n11 - n1) * gcn + (f11 - f1) * gcf

    f112, p112 = gate.purify(f11, f2)
    n112 = (n11 + n2)/p112

    dis_f = f112 - (f12 + df)
    dis_c = n112 - (n12 + dc)
    print(dis_f, dis_c)


    
def comments():
    # if node.is_root():
    # pf, p = self.gate.purify(node.fid, node.fid)
    # # c = node.cost*2 / p
    # # df = pf - node.fid
    # # dc = c - self.root.cost
    # df = (pf - node.fid)*node.grad_f
    # dc = (pf - node.fid)*node.grad_cf + node.cost*node.grad_cn
    pass


if __name__ == '__main__':
    # test_SST()
    # test_PST()
    # for i in range(100):
    #     test_SPP_PSS()
    pass
