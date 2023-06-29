
from network.network import QuNet, QuNetTask, EdgeTuple
from solver.spst import GRDYSolver
from physical.quantum import Operation


class Hammock:
    def __init__(self, net: QuNet):
        self.net = net

    def weave(self, edges: 'dict[EdgeTuple, float]', frag_len: int, width: int):
        """
        Weave the hammock into the network
        - path: the path to weave
        - distance: distance between two nodes on the path
        - width: number of sub-hammocks
        """
        # stop resurrsion
        if frag_len <= 1:
            return
        if width <= 0:
            return
        
        solver = GRDYSolver(edges, Operation())
        f, allocs = solver.solve(fth=0.9)
        
        edges = list(edges.keys())
        fids = list(edges.values())
        length = len(edges)
        assert frag_len < length


        
        frag_num = len(fids) // frag_len
        rems = len(fids) % frag_len
        fragments = []
        for i in range(frag_num):
            fragments.append(fids[i * frag_len: (i + 1) * frag_len])
        if rems > 0:
            fragments.append(fids[-rems:])

        # recursive weaving
        for i in range(frag_num):
            self.weave(fragments[i], frag_len - 1, width - 1)
        if rems > 0:
            self.weave(fragments[-1], rems - 1, width - 1)


if __name__ == '__main__':
    net = QuNet()
    net.net_gen()
    task = QuNetTask(net)
    task.set_user_pairs(6)
    task.set_up_paths(3)
    task.workload_gen()
    hammock = Hammock(net)
    hammock.weave()