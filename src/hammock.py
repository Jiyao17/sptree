
from network import QuNet, StaticPath

class Hammock:
    def __init__(self, net: QuNet):
        self.net = net

    def weave(self, path: StaticPath, distance: int, width: int):
        """
        Weave the hammock into the network
        - path: the path to weave
        - distance: distance between two nodes on the path
        - width: number of sub-hammocks
        """
        
        assert distance > 0
        assert width > 0
        assert distance < len(path)
        
        pairs = []
        for i in range(len(path) - distance):
            pairs.append((path[i], path[i+distance]))

        for i in range(width):
            