from enum import IntEnum
from lib.activator_funcs import sigmoid_3

class CustomTopologyNeuralNetwork:
    class NodeType(IntEnum):
        INPUT = 1
        BIAS = 2
        HIDDEN = 3
        OUTPUT = 4
    
    class Node:
        def __init__(self, id, type):
            self.type = type
            self.id = id

            if type == CustomTopologyNeuralNetwork.NodeType.BIAS:
                self.score = 1
        def __repr__(self):
            return f"node {self.id} : {self.type.name}"

    class Link:
        def __init__(self, nodeFrom, nodeTo, weight):
            self.nodeFrom = nodeFrom
            self.nodeTo = nodeTo
            self.weight = weight
        
        def __repr__(self):
            return f"link {self.nodeFrom} -> {self.nodeTo}; {self.weight}"

    #nodes must have type and id fields, link must have nodeFrom, nodeTo and weight fields.
    #the order of nodes determines the input is passed (first input node in list will take first input value and so on)
    #the same is with output
    def __init__(self, nodes, links, activation_function=sigmoid_3):
        self.nodes = {}
        self.outputs = []
        self.inputs = []
        self.linksTo = {}
        for node in nodes:
            self.nodes[node.id] = CustomTopologyNeuralNetwork.Node(node.id, node.type)
            self.linksTo[node.id] = []

            if node.type == CustomTopologyNeuralNetwork.NodeType.OUTPUT:
                self.outputs.append(node.id)
            
            if node.type == CustomTopologyNeuralNetwork.NodeType.INPUT:
                self.inputs.append(node.id)

        for link in links:
            assert link.nodeTo in self.nodes, f"destination node of link doesn't exist: {link}"
            assert link.nodeFrom in self.nodes, f"source node of link doesn't exist: {link}"
            assert self.nodes[link.nodeTo].type >= CustomTopologyNeuralNetwork.NodeType.HIDDEN, f"destination node must be hidden or output: {link}"
            assert self.nodes[link.nodeFrom].type < CustomTopologyNeuralNetwork.NodeType.OUTPUT, f"source node must not be output: {link}"

            self.linksTo[link.nodeTo].append(CustomTopologyNeuralNetwork.Link(link.nodeFrom, link.nodeTo, link.weight))
        print(self.nodes)
        print(self.linksTo)

        self.act_fun = activation_function

        self.preCalcOrder()

    def preCalcOrder(self):
        self.nodeOrder = []
        self.visited = {}
        self.onStack = {}

        for nodeId in self.outputs:
            self.calcOrderRecursive(nodeId)

        del self.visited
        del self.onStack

    def calcOrderRecursive(self, nodeId):
        assert nodeId not in self.onStack, "the network has cycle"
            
        if nodeId in self.visited:
            return
        self.visited[nodeId] = True
        self.onStack[nodeId] = True

        for link in self.linksTo[nodeId]:
            self.calcOrderRecursive(link.nodeFrom)
        
        del self.onStack[nodeId]

        if self.nodes[nodeId].type >= CustomTopologyNeuralNetwork.NodeType.HIDDEN:
            self.nodeOrder.append(nodeId)


    def eval(self, input):
        results = []
        for i in range(len(input)):
            self.nodes[self.inputs[i]].score = input[i]
        
        for node_id in self.nodeOrder:
            score = 0
            for link in self.linksTo[node_id]:
                score += link.weight * self.nodes[link.nodeFrom].score
            self.nodes[node_id].score = self.act_fun(score)
        
        for node_id in self.outputs:
            results.append(self.nodes[node_id].score)
        
        return results


nodes = [CustomTopologyNeuralNetwork.Node(6, CustomTopologyNeuralNetwork.NodeType.INPUT),
         CustomTopologyNeuralNetwork.Node(3, CustomTopologyNeuralNetwork.NodeType.OUTPUT),
         CustomTopologyNeuralNetwork.Node(1, CustomTopologyNeuralNetwork.NodeType.HIDDEN),
         CustomTopologyNeuralNetwork.Node(5, CustomTopologyNeuralNetwork.NodeType.BIAS),
         CustomTopologyNeuralNetwork.Node(7, CustomTopologyNeuralNetwork.NodeType.HIDDEN),
         CustomTopologyNeuralNetwork.Node(4, CustomTopologyNeuralNetwork.NodeType.OUTPUT)]

links = [CustomTopologyNeuralNetwork.Link(6, 7, 1.5),
         CustomTopologyNeuralNetwork.Link(7, 1, 3),
         CustomTopologyNeuralNetwork.Link(5, 1, 2),
         CustomTopologyNeuralNetwork.Link(6, 3, 3),
         CustomTopologyNeuralNetwork.Link(6, 1, 0.5),
         CustomTopologyNeuralNetwork.Link(5, 4, -2),
         CustomTopologyNeuralNetwork.Link(1, 4, -3),
         CustomTopologyNeuralNetwork.Link(1, 3, 0)]
