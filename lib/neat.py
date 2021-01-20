import numpy as np
import copy
from lib.custom_top_nn import CustomTopologyNeuralNetwork as NN
import operator

class Neat:
    class Node(NN.Node):
        def __init__(self, id, type, depth):
            super().__init__(id, type)
            self.depth = depth 
        
        def __repr__(self):
            return super().__repr__() + f": d = {self.depth}"


    class Link(NN.Link):
        def __init__(self, nodeFrom, nodeTo, weight, innovationNumber):
            super().__init__(nodeFrom, nodeTo, weight)
            self.id = innovationNumber
            self.disabled = False

        def __repr__(self):
            return super().__repr__() + f"; id = {self.id}; off = {self.disabled}"


    class Genotype:
        def __init__(self, nodesGens, linksGens):
            self.nodesGens = {}
            for node in nodesGens:
                self.nodesGens[node.id] = node
            
            self.linksGens = linksGens
        
        def __repr__(self):
            return f"nodes: {self.nodesGens} \nlinks: {self.linksGens}"
        
        def getInNodesIds(self):
            return [x for x in self.nodesGens.keys() if self.nodesGens[x].type >= NN.NodeType.HIDDEN]

        def getOutNodesIds(self):
            return [x for x in self.nodesGens.keys() if self.nodesGens[x].type != NN.NodeType.OUTPUT]


        def sortByDepth(self, id1, id2):
            if self.nodesGens[id1].depth < self.nodesGens[id2].depth:
                return id1, id2
            elif self.nodesGens[id1].depth > self.nodesGens[id2].depth:
                return id2, id1
            elif id1 < id2:
                return id1, id2
            else:
                return id2, id1            
        
        def getLink(self, nodeFrom, nodeTo):
            for link in self.linksGens:
                if link.nodeFrom == nodeFrom and link.nodeTo == nodeTo:
                    return link
            return None

        def sortLinks(self):
            self.linksGens.sort(key=operator.attrgetter('id'))

    class Species:
        def __init__(self, leader):
            self.population = [leader]
            self.leader = copy.deepcopy(leader)
            self.stagnationTime = 0
            self.bestScore = leader.fitness

        def add(self, genotype):
            self.population.append(genotype)
            if genotype.fitness > self.bestScore:
                self.bestScore = genotype.fitness
                self.stagnationTime = 0

        def updateLeader(self):
            self.stagnationTime += 1
            self.leader = copy.deepcopy(self.population[0])
            for genotype in self.population:
                if genotype.fitness > self.leader.fitness:
                    self.leader = copy.deepcopy(genotype)

        def getLeader(self):
            return self.leader

        def calcSharedFitness(self):
            self.totalSharedFitness = 0
            for genotype in self.population:
                genotype.sharedFitness = genotype.fitness / len(self.population)
                self.totalSharedFitness += genotype.sharedFitness
        
        def keepBestGenotypes(self, fraction):
            self.population.sort(reverse=True, key=operator.attrgetter('fitness'))
            bestSize = int(round(len(self.population) * fraction))
            self.population = self.population[:bestSize]


    def __init__(self, inputs, outputs, popsize=150):
        self.popsize = popsize
        
        self.genotypes = []
        self.fenotypes = []
        for _ in range(popsize):
            nodes, links = self.generateSampleNetwork(inputs, outputs)
            self.genotypes.append(Neat.Genotype(nodes, links))
            self.fenotypes.append(NN(nodes, links))

        self.bestgens = []

        self.species = []

        self.innovationCounter = 0
        self.innovationDict = {}

        self.nodeCounter = outputs + inputs + 1
        self.nodesDict = {}

        self.ADD_NODE_PBB = 0.03
        self.ADD_LINK_PBB_SMALL = 0.05
        self.ADD_LINK_PBB_BIG = 0.3
        self.BIG_POPULATION_TRESHOLD = int(popsize * 0.2)
        self.SMALL_POPULATION_TRESHOLD = 5

        self.WEIGHT_MUTATION_PBB = 0.7
        self.WEIGHT_PERTUBATION_PBB = 0.9
        self.PERTUBATION_RANGE = 0.5
        self.REACTIVE_LINK_PBB = 0.25
        self.CROSSOVER_PBB = 0.75
        self.C1 = 1
        self.C2 = 1
        self.C3 = 0.3
        self.SPECIES_TRESHOLD = 0.75
        self.MAX_STAGNANT_TIME = 15
        self.MAX_ADD_LINK_TRIES = 15

        self.BEST_INDIVIDUAL_COUNT = int(popsize * 0.75) 
        self.BEST_POPULATION_FRACTION = 0.75 #how many best indiviuals will be used for mating

    
    def generateSampleNetwork(self, inputs, outputs):
        inputNodes = []
        outputNodes = []
        links = []

        for i in range(inputs):
            inputNodes.append(Neat.Node(i, NN.NodeType.INPUT, 0))
        inputNodes.append(Neat.Node(inputs, NN.NodeType.BIAS, 0))

        for i in range(outputs):
            outputNodes.append(Neat.Node(inputs + i + 1, NN.NodeType.OUTPUT, 1))
        
        for inNode in inputNodes:
            for outNode in outputNodes:
                links.append(Neat.Link(inNode.id, outNode.id, np.random.normal(), -1))
        
        return inputNodes + outputNodes, links

    def getDistance(self, gen1, gen2):
        gen1.sortLinks()
        gen2.sortLinks()

        numDisjont = 0
        numMatching = 0
        numExcess = 0
        weightsDiff = 0

        i = 0
        j = 0
        while i < len(gen1.linksGens) and j < len(gen2.linksGens):
            if gen1.linksGens[i].id < gen2.linksGens[j].id:
                numDisjont += 1
                i += 1
            elif gen1.linksGens[i].id > gen2.linksGens[j].id:
                numDisjont += 1
                j += 1
            elif gen1.linksGens[i].disabled and gen2.linksGens[j].disabled:
                i += 1
                j += 1
            elif gen1.linksGens[i].disabled or gen2.linksGens[j].disabled:
                i += 1
                j += 1
                numDisjont += 1
            else:
                numMatching += 1
                if abs(gen1.linksGens[i].weight) < 1:
                    weightsDiff += abs(gen1.linksGens[i].weight - gen2.linksGens[j].weight)
                else:
                    weightsDiff += abs(gen1.linksGens[i].weight - gen2.linksGens[j].weight) / abs(gen1.linksGens[i].weight)
                i += 1
                j += 1

        numExcess = len(gen1.linksGens) - i + len(gen2.linksGens) - j
        N = max(len(gen1.linksGens), len(gen2.linksGens))

        return  self.C1 * numExcess / N  + self.C2 * numDisjont / N + self.C3 * weightsDiff / numMatching

    def getInnovationNumber(self, nodeFrom, nodeTo):
        if (nodeFrom, nodeTo) not in self.innovationDict:
            self.innovationDict[(nodeFrom, nodeTo)] = self.innovationCounter
            self.innovationCounter += 1
        return self.innovationDict[(nodeFrom, nodeTo)]

    def getNewNodeId(self, genotype, nodeFrom, nodeTo):
        if (nodeFrom, nodeTo) not in self.nodesDict:
            self.nodesDict[(nodeFrom, nodeTo)] = [self.nodeCounter]
            self.nodeCounter += 1
        
        ids = self.nodesDict[(nodeFrom, nodeTo)]
        for id in ids:
            if id not in genotype.nodesGens:
                return id
        
        ids.append(self.nodeCounter)
        self.nodeCounter += 1
        return ids[-1]

    def getFenotypes(self):
        return self.fenotypes

    def addNode(self, genotype):
        if np.random.uniform() > self.ADD_NODE_PBB:
            return
        
        linkToDivide = None
        while not linkToDivide:
            linkId = np.random.randint(len(genotype.linksGens))
            link = genotype.linksGens[linkId]
            if link.disabled or genotype.nodesGens[link.nodeFrom].type == NN.NodeType.BIAS:
                continue
            linkToDivide = link
        
        linkToDivide.disabled = True
        id1 = linkToDivide.nodeFrom
        id2 = linkToDivide.nodeTo
        newNodeId = self.getNewNodeId(genotype, id1, id2)
        depth = (genotype.nodesGens[id1].depth + genotype.nodesGens[id2].depth) / 2
        
        genotype.nodesGens[newNodeId] = Neat.Node(newNodeId, NN.NodeType.HIDDEN, depth)
        genotype.linksGens.append(Neat.Link(id1, newNodeId, 1, self.getInnovationNumber(id1, newNodeId)))
        genotype.linksGens.append(Neat.Link(newNodeId, id2, linkToDivide.weight, self.getInnovationNumber(newNodeId, id2)))

    def addLink(self, genotype, bigPopulation=False):
        if bigPopulation and np.random.uniform() > self.ADD_LINK_PBB_BIG:
            return
        if not bigPopulation and np.random.uniform() > self.ADD_LINK_PBB_SMALL:
            return
        
        node1Id = -1
        node2Id = -1

        inNodeIds = genotype.getInNodesIds()
        outNodeIds = genotype.getOutNodesIds()
        for _ in range(self.MAX_ADD_LINK_TRIES):
            id1 = np.random.choice(outNodeIds)
            id2 = np.random.choice(inNodeIds)
            if id1 == id2 or genotype.nodesGens[id1].depth == genotype.nodesGens[id2].depth:
                continue
            id1, id2 = genotype.sortByDepth(id1, id2)
            if not genotype.getLink(id1, id2):
                node1Id = id1
                node2Id = id2
                break
        if node1Id == -1: #cannot find place for link
            return
        
        innovationNumber = self.getInnovationNumber(node1Id, node2Id)
        genotype.linksGens.append(Neat.Link(node1Id, node2Id, np.random.normal(), innovationNumber))

    def mutateWeights(self, genotype):
        if np.random.uniform() > self.WEIGHT_MUTATION_PBB:
            return
        if np.random.uniform() <= self.WEIGHT_PERTUBATION_PBB:
            for link in genotype.linksGens:
                if abs(link.weight) > 1:
                    link.weight *= 1 - np.random.uniform(-self.PERTUBATION_RANGE, self.PERTUBATION_RANGE)
                else:
                    link.weight += np.random.uniform(-self.PERTUBATION_RANGE, self.PERTUBATION_RANGE)

        else:
            for link in genotype.linksGens:
                link.weight = np.random.normal()

    def mutateDisabled(self, genotype):
        for link in genotype.linksGens:
            if link.disabled and np.random.uniform() < self.REACTIVE_LINK_PBB:
                link.disabled = False

    def mutate(self, genotype, bigPopulation=False):
        self.addLink(genotype, bigPopulation)
        self.addNode(genotype)
        self.mutateWeights(genotype)
        self.mutateDisabled(genotype)

    def crossover(self, par1, par2):
        child = None
        if np.random.uniform() > self.CROSSOVER_PBB:
            if np.random.uniform() > 0.5:
                child = copy.deepcopy(par1)
            else:
                child = copy.deepcopy(par2)
            return child
        

        #swap parents so par1 is the one with better fitness value
        if (par1.fitness < par2.fitness or
            (par1.fitness == par2.fitness and len(par2.linksGens) < len(par1.linksGens)) or
            (par1.fitness == par2.fitness and len(par2.linksGens) <  len(par1.linksGens) and np.random.uniform() < 0.5)):
            par1, par2 = par2, par1
        
        par1.sortLinks()
        par2.sortLinks()

        links = []
        nodes = copy.deepcopy(list(par1.nodesGens.values()))

        i = 0
        j = 0
        while i < len(par1.linksGens):
            if j == len(par2.linksGens) or par1.linksGens[i].id < par2.linksGens[j].id:
                links.append(copy.deepcopy(par1.linksGens[i]))
                i += 1
            elif par1.linksGens[i].id > par2.linksGens[j].id:
                j += 1
            elif np.random.uniform() < 0.5:
                links.append(copy.deepcopy(par1.linksGens[i]))
                i += 1
                j += 1
            else:
                links.append(copy.deepcopy(par2.linksGens[j]))
                i += 1
                j += 1

        child = Neat.Genotype(nodes, links)
        return child

    #the input is fitness score for each fenotype
    def update(self, scores, verbose=False):
        for species in self.species:
            species.population = []

        self.nodesDict = {}
        self.innovationDict = {}


        #save best genotype
        best_id = np.argmax(scores)
        self.bestgens.append((self.genotypes[best_id],scores[best_id], self.fenotypes[best_id]))

        # best = np.argsort(-scores)[:self.BEST_INDIVIDUAL_COUNT]
        # copied = self.genotypes
        # self.genotypes = []
        # for i in best:
        #     self.genotypes.append(copied[i])
        # scores = scores[best]

        for i in range(len(scores)):
            self.genotypes[i].fitness = scores[i]
            added = False
            for species in self.species:
                if self.getDistance(species.getLeader(), self.genotypes[i]) <= self.SPECIES_TRESHOLD:
                    species.add(self.genotypes[i])
                    added = True
            
            if not added:
                self.species.append(Neat.Species(self.genotypes[i]))
        
        self.species = [species for species in self.species if len(species.population) > 0]
        if verbose:
            print("we have this many species:", len(self.species))

        for species in self.species:
            species.calcSharedFitness()
            species.updateLeader()
        
        #remove stagnant species
        if len(self.species) > 2:
            self.species.sort(reverse=True, key=lambda x: x.bestScore)

            #always take two best species
            self.species = self.species[:2] + [species for species in self.species[2:] if species.stagnationTime <= self.MAX_STAGNANT_TIME]

        if verbose:
            print("and that many survied:", len(self.species))
        meanSharedFitness = 0
        for species in self.species:
            meanSharedFitness += species.totalSharedFitness
        meanSharedFitness /= self.popsize

        if verbose:
            print("the mean fitness is", meanSharedFitness)
        newPopSize = 0
        for species in self.species:    
            species.childrenCount = int(round(species.totalSharedFitness / meanSharedFitness))
            newPopSize += species.childrenCount
        
        if verbose:
            print("and the new pop size before fixing is:", newPopSize)
        while newPopSize > self.popsize:
            newPopSize -= 1
            id = np.random.choice(len(self.species))
            self.species[id].childrenCount -= 1

        while newPopSize < self.popsize:
            newPopSize += 1
            id = np.random.choice(len(self.species))
            self.species[id].childrenCount += 1

        if verbose:
            print("new population size fixed")
        newGenotypes = []
        
        for species in self.species:
            needToCopyBest = False
            n = len(species.population)
            isBigPopulation = n > self.BIG_POPULATION_TRESHOLD
            probs = np.ones(n) / n
            if n > self.SMALL_POPULATION_TRESHOLD:
                species.keepBestGenotypes(self.BEST_POPULATION_FRACTION)
                needToCopyBest = True
                probs = np.array([genotype.fitness for genotype in species.population])
                probs -= probs.min()
                if (probs > 0).sum() < 2:
                    probs = np.ones(n) / n
                else:
                    # probs += 0.1 / n
                    probs /= probs.sum()

            for i in range(species.childrenCount):
                if needToCopyBest:
                    newGenotypes.append(copy.deepcopy(species.population[0]))
                    needToCopyBest = False
                    continue
                
                if n == 1:
                    child = copy.deepcopy(species.population[0])
                    self.mutate(child, isBigPopulation)
                    newGenotypes.append(child)
                else:
                    id1, id2 = np.random.choice(len(species.population), 2, False, p=probs)
                    child = self.crossover(species.population[id1], species.population[id2])
                    self.mutate(child, isBigPopulation) 
                    newGenotypes.append(child)


        self.genotypes = newGenotypes
        self.fenotypes = []
        for genotype in newGenotypes:
            self.fenotypes.append(NN(genotype.nodesGens.values(), [link for link in genotype.linksGens if not link.disabled]))



        

