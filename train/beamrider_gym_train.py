from games.beamrider_gym import BeamRiderGame
from lib.cma_es import CMA_ES_Active
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

_processes = 8
topology = [(64, sigmoid), (4, sigmoid)]

class beamriderThread(Process):
    def __init__(self, id, output, population, input_size, topology):
        super().__init__()
        self.id = id
        self.population = population
        self.input_size = input_size
        self.topology = topology
        self.output = output

    def run(self):
        print('running {0}'.format(self.id))
        self.output[self.id] = -np.array([games[self.id].play(
            FixedTopologyNeuralNetwork(
                self.input_size, self.topology, ind)
                ) for ind in self.population])

cma_es = CMA_ES_Active(np.zeros(calc_weight_count(128, topology)), 1.0)
#cma_es = load_obj(1500, 'models/cmaes_pong_v8')
    
games = [BeamRiderGame() for i in range(_processes)]

generation = 0

while not cma_es.terminate():   
    generation += 1
    print('Generation:', generation)
    population = cma_es.sample()

    chunks = np.array_split(population, _processes)
    results = Manager().list([None] * _processes)
    threads = [beamriderThread(i, results, chunks[i], 128, topology) for i in range(_processes)]

    for th in threads:
        th.start()
    
    for th in threads:
        th.join()

    f_eval = np.concatenate(tuple(results))
    print(f_eval)
    print(f_eval.mean())
    cma_es.update(population, f_eval)
 

    if generation % 10 == 0:
        save_obj(cma_es, generation, 'models/beamrider/cmaes_v0')

