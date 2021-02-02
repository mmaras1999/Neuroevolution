from games.balance_game import BalanceGame
from lib.neat import Neat
from lib.custom_top_nn import CustomTopologyNeuralNetwork as NN
from lib.cma_es import CMA_ES_Active
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

_processes = 4
topology = [(4, sigmoid), (1, sigmoid)]

class BalanceProcess(Process):
    def __init__(self, id, output, population, input_size, topology):
        super().__init__()
        self.id = id
        self.population = population
        self.input_size = input_size
        self.output = output
        self.topology = topology
        self.game = BalanceGame()

    def run(self):
        # print('running {0}'.format(self.id))
        self.output[self.id] = -np.array([self.game.play(
            FixedTopologyNeuralNetwork(
                self.input_size, self.topology, ind)
                ) for ind in self.population])


cma_es = CMA_ES_Active(np.zeros(calc_weight_count(4, topology)), 1, popsize=32)
game = BalanceGame()
generation = 0

while True:  
    generation += 1
    print('Generation:', generation)
    population = cma_es.sample()

    chunks = np.array_split(population, _processes)
    results = Manager().list([None] * _processes)
    threads = [BalanceProcess(i, results, chunks[i], 4, topology) for i in range(_processes)]

    for th in threads:
        th.start()
    
    for th in threads:
        th.join()

    f_eval = np.concatenate(tuple(results))
    print(f_eval)
    print(f_eval.mean(), f_eval.max(), f_eval.min())
    cma_es.update(population, f_eval)

    if generation % 20 == 0:
        save_obj(cma_es, generation, 'models/cmaes_balance_v3')
        game.play(FixedTopologyNeuralNetwork(4, topology, cma_es.sample()[0]), render=True, games_amount=1)

#v1 - 10 random games
#v2 - 25 specific games -> 200 iterations
#v3 same as #v25 but games are created in threads
