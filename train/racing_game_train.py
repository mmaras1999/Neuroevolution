from games.racing_game import RacingGame
from lib.lm_ma_es import LM_MA_ES
from lib.custom_top_nn import CustomTopologyNeuralNetwork as NN
from lib.cma_es import CMA_ES_Active
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

_processes = 8
topology = [(6, sigmoid), (2, sigmoid)]

class RacingProcess(Process):
    def __init__(self, id, output, population, input_size, topology):
        super().__init__()
        self.id = id
        self.population = population
        self.input_size = input_size
        self.output = output
        self.topology = topology

    def run(self):
        # print('running {0}'.format(self.id))
        self.output[self.id] = -np.array([games[self.id].play(
            FixedTopologyNeuralNetwork(
                self.input_size, self.topology, ind),
            map_id=3) for ind in self.population])


lm_ma_es = LM_MA_ES(np.zeros(calc_weight_count(6, topology)), 1, popsize=32)
# lm_ma_es = load_obj(2900, 'models/maes_racing_v1')

games = [RacingGame() for i in range(_processes)]

generation = 0

while True:  
    generation += 1
    print('Generation:', generation)
    population = lm_ma_es.sample()

    chunks = np.array_split(population, _processes)
    results = Manager().list([None] * _processes)
    threads = [RacingProcess(i, results, chunks[i], 6, topology) for i in range(_processes)]

    for th in threads:
        th.start()
    
    for th in threads:
        th.join()

    f_eval = np.concatenate(tuple(results))
    print(f_eval)
    print(f_eval.mean(), f_eval.max(), f_eval.min())
    lm_ma_es.update(population, f_eval)

    if generation % 10 == 0:
        save_obj(lm_ma_es, generation, 'models/maes_racing_v2')
        print('score:', games[0].play(FixedTopologyNeuralNetwork(6, topology, lm_ma_es.sample()[0]), render=True, map_id=3))

#v1 - 10 random games
#v2 - 25 specific games -> 200 iterations
