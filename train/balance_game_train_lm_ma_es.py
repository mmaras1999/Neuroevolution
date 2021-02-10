from games.balance_game import BalanceGame
from lib.neat import Neat
from lib.custom_top_nn import CustomTopologyNeuralNetwork as NN
from lib.lm_ma_es import LM_MA_ES
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

_processes = 8
topology = [(4, sigmoid), (1, sigmoid)]

class BalanceProcess(Process):
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
                self.input_size, self.topology, ind)
                ) for ind in self.population])


lm_ma_es = LM_MA_ES(np.zeros(calc_weight_count(4, topology)), 1)

games = [BalanceGame() for i in range(_processes)]

generation = 0

while True:  
    generation += 1
    print('Generation:', generation)
    population = lm_ma_es.sample()

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
    lm_ma_es.update(population, f_eval)

    if generation % 10 == 0:
        save_obj(lm_ma_es, generation, 'models/lm_ma_es_balance_v2')
        games[0].play(FixedTopologyNeuralNetwork(4, topology, lm_ma_es.sample()[0]), render=True, games_amount=1)

#v1 - 10 random games
#v2 - 25 specific games -> 200 iterations