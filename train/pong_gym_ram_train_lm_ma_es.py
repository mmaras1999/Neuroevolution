from games.pong_gym_ram import PongGame
from lib.lm_ma_es import LM_MA_ES
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

_processes = 8
topology = [(64, sigmoid), (32, sigmoid), (1, sigmoid)]

class pongThread(Process):
    def __init__(self, id, output, population, input_size, topology):
        super().__init__()
        self.id = id
        self.population = population
        self.input_size = input_size
        self.topology = topology
        self.output = output

    def run(self):
        game = PongGame()
        print('running process {0}'.format(self.id))
        self.output[self.id] = -np.array([game.play(
            FixedTopologyNeuralNetwork(
                self.input_size, self.topology, ind)
                ) for ind in self.population])

#lm_ma_es = LM_MA_ES(np.zeros(calc_weight_count(128, topology)), 1.0)
lm_ma_es = load_obj(15900, 'models/pong_gym_ram/lm_ma_es_pong_ram')

generation = 15900

while True:   
    generation += 1
    print('Generation:', generation)
    population = lm_ma_es.sample()

    chunks = np.array_split(population, _processes)
    results = Manager().list([None] * _processes)
    threads = [pongThread(i, results, chunks[i], 128, topology) for i in range(_processes)]

    for th in threads:
        th.start()
    
    for th in threads:
        th.join()

    f_eval = np.concatenate(tuple(results))
    print(f_eval)
    print(f_eval.mean())
    lm_ma_es.update(population, f_eval)
 
    if generation % 100 == 0:
        save_obj(lm_ma_es, generation, 'models/pong_gym_ram/lm_ma_es_pong_ram')
 