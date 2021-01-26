from games.game_2048 import Game2048
from lib.neat import Neat
from lib.activator_funcs import sigmoid
from lib.fixed_top_nn import FixedTopologyNeuralNetwork
from lib.utilities import save_obj, load_obj, calc_weight_count
from multiprocessing import Process, Manager
import numpy as np
import sys
import os

_processes = 8

class game2048Thread(Process):
    def __init__(self, id, output, population, input_size):
        super().__init__()
        self.id = id
        self.population = population
        self.input_size = input_size
        self.output = output

    def run(self):
        print('running {0}'.format(self.id))
        self.output[self.id] = np.array([games[self.id].play(
            ind) for ind in self.population])

neat = Neat(16, 4)
#lm_ma_es = load_obj(1500, 'models/beamrider/lm_ma_es_v1')
    
games = [Game2048() for i in range(_processes)]

generation = 0

while True:   
    generation += 1
    print('Generation:', generation)
    population  = neat.getFenotypes()

    chunks = np.array_split(population, _processes)
    results = Manager().list([None] * _processes)
    threads = [game2048Thread(i, results, chunks[i], 16) for i in range(_processes)]

    for th in threads:
        th.start()
    
    for th in threads:
        th.join()

    f_eval = np.concatenate(tuple(results))
    print(f_eval)
    print(f_eval.mean())
    neat.update(f_eval)

    if generation % 100 == 0:
        save_obj(neat, generation, 'models/game2048/neat_v1')

